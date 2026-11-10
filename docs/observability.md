# Observability

GAI is instrumented with OpenTelemetry. Agent runs, workflows, model calls, and
loop tool executions produce spans, a small set of low-cardinality metrics
covers latency, token usage, retries, and run length, and an optional sink
receives structured lifecycle observations. Spans and metrics are no-ops until
the application installs OpenTelemetry providers, and sensitive content stays
out of telemetry unless it is explicitly enabled.

## Traces

GAI creates OpenTelemetry spans across agent runs, workflows, model attempts, and tools. Safe metadata is recorded by default; prompts, completions, reasoning, tool arguments, tool results, and metadata values are not exported unless the application deliberately enables them.

## Observation sink

Set `Definition.ObservationSink` for structured lifecycle observations. Observation events and attributes use the `observation.*` OpenTelemetry namespace. Install any OpenTelemetry-compatible exporter before creating agents.

## Metrics

GAI also records OpenTelemetry metrics through the global `MeterProvider`: model call duration and token usage, tool execution duration, loop retries, and loop run duration and iteration count. The model and tool metrics follow the GenAI semantic conventions (`gen_ai.client.inference.*`, `gen_ai.execute_tool.duration`). Metric attributes are limited to low-cardinality values such as provider, model, tool name, outcome, and a bounded `error.type`; IDs and content are never attached. With the default no-op provider nothing is recorded. See [runtime metrics](metrics.md) for every metric, attribute, and error value.

## Langfuse

The optional `observability/langfuse` package provides a batched OTLP tracer provider for Langfuse:

```go
provider, err := langfuse.NewTracerProvider(ctx, langfuse.Config{
  Endpoint:    "https://cloud.langfuse.com/api/public/otel",
  PublicKey:   os.Getenv("LANGFUSE_PUBLIC_KEY"),
  SecretKey:   os.Getenv("LANGFUSE_SECRET_KEY"),
  ServiceName: "my-agent-service",
})
if err != nil {
  return err
}
defer func() {
  if err := provider.Shutdown(context.Background()); err != nil {
    log.Printf("flush Langfuse traces: %v", err)
  }
}()
otel.SetTracerProvider(provider)
```

The caller owns provider shutdown so the final batch can be flushed and any export error can be observed.

## Trace context

Trace-wide filtering dimensions are opt-in and separate from `RunInput.Meta`:

```go
traceContext := &gai.TraceContext{
  Name:        "lace-chat",
  UserID:      userID,
  SessionID:   conversationID,
  Tags:        []string{"mobile"},
  Release:     version,
  Environment: "production",
  Metadata: map[string]string{
    "feature": "chat",
  },
}

workflow, err := assistant.NewRun(ctx, agent.RunInput{
  TraceContext: traceContext,
  Prompt:       input,
  Meta:         applicationData, // values are still never exported implicitly
})
```

`RunInput.TraceContext == nil` inherits trace dimensions already installed with
`gai.WithTraceContext`; a non-nil value replaces them as a complete set. GAI
copies and validates the value before asynchronous work starts. Scalar values
are limited to 200 bytes, tags to 32 values of 200 bytes, and selected metadata
to 32 entries with 64-byte alphanumeric keys and 200-byte values. Invalid or
oversized entries are omitted instead of truncated. Trace context uses an internal Go
context value, not OpenTelemetry baggage, so GAI does not add user or session
identifiers to outbound request headers.

Environment values are trimmed and normalized to lowercase. They may contain
letters, digits, hyphens, and underscores; values beginning with `langfuse` are
reserved by Langfuse and omitted.

The core library records provider-neutral `gai.trace.*` attributes. The
Langfuse tracer provider maps them to Langfuse v4 trace attributes on every
span. Applications assembling their own Langfuse tracer provider can wrap a
span processor with `langfuse.NewTraceContextSpanProcessor` to use the same
mapping.

## Model call spans

Every OpenAI, Mistral, Gemini, and Anthropic model invocation also creates one
`chat {requested_model}` client span. Streaming spans remain open until the
stream ends or is canceled. They use the same `gen_ai.*` contract for requested
and resolved models, finish reasons, request IDs, reported token usage, and
provider HTTP status. On failure, `error.type` is the same bounded class the
inference metrics use, such as `rate_limited` or `timeout`. Langfuse generation type, model parameters, usage
details, and completion-start time are included where known. Prompts,
completions, reasoning, tool arguments, and raw provider payloads are not
attached to these spans.

The older `ai.provider`, `ai.model`, `ai.max_tokens`, `ai.input_tokens`,
`ai.output_tokens`, `ai.reasoning_tokens`, and `ai.tool_call_count` attributes
remain during the semantic-attribute migration. New integrations should use
the `gen_ai.*` attributes.

## Tool spans

Every tool invocation executed through `Loop` creates one `loop.tool` child
span. Direct calls to `CallTool` are not instrumented.

| Attribute | Value |
| --- | --- |
| `loop.operation` | `tool` |
| `gen_ai.operation.name` | `execute_tool` |
| `gen_ai.tool.name`, `gen_ai.tool.call.id` | Stable tool name and model-issued call ID |
| `gen_ai.tool.type` | `function` |
| `langfuse.observation.type` | `tool` |
| `gai.tool.outcome` | `success`, `tool_error`, `processing_error`, `denied`, `approval_required`, `invalid_call`, `panic`, `deadline`, or `cancellation` |
| `error.type` | Fixed `gai.tool.*` classification on failures |
| `tool.name`, `tool.call_id`, `tool.status` | Legacy compatibility attributes |

Span errors use fixed, low-cardinality classifications; raw tool errors and
panic values are never exported implicitly. `Loop` contains handler and processor
panics as terminal `ErrToolPanic` errors, cancels siblings, and joins admitted work.
Direct `CallTool` invocations retain ordinary Go panic behavior.

## Content capture

Tool arguments and results are absent by default. Enable them independently
with the same request-scoped capture policy used for other sensitive content:

```go
ctx = gai.WithContentCapturePolicy(ctx, gai.ContentCapturePolicy{
  ToolInput:  gai.CaptureEnabled,
  ToolOutput: gai.CaptureEnabled,
  MaxBytes:   4096,
  Redact: func(ctx context.Context, kind gai.ContentKind, value []byte) ([]byte, error) {
    return redactApplicationSecrets(value), nil
  },
})
```

Captured input is written to `gen_ai.tool.call.arguments` and
`langfuse.observation.input`; captured output is written to
`langfuse.observation.output`, and successful output is also written to
`gen_ai.tool.call.result`. The legacy `tool.input` and `tool.output` attributes
remain during migration and carry the redaction, truncation, and byte-count
metadata. Content is captured once and reused for every applicable alias.
Valid JSON remains serialized JSON; invalid JSON is retained as bounded UTF-8
text. Redactor errors and panics fail closed.
