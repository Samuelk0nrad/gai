# Runtime metrics

GAI records a small set of OpenTelemetry metrics for model calls, tool
executions, retries, and loop runs. They answer aggregate operational questions
(latency, token volume, retry rate, tool failure rate, run length) without
querying individual traces.

Metrics are optional. GAI records them through the global OpenTelemetry
`MeterProvider`, so with the default no-op provider nothing is exported and no
setup is required. GAI does not ship a metrics backend or exporter.

## Enabling metrics

Install any OpenTelemetry metrics SDK provider before running models or loops.
For example, with the OTLP HTTP exporter:

```go
import (
  "go.opentelemetry.io/otel"
  "go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp"
  sdkmetric "go.opentelemetry.io/otel/sdk/metric"
)

exporter, err := otlpmetrichttp.New(ctx)
if err != nil {
  return err
}
provider := sdkmetric.NewMeterProvider(
  sdkmetric.WithReader(sdkmetric.NewPeriodicReader(exporter)),
)
defer func() { _ = provider.Shutdown(context.Background()) }()
otel.SetMeterProvider(provider)
```

GAI binds its instruments to whichever provider is currently installed, so a
provider set after GAI was first used still receives data.

Instrumentation scopes are `github.com/lace-ai/gai/ai` for model calls and
`github.com/lace-ai/gai/loop` for tools, retries, and runs.

## Metrics

| Metric | Instrument | Unit | Recorded |
|---|---|---|---|
| `gen_ai.client.inference.duration` | Histogram | `s` | Once per provider model call, when the call or stream ends |
| `gen_ai.client.inference.usage.input_tokens` | Counter | `{token}` | Once per provider model call that reported input usage |
| `gen_ai.client.inference.usage.output_tokens` | Counter | `{token}` | Once per provider model call that reported output usage |
| `gen_ai.execute_tool.duration` | Histogram | `s` | Once per invoked tool execution |
| `gai.loop.retries` | Counter | `{retry}` | Once per model attempt the loop schedules for retry |
| `gai.loop.run.duration` | Histogram | `s` | Once per loop run, when the run ends |
| `gai.loop.run.iterations` | Histogram | `{iteration}` | Once per loop run, when the run ends |

The `gen_ai.*` metrics follow the OpenTelemetry
[GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai)
for client inference, token usage, and tool execution. Those conventions are
still in Development status and may change; GAI is pre-v1 and will follow them.
The `gai.*` metrics cover loop concepts the conventions do not define.

Histograms use the explicit bucket boundaries recommended by the conventions:
`0.01` to `81.92` seconds for inference and tool durations, `0.1` seconds to
one day for run durations, and `1` to `128` for iteration counts.

### Model calls

Every OpenAI, Anthropic, Gemini, and Mistral call records the inference metrics
from the same finalized generation outcome used for the `chat` span. Each retry
attempt is a separate call, so every failed attempt is recorded with
`error.type`.

Attributes:

| Attribute | Metrics | Values |
|---|---|---|
| `gen_ai.operation.name` | all inference metrics | `chat` |
| `gen_ai.provider.name` | all inference metrics | `openai`, `anthropic`, `gcp.gemini`, `mistral_ai` |
| `gen_ai.request.model` | all inference metrics | Configured model name, when set |
| `gen_ai.response.model` | all inference metrics | Provider-reported model, when known |
| `gen_ai.token.modality` | usage counters | Always `unknown`; GAI's normalized usage is not split by modality |
| `error.type` | duration, only on failure | See below |

Inference `error.type` values:

| Value | Meaning |
|---|---|
| `rate_limited`, `transient`, `authentication`, `invalid_request`, `unsupported`, `unknown` | The provider error's `ai.ProviderErrorKind` |
| `unsuccessful_generation` | The provider ended in an unsuccessful terminal state, such as a token limit or safety block |
| `timeout` | A context deadline expired |
| `canceled` | The context was canceled |
| `_OTHER` | Any other error |

Token usage uses GAI's normalized `ai.Usage`:

- Input tokens include cached tokens (cache reads and writes).
- Output tokens include reasoning tokens.
- When the provider reports no usage, no token measurement is recorded. Zero
  counts are not recorded either, because usage counters accept only positive
  contributions.
- Failed calls still record any usage the provider reported.

### Tools

`gen_ai.execute_tool.duration` measures how long the tool function ran,
excluding result processing. Only invoked tools are recorded. Calls that never
reach a tool (unknown tool names, malformed arguments, policy denials, pending
approvals) have no execution time and are not recorded; they remain visible in
the loop's tool events. Because only registered tools can be invoked, tool
names in metrics are always names the application registered, never names a
model invented.

| Attribute | Values |
|---|---|
| `gen_ai.tool.name` | Registered tool name |
| `gen_ai.tool.type` | `function` |
| `error.type` | On failure only: `gai.tool.tool_error`, `gai.tool.invalid_call`, `gai.tool.deadline`, `gai.tool.cancellation`, `gai.tool.panic`, `gai.tool.processing_error` |

### Retries and runs

`gai.loop.retries` counts accepted retries.

| Attribute | Values |
|---|---|
| `gai.retry.reason` | A known `ai.ProviderErrorKind` value, `attempt_timeout`, or `_OTHER` |
| `gen_ai.request.model` | Loop model name, when the model reports one |

`gai.loop.run.duration` and `gai.loop.run.iterations` are recorded for every
loop run, including loops started by agents and history summarization. The
iteration count is the number of iterations the run started.

| Attribute | Values |
|---|---|
| `gai.loop.outcome` | `success`, `error`, `canceled` |
| `gen_ai.request.model` | Loop model name, when the model reports one |
| `error.type` | For `error` and `canceled` runs, see below |

Run `error.type` values: `max_retries`, `max_iterations`, `attempt_timeout`,
`tool_panic`, `tool_result_processing`, `tool_policy`, `tool_approval`,
`terminal_tool_batch`, `build_prompt`, `request_budget_exceeded`,
`unsuccessful_generation`, `invalid_configuration`, the known
`ai.ProviderErrorKind` values, `timeout`, `canceled`, and `_OTHER`.

## Cardinality and privacy

Metric attributes are limited to the bounded values listed above. Model and
tool names are application-controlled; keep them to a stable, small set if you
generate them dynamically.

GAI never adds request IDs, response IDs, tool call IDs, run or trace IDs, user
or session identifiers, `gai.TraceContext` values, prompts, completions, tool
arguments, tool results, raw error messages, retry ordinals, or retry delays to
metrics. Those remain available in traces and observations, subject to the
content-capture policy.

## Common questions

| Question | Query |
|---|---|
| Model latency | Percentiles of `gen_ai.client.inference.duration`, by `gen_ai.request.model` |
| Model error rate | Count of `gen_ai.client.inference.duration` with `error.type` ÷ total count |
| Token volume | Rate of the `gen_ai.client.inference.usage.*` counters |
| Retry rate | Rate of `gai.loop.retries` ÷ rate of `gen_ai.client.inference.duration` count |
| Tool latency and failure rate | `gen_ai.execute_tool.duration` percentiles, and count with `error.type` ÷ total count, by `gen_ai.tool.name` |
| Run length | `gai.loop.run.duration` and `gai.loop.run.iterations`, by `gai.loop.outcome` |
