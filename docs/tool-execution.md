# Tool execution

`loop` owns the execution contract. `agent` exposes it through reusable
`Definition` defaults and `RunInput.Execution` overrides. Provider-neutral tool
schemas and calls remain in `ai`; prompt rendering uses `context.ToolSignature`.

## Register a handler

```go
lookup, err := loop.NewTool("lookup", "Find a record", ai.ToolParameters{
    Properties: []ai.ToolParameter{
        {Name: "id", Type: ai.ToolParameterString, Required: true},
    },
}, func(ctx context.Context, call ai.ToolCall) (string, error) {
    var args struct { ID string `json:"id"` }
    if err := loop.DecodeToolArgs(call, &args); err != nil {
        return "", err
    }
    return store.LookupJSON(ctx, args.ID) // application-owned dependency
})
if err != nil { return err }
lookup, err = loop.WithToolOptions(lookup, loop.ToolOptions{
    Traits: loop.ToolTraits{Effect: loop.ToolEffectReadOnly, Idempotent: true},
})
if err != nil { return err }
```

A struct can implement `loop.Tool` directly. `Function` receives an isolated
`ai.ToolCall` value, including copied argument bytes and provider extensions.
An empty string is a valid success. An error takes precedence over returned text
and becomes a model-visible tool error. Handlers are shared dependencies and must
be safe for the concurrency their registration allows. `NewTool` copies the
schema; custom tool implementations own the stability of their name and schema.
Traits are application assertions, not inferred guarantees. Idempotency is
metadata only: GAI never automatically retries tool invocations.

## Finish with a terminal presentation batch

Mark application-owned presentation tools as terminal when their successful
execution is the complete primary response. Terminal behavior is trusted
registration metadata; it is never inferred from an empty result, a tool name,
or model-supplied arguments.

```go
displayText, err := loop.NewTool("display_text", "Append response text",
    ai.ToolParameters{}, func(ctx context.Context, call ai.ToolCall) (string, error) {
        // Decode and validate call.Args, then append one application response part.
        return "", response.AppendText(ctx, call.Args)
    })
if err != nil { return err }
displayText, err = loop.WithToolOptions(displayText, loop.ToolOptions{
    Terminal: true,
})
if err != nil { return err }

displayProducts, err := loop.NewTool("display_products", "Append product cards",
    ai.ToolParameters{}, func(ctx context.Context, call ai.ToolCall) (string, error) {
        // Product validation and rendering remain application responsibilities.
        return "", response.AppendProducts(ctx, call.Args)
    })
if err != nil { return err }
displayProducts, err = loop.WithToolOptions(displayProducts, loop.ToolOptions{
    Terminal: true,
})
if err != nil { return err }

presentationTools := []loop.Tool{displayText, displayProducts}
worker := agent.New(agent.Definition{
    Model:         model,
    Prompt:        promptFactory,
    Tools:         presentationTools,
    ToolExecution: loop.ToolExecutionConfig{MaxConcurrent: 1},
})

// Low-level users register the same tools directly.
runner := loop.New(model, presentationTools, promptBuilder, nil)
runner.ToolExecution.MaxConcurrent = 1
```

When one generation contains only registered terminal calls, GAI validates the
whole batch, applies the normal policy and approval checks, executes every call,
processes every result, and joins all started work. It then accepts the iteration
and completes the primary loop without another model generation. Empty successful
handler results are valid. The same registrations work in `agent.Definition.Tools`
and per-run `ExecutionOverrides.Tools`.

Canonical tool results always retain the model's requested call order, even when
independent handlers finish out of order. If handler side effects must have that
same order across different tool names, set `MaxConcurrent: 1` as above.
`Serial` orders repeated calls to the same name only; a shared `*ToolGuard`
provides mutual exclusion across registrations and runs, not a cross-run ordering
contract.

A generation that mixes registered terminal and ordinary calls is rejected before
policy, approval, or handler callbacks. A terminal candidate also cannot succeed
when any call is malformed, unknown, excluded, denied, unapproved, failed,
canceled, or rejected by result processing. Eligible siblings may already have
run; GAI joins started pipelines before returning and does not roll back, replay,
or automatically retry calls. Terminal-batch failure does not trigger another
model generation.

Terminal completion ends only the current loop/primary stage. Workflow middleware
continues normally. Usage, events, execution diagnostics, call/result identity,
and canonical conversation remain available, but terminal result strings are not
promoted to visible answer text. Applications own response rendering and may
therefore complete successfully with an empty `AgentResult.Text`.

## Configure execution and authorization

```go
policy, err := loop.NewToolPolicy(loop.ToolPolicyRules{
    Allow: []string{"lookup"},
    RequireApproval: []string{"update_record"},
    Deny: []string{"delete_database"},
})
if err != nil { return err }
limit, err := loop.LimitToolResultBytes(16 * 1024)
if err != nil { return err }

worker := agent.New(agent.Definition{
    Model: model,
    Prompt: promptFactory,
    Tools: []loop.Tool{lookup, updateRecord},
    ToolExecution: loop.ToolExecutionConfig{
        MaxConcurrent: 4,
        DefaultTimeout: 10 * time.Second,
    },
    ToolPolicy: policy,
    ToolApprovalResolver: approvals,
    ToolResultProcessor: limit,
})
```

The examples assume application-owned `model`, `promptFactory`, `updateRecord`,
`store`, and `approvals` dependencies. The complete
[order support example](../examples/order-support) demonstrates a read-only setup.

| Setting | Default and behavior |
| --- | --- |
| `MaxConcurrent` | Zero permits all eligible calls in a batch; positive values bound active handler/processor pipelines. Negative values are invalid. |
| `DefaultTimeout` | Zero adds no handler deadline. Parent deadlines still apply. Negative values are invalid. |
| `ToolOptions.Timeout` | Nil inherits; a pointer to zero disables the default. The duration is copied. |
| `ToolOptions.Serial` | Preserves same-name FIFO within a run, including result processing. It does not serialize other names or runs. |
| `ToolOptions.Terminal` | False preserves ordinary result-to-model behavior. True allows a successful terminal-only generation to finish the primary loop without another model request. |
| `ToolOptions.Guard` | The exact shared `*ToolGuard` serializes invoked pipelines across runs. Synthetic refusals bypass it. Zero value is ready; never copy a used guard. It is an in-process gate. |
| `ToolPolicy` | Nil permits registered calls. A decision is allow, deny, or require approval. |
| `ToolApprovalResolver` | Nil refuses approval-required calls with `ErrToolApprovalRequired`; it never implies consent. |
| `ToolResultProcessor` | Nil publishes the normalized handler result. |

Scheduling does not put guard waiters or serially blocked calls in worker slots.
Independent eligible calls can proceed. Queue and approval waits do not consume
the handler timeout. Deadlines are cooperative: the handler's returned result is
authoritative. A returned deadline error is a timeout; a completed successful
result remains successful even if the deadline has elapsed. Parent cancellation
still terminates the run and joins admitted work. Serial/guard ownership and
concurrency capacity extend through result processing; the handler deadline does
not limit the processor. Refused or malformed calls bypass shared guards because
no handler uses that resource; their processors still obey the run's concurrency
limit and same-name Serial order.
Processors receive the run context and must honor its cancellation. Results enter
the model transcript in original call order, even if completion events arrive in
a different order. A batch with more than 65,534 distinct guards for admitted handler candidates is
rejected before dispatch to stay within Go's wait-set limit; repeated calls using
the same guard share one wait registration.

`ToolChoice` requests model tool selection; it is not an authorization boundary.
Use `ToolPolicy` to enforce permissions when executing calls.

Policy precedence is deny, then require approval (by name or `ApprovalEffects`),
then allow, then default. An omitted default becomes deny when an allowlist is
present, otherwise allow. Set `Default` explicitly when that distinction is
unwanted. Effect rules trust registered traits. `ChainToolPolicies` uses the most
restrictive decision and preserves the first decision at equal priority; each
policy receives its own call snapshot. Invalid decisions and policy failures
terminate the batch before any handler runs. Trusted caller identity can be
carried in `context.Context`; model arguments must not authorize themselves.

All policies run before approval resolution, and all approvals finish before
worker admission for that batch. A refusal is a per-call tool result; approved
and otherwise allowed calls may still proceed. This is not an atomic transaction
across external side effects.

## Resolve approval within a live run

```go
approvals := loop.ToolApprovalResolverFunc(func(
    ctx context.Context, request loop.ToolApprovalRequest,
) (loop.ToolApprovalDecision, error) {
    // The application owns authentication, display, transport, and waiting.
    // Route the response by request.ID and honor ctx cancellation.
    answer, err := approvalUI.Ask(ctx, request.Clone())
    if err != nil { return loop.ToolApprovalDecision{}, err }
    return loop.ToolApprovalDecision{
        RequestID: answer.RequestID,
        Approved: answer.Approved,
        Reason: answer.SafeReason,
    }, nil
})
```

The UI dependency above is illustrative. The returned ID must exactly match the
request, whether approving or denying. IDs are fresh for every request and run;
stale IDs, resolver errors, and resolver panics abort the batch without starting
handlers. Cancellation is checked again after the resolver returns. Resolution
is sequential within a batch; a shared resolver may receive simultaneous requests
from different runs and must handle them safely.

`EventToolApprovalRequested` and `EventToolApprovalResolved` carry independent
request snapshots. `ToolExecution.Decision` retains the original policy;
`ApprovalID` and `Approval` retain the resolution. Missing resolvers emit a
resolved refusal. Resolver errors, panics, and mismatched IDs emit a resolved
failure with `Approval.Code=approval_failed` before ending the run. Cancellation
of the run context records `approval_canceled`. A resolver's own timeout or
cancellation remains `approval_failed` and terminates with an error, preserving
its cause without marking the workflow canceled. These outcomes retain the
request ID and `ToolNotStarted`; they do not represent a human denial. On cancellation, resolved
event delivery is nonblocking so a stopped consumer cannot prevent termination.
Canceled delivery is best-effort: if the buffer is full, the resolved event and
terminal snapshot may be omitted. Consumers must reconcile any pending approvals
when the stream closes; they cannot rely on a final event after cancellation.
Applications must consume the event stream while resolving approvals, or use
`Workflow.Run`, which drains it automatically. Blocking the event consumer on an
approval event while waiting for later events from the same run can deadlock.

A missing resolver is not a paused workflow. The current implementation waits
only within a live run; it supplies no persistence, recovery, durable invocation
identity, or exactly-once guarantee. Durable pause/resume is separate work tracked
in [#157](https://github.com/Samuelk0nrad/gai/issues/157).

## Process results before publication

```go
redact, err := loop.RedactToolResult(func(ctx context.Context, text string) (string, error) {
    return applicationRedactor.Redact(ctx, text)
})
if err != nil { return err }
limit, err := loop.LimitToolResultBytes(16 * 1024)
if err != nil { return err }
processor, err := loop.ChainToolResultProcessors(redact, limit)
if err != nil { return err }
```

Place the byte limit last. Zero disables it. Positive limits must be at least
`loop.MinToolResultBytes` (22 bytes) so every fixed refusal diagnostic fits in
full; smaller limits fail construction. It bounds the published result text,
including error text, by replacing oversized output with `ErrToolOutputLimit`;
it never truncates the original JSON or UTF-8 payload. It does not cap memory
allocated inside a handler. `RedactToolResult` sanitizes success and error text,
dropping the original error chain rather than retaining hidden sensitive causes.
Only safe framework classifications are retained. Oversized refusals preserve
`ErrToolDenied` or `ErrToolApprovalRequired` alongside `ErrToolOutputLimit`, using
a complete safe refusal message. Custom processors may replace refusal text,
but the executor preserves the original denial or approval-required classification
and the model-facing error flag. Filtering cannot turn an unapproved call into a
successful invocation.

A custom processor returns `ToolResult{Text: ..., Err: ...}` as a replacement.
Use `RejectToolResult("safe reason")` to withhold output while continuing the run
with a model-visible error. A processing error or panic terminates the run and
withholds the raw output. Error implementations retained in snapshots must be
immutable. Policy/resolver reasons and processor errors must themselves be safe
to publish.

Processing precedes outer tool events, retained results, conversation history,
and loop tool-output telemetry. It cannot undo content already logged inside a
handler/provider, or remove model-emitted call arguments. Configure content
capture at those boundaries separately. Direct `loop.CallTool` and direct handler
calls bypass policy, approval, scheduling, result processing, and loop telemetry.

`ToolExecution` separates authorization, invocation, and output state. A handler
can be `succeeded` while its output is `rejected`. A timeout or handler error does
not prove that an external side effect did not happen. Tool pipelines are joined
before terminal snapshots are returned. Context deadlines are cooperative:
handlers, processors, and resolvers must return after cancellation; GAI does not
abandon them in detached goroutines. Loop handler/hook panics are terminal
`ErrToolPanic` failures without exposing panic values. Direct `CallTool` keeps
ordinary Go panic behavior. Model-generation retries never replay tool execution.

Refused, malformed, and unknown calls emit a result/error event with
`ToolNotStarted` and no invocation start event or tool span. They still emit
`loop_tool_finished` metadata with duration zero. Finished observations carry
`tool_outcome`, `status`, `duration_ms`, and a safe `error_code` on errors; the
shared observation finalizer strips raw `Err` and normalizes `outcome` to `error`.
A processor failure has `tool_outcome=processing_error`, independently of the
handler execution state.

## Override one workflow

```go
workflow, err := worker.NewRun(ctx, agent.RunInput{
    Execution: &agent.ExecutionOverrides{
        // Atomic replacement: also resets the inherited default timeout.
        ToolExecution: &loop.ToolExecutionConfig{MaxConcurrent: 2},
        ToolPolicy: agent.Optional[loop.ToolPolicy]{Set: true, Value: sessionPolicy},
        // Explicitly clear approval handling; required approvals are refused.
        ToolApprovalResolver: agent.Optional[loop.ToolApprovalResolver]{Set: true},
    },
})
```

Nil configuration pointers inherit. A pointer to an empty execution config resets
the concurrency and timeout defaults. For policy, resolver, and processor,
`Optional.Set == false` inherits and ignores `Value`; `Set == true` replaces,
including nil. Clear the policy only when allowing all registered calls is the
intended behavior. Overrides apply to the primary agent, not nested middleware
agents.

`Agent.New` snapshots definition configuration and registration options.
`NewRun` snapshots override configuration and registration options. Timeout
pointers and rule slices are copied; tool handlers, guards, policies, resolvers,
and processors remain shared dependencies. Custom options providers must be safe
to read during capture. Effective settings are validated after overrides and
before prompt construction, so a valid override can replace an invalid default.

## Pre-v1 migration

| Old API | Replacement |
| --- | --- |
| `Function(ctx, *ai.ToolCall) *ToolResponse` | `Function(ctx, ai.ToolCall) (string, error)` |
| `NewToolSuccess(text)` | Return `text, nil` from the handler. |
| `NewToolError(err)` | Return `"", err` from the handler. |
| `ToolResponse` and its getters | `ToolResult{Text, Err}` fields; `String()` selects error text when present. |
| `ToolResponseProcessor` | `ToolResultProcessor.Process(ctx, ToolPolicyInput, ToolResult) (ToolResult, error)` |
| Event `ToolResponse` payload | Event `ToolResult` payload. |

`IterationPart.ToolResp` retains its name and now holds `*ToolResult`; nil means
there is no publishable result. This migration intentionally removes the old
response wrapper and constructors before v1. Handler errors are tool results;
framework processing errors terminate execution. Rich/multimodal handler output,
automatic tool retries, distributed locks, and durable approval recovery are not
part of this contract.
