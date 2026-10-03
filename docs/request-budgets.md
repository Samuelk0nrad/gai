# Request budgets and token accounting

The loop checks the finalized `ai.AIRequest` immediately before each generation
attempt. Context allocation helps select optional input; the final check also
includes rendered system/user/context content, accepted conversation, tool calls
and results, native schemas, response/reasoning options, framing, output capacity,
and a safety margin. Direct model calls do not acquire a loop budget automatically.

## Configure a window

```go
definition := agent.Definition{
    Model: model,
    Limits: agent.Limits{MaxTokens: 512},
    RequestBudget: &ai.RequestBudgetConfig{
        Limit: 2048,
        OutputReserve: 512,
        SafetyMargin: 128,
    },
    Prompt: promptFactory, // returns a fresh builder for each run
}
```

`Limit` applies to **input + output reserve + safety margin**. `InputLimit`
independently applies to **input + safety margin**. Each zero limit disables
that limit; both can be configured together. Equality fits. Values and counts
must be non-negative, and arithmetic overflow is an error.

The effective reserve is `max(OutputReserve, finalizedRequest.MaxTokens)`. Thus
increasing `Limits.MaxTokens` cannot accidentally leave a smaller output reserve.
When `MaxTokens` is zero, GAI does not guess the provider's output default:
`OutputLimitUnknown` is true and the application must choose an adequate reserve
for its window. `SafetyMargin` is an absolute number of tokens. Zero adds no
allowance; choose it for the estimator and workload rather than treating local
estimates as a hard bound on the provider's actual encoding.

A nil agent/loop `RequestBudget` inherits the standard builder's configured
`TokenBudget` and `OutputTokenReserve`. Existing positive builder budgets
therefore gain a final request guard. An explicit `&ai.RequestBudgetConfig{}`
disables inherited limits. Custom builders may optionally implement
`RequestBudget() ai.RequestBudgetConfig`; the minimal `PromptBuilder` contract
is unchanged. Applications using opaque builders can configure the loop/agent
policy directly. Standard builders receive temporary source-allocation hints
without changing their configured defaults.

Per-run settings use the existing execution overrides:

```go
input.Execution = &agent.ExecutionOverrides{
    RequestBudget: agent.Optional[*ai.RequestBudgetConfig]{
        Set: true,
        Value: &ai.RequestBudgetConfig{Limit: 4096, SafetyMargin: 256},
    },
}
```

The object replaces the definition's policy atomically. An unset option inherits
the definition; `Set: true, Value: nil` restores builder inheritance. An explicit
zero object disables limits. Run creation snapshots configuration, so later
caller mutations do not alter an active workflow.

## Local estimates and selection

Automatic counting performs local work only. Selection is an explicit application
`TokenCounter`, then the selected model's optional `TokenCounterProvider`, then
`ai.TextTokenEstimator`. A provider can return its local encoding counter or its
own local estimator. An unavailable encoding returns nil and permits the generic
fallback; an actual counting failure propagates rather than becoming zero.
Custom models need only implement `GenerateStream`.

For a directly constructed loop, `Loop.TokenCounter` takes precedence over an
explicit standard-builder counter, then the model's optional counter and the
generic fallback. The builder's generic default and temporary run injection do
not become application overrides. Custom builders can expose
`ConfiguredTokenCounter` and `SetBudgetTokenCounter` to keep the same distinction;
an older custom `TokenCounter()` getter is treated as an explicit override.

OpenAI uses the upstream local encoding in blocks of at most 10,000 Unicode code
points, with cancellation between blocks. Its versioned ID and estimated fidelity
identify the block algorithm; block boundaries can change whole-text counts.
Anthropic, Gemini, and Mistral currently use the generic local estimator. Gemini's
SDK text tokenizer can download assets and does not demonstrate equivalent
full-request counting, so it is not selected automatically. Mistral does not make
a one-token generation request to count input. Application overrides on this path
must likewise be cheap and local.

`ai.EstimateRequestTokens` versions its portable projection with
`gai.request/canonical-messages-v1`. Plain text parts in a message are concatenated
literally, with four estimated framing tokens per message and three for the reply
prefix. Structured or extension-bearing messages use their canonical JSON
envelope, which already includes framing. Native tool schemas and non-default
response/tool/reasoning options are counted separately once. Text tool protocols
already live in rendered messages and incur no additional native-schema cost.

During request-budget allocation, the standard builder reserves fixed system and
input context using the same rendered or canonical messages as `BuildRequest`,
including their framing. It also debits each returned source's emitted cost
before allocating the next source. Canonical history includes framing once and
hands off its build-local count without recounting the selected snapshot.

Optional sources can implement `context.ContextSourceWithBudgetProjection` to
select content that fits the rendered allowance. `FunctionWithBudget` receives a
build-local `project(ctx, part)` callback and returns the selected part, its
projected count, and an error. The callback includes escaping, renderer markup,
and message framing, or the complete canonical messages of a `ConversationPart`.
For example, a source can try its own candidates in preference order:

```go
func (s *Source) FunctionWithBudget(ctx context.Context, budget int,
    project func(context.Context, gaictx.Part) (int, error),
) (gaictx.Part, int, error) {
    for _, text := range s.candidates {
        part := gaictx.NewTextPart(text)
        tokens, err := project(ctx, part)
        if err != nil { return nil, 0, err }
        if tokens <= budget { return part, tokens, nil }
    }
    return nil, 0, nil
}
```

Do not retain the callback or change the selected part afterward. Exact-fit
selection assumes a stable renderer and remains a local estimate; the final
request guard is authoritative. Legacy `Function(budget)` and
`FunctionWithTokens` receive allocation hints because their unknown return shape
cannot be costed before invocation. Their raw counts cannot guarantee a rendered
fit. The builder calls each source once and does not drop or reconstruct a
returned part to fit. Direct builder use without a loop's request allocation
retains its existing raw-part counting behavior.

Built-in renderers use quiet budget previews and publish render
callbacks and observations only during final request rendering. Custom renderers
with notifications should implement the optional `context.PreviewRenderer`:
`RenderPreview` must produce the same text and errors as `Render` without the
renderer's notifications. Render-only implementations remain supported through
`Render`, including wrappers that override an embedded renderer's output.
Preview still invokes `Part.Render`, so custom parts remain responsible for
effects inside that method.

`TokenCounter.Fidelity` describes text counting, while request diagnostics always
mark this portable projection as estimated. Media, provider serialization, hidden
reasoning, and native options can differ from the projection. Its breakdowns are
diagnostics, not authoritative message counts or provider billing totals.

## Reported usage and checkpoints

An accepted response with `Completion.UsageReported` replaces the complete
request's estimate with its reported `Usage.InputTokens`. Explicit zero is valid;
missing usage is not zero. Custom models must set `UsageReported: true` when
reporting usage. Metadata-only completion updates do not erase earlier usage.

The active execution keeps a detached request checkpoint. When its messages are
an unchanged prefix and the model identity/name, counter algorithm, schemas,
options, and other request fields match, the next request uses the reported input
count plus local estimates of appended assistant/tool messages. The next accepted
usage report replaces that checkpoint. If later responses omit usage, the last
valid checkpoint can still anchor the growing prefix. Any changed base or model
invalidates reuse and triggers a full local estimate. A refreshed builder request
is checked normally; budget checks and model retries do not rebuild context sources.
Shared model/counter implementations must keep their hidden configuration stable
during a run. Represent a configuration change with a new model instance or a
new versioned counter identity so reuse can be invalidated.

Checkpoint-based decisions still have estimated fidelity. Output usage is not
added as replay input: it may include hidden reasoning or content not retained in
the canonical messages. Failed, discarded, and retried attempts contribute to
existing billed-attempt diagnostics, but cannot replace the accepted checkpoint.
Nothing distributes request usage back into message/turn counts or history stores.

## Explicit full-request preflight

Set `Mode: ai.RequestCountAccurate` to require the optional
`ai.InputTokenCounter.CountInputTokens(ctx, request)` capability. Context selection
still uses local estimates. Immediately before every generation attempt, the loop
counts the finalized request once, enforces the configured limits, and generates
with the same captured request. Accurate mode also counts when both limits are
zero. There is no fallback to local estimation after unsupported/failed preflight.

Anthropic implements preflight through native Messages `count_tokens`, sharing
generation's message/tool/system/response/reasoning mapping. Native parameter hooks
are unsupported because rerunning a mutable hook cannot guarantee equivalent
input. [Anthropic documents the endpoint's count as an estimate](https://platform.claude.com/docs/en/build-with-claude/token-counting),
so `provider_preflight` has estimated fidelity and reported generation usage remains
authoritative. Provider rejection, malformed counts, and cancellation stop execution.

OpenAI, Gemini, and Mistral currently have no supported complete-request capability
and return typed unsupported errors through the loop. Local text encoding alone
does not qualify. A custom model may implement the optional capability when it
can count equivalent generation input. Preflight uses the caller/total execution
deadline; the generation-only attempt timeout starts after successful preflight.

## Errors, events, and migration

Use `errors.Is` with `ai.ErrRequestBudgetExceeded`, `ai.ErrRequestCountFailed`, or
`ai.ErrInputTokenCountUnsupported`; `RequestBudgetExceededError` carries the
rejected decision, and `RequestCountError` preserves the underlying failure.
Invalid settings use `ai.ErrInvalidRequestBudget`. Required input that exceeds
the window fails before generation. GAI does not implicitly truncate, summarize,
reduce schemas, retry preflight, or rebuild sources to fit.

`loop.Iteration.RequestBudget` travels through existing iteration/error/discard
events and workflow results. The `loop_request_budget` observation exposes the
method, fidelity, identity, limits, reserve, margin and breakdown. With
`local_estimate`, `MessageTokens` excludes `FramingTokens`; with `usage_checkpoint`,
`MessageTokens` includes framing for additions and `CheckpointTokens` covers the
unchanged prefix. `provider_preflight` reports only a complete input count.
Reported usage and provider/request/timing metadata remain in canonical execution
results/observations and may be persisted separately by applications.

This is a pre-v1 breaking migration:

- Replace `ai.Tokenizer`/`Tokenize` and concrete `Model.Tokenizer()` with the optional
  count-only `TokenCounter` contract. No GAI consumer requires token representations.
- Replace `openai.NewTokenizer` with `openai.NewTokenCounter`,
  `TokenizerUnavailableError` with `TokenCounterUnavailableError`, and
  `IsTokenizerUnavailable`/`ErrTokenizerUnsupported` with
  `IsTokenCounterUnavailable`/`ai.ErrTokenCounterUnsupported`.
- Replace explicit Anthropic text-tokenizer calls with `CountInputTokens` on the
  canonical request. Gemini/Mistral legacy text-tokenizer APIs are removed.
- Migrate agent/context/history counter fields and setters to `TokenCounter` and
  `SetTokenCounter`; clear a per-run counter with `Optional[ai.TokenCounter]{Set: true}`.
- Remove calculated token maps and `TurnTokenStore`/`UpdateTurnTokens` use. History
  stores semantic conversation/summaries; execution owns temporary checkpoints.
  `history.Part.Tokens` now includes message framing for its replay projection;
  summary text counts remain local text counts, not complete request counts.

The [order-support example](../examples/order-support) configures a 2048-token
window, output capacity and margin for both the initial and tool-follow-up request.
