# Checkpointed workflows

Durability is optional. Without `RunInput.Durability`, workflows keep their existing single-use lifecycle and do not write checkpoints. With durability enabled, GAI produces full semantic snapshots and waits for the application's acknowledgement before admitting tool side effects.

```text
materialize input → commit model attempt → generate
  → commit accepted response and invocation IDs
  → commit dispatch intent → invoke tool → process output → commit result
  → commit completed iteration → continue or pause
```

Run the credential-free [example](../examples/checkpointed-workflow/main.go) with `go run ./examples/checkpointed-workflow`. It pauses for approval, JSON-roundtrips the checkpoint, creates a fresh agent, and resumes the same logical run. Its in-memory store demonstrates the compare-and-swap contract; production applications must replace it with durable storage.

## Storage and ownership

```go
durability := agent.Durability{
    DefinitionRevision: "support-v1",
    Commit: func(ctx context.Context, expected uint64, next agent.Checkpoint) error {
        return store.CompareAndSwap(ctx, next.RunID, expected, next)
    },
}
w, err := a.NewRun(ctx, agent.RunInput{
    ID: "run-123",
    Prompt: modelinput.PromptInput{User: ai.TextParts("Check my order")},
    Durability: &durability,
})
```

The application owns save/load, worker ownership, queues, cancellation intent, external event delivery, and external idempotency. `Commit` returning nil promises that **the exact supplied checkpoint** has been durably saved. `expected == 0` means create-if-absent. Later revisions must update atomically only when the stored revision matches `expected`. One current checkpoint per run is sufficient; keeping historical versions is optional.

For PostgreSQL, use a table with `run_id` as the primary key, `revision`, and `state JSONB`. Within the application's transaction, insert revision 1 only if absent, or update `WHERE run_id = ... AND revision = expected`. Require exactly one affected row. A transaction can also write an outbox entry for durable event forwarding.

If acknowledgement is ambiguous, the callback must reload and compare both the proposed revision and content before returning nil. Retry that same write inside the callback when appropriate. An error stops further admission and returns `loop.ErrCheckpointCommit`; GAI never reruns the handler to repair a failed result write. Already-started handlers settle where possible. The last acknowledged checkpoint remains authoritative, even if the callback saved a later revision before returning an error: load storage before recovery.

`Resume` claims ownership by committing a fresh execution identity against the loaded revision before model or tool work starts. CAS alone cannot revoke effects already started by another worker. Use leases/fencing and external idempotency where required. Commit callbacks may call `Result` or `RequestPause`, but must not synchronously wait for the workflow to finish or consume its event stream.

Each commit carries the complete checkpoint, including the accepted conversation and tool results. The checkpoint passed to `Commit` shares memory with the workflow and with later revisions, so the callback must not modify it; encoding it or keeping it unchanged is safe, and `Checkpoint.Clone` gives a copy to modify. Storage cost therefore grows with the transcript: a run with many large tool results writes its whole history at every boundary. Keep large tool output out of the conversation where possible, for example by returning a reference that the application stores separately.

Persistence of settled tool results, terminal provider usage, and stage/workflow termination retains context values but detaches execution cancellation. The callback must enforce its own storage timeout. This lets a cooperative cancellation save known outcomes and `StatusCanceled` without keeping event consumers alive. Admission writes still use the live execution context. A genuine storage failure remains a checkpoint error, including during cancellation; GAI does not claim cancellation was durably recorded when that write failed.

Checkpoints contain selected prompts, user content, tool arguments and processed results. Apply the application's access, encryption and retention rules. Raw output rejected by the tool result processor is not checkpointed. Error text in saved stage failures is deliberately reduced to safe classifications.

## Pause and resume

`RequestPause()` is cooperative and non-blocking. Continue draining `RunEvents` until it closes. GAI finishes an in-flight model response, stops admitting new tools, waits for already-started tool pipelines, and commits the paused state. Long-running tools may delay pause; retry backoff is interrupted. Pause stops the admission of tools, so a model response without tool calls is accepted rather than paused, and a stage whose final answer was streaming when the pause arrived completes. A pause that arrives after a stage completes still decides the next stage and saves its prompt, then pauses before that stage's first model call. The paused checkpoint therefore always has an active stage that a resume command can steer. If no later stage runs, the workflow completes instead.

On an acknowledged pause, `Wait()` returns nil error with `Complete == true`, `Status == agent.StatusPaused`, and an authoritative `Checkpoint`. Complete describes the execution handle, not the logical run. That handle remains single-use; repeated `Wait` calls return the same result.

```go
checkpoint, err := store.Load(ctx, "run-123")
// Construct the same revision-compatible agent definition in a new process.
resumed, err := a.Resume(ctx, checkpoint, agent.ResumeInput{}, durability)
result, err := resumed.Run(ctx)
```

The logical run ID stays the same; each handle gets a new execution ID. Only a checkpoint whose status is `StatusSucceeded` is finished; `Resume` returns `agent.ErrWorkflowFinished` for it. `StatusFailed` and `StatusCanceled` describe the last execution handle, not the logical run: both resume from the last acknowledged continuation, subject to limits and tool reconciliation. A canceled or failed model attempt is saved with its known usage and settled, so a resume starts a new attempt without charging retry budget. An attempt interrupted by a lost worker has an unknown outcome and cost; it consumes one retry. Without a `RetryPolicy` no retry is allowed, as in ordinary runs, so such a stage cannot continue; set `RetryPolicy.MaxRetries` to allow recovery from lost workers. Every new attempt gets a new generation identity. Accepted tool plans are never regenerated.

A run that stops before its first checkpoint commit, for example because a context source fails or the context is already canceled, stores nothing and returns `Result().Checkpoint == nil`. Start it again with `NewRun` and the same run ID.

`RetryPolicy.TotalTimeout` limits active execution time for each stage. `RetryPolicy.AttemptTimeout` covers the model call only, not the checkpoint commit and event delivery before it. Each continuation records the time used so far (`Continuation.Elapsed`), measured up to its last acknowledged commit. Time spent paused or waiting for a resume does not count, so an approval may wait longer than `TotalTimeout`. A worker lost between commits does not charge the time since its last commit. `Resume` checks the limits of every unfinished stage before claiming the checkpoint (see `loop.Continuation.LimitError`). A stage that still has work but has used its whole time budget returns `loop.ErrTimeoutExhausted`, which matches `context.DeadlineExceeded`. A stage whose interrupted attempt has no retry left returns `loop.ErrMaxRetries`, and a stage with no iteration left for new model work returns `loop.ErrMaxIterations`. The retry policy comes from the live `Definition`, so a compatible definition with a larger `TotalTimeout` or `MaxRetries` can continue a stage that used up those budgets. The iteration limit is saved with the stage's settings, including any per-run override, so raising `Limits.MaxLoopIterations` in the definition does not let an exhausted stage continue; start a new run instead.

| Transition | Preserved state and behavior |
| --- | --- |
| Provider retry | Starts a new physical model attempt under the existing retry policy and consumed retry budget. |
| Generation discard | Rejects that response from accepted conversation while retaining its billed usage and consumed iteration. |
| Workflow recovery | Restores committed prompt, conversation, pending plan, tool ledger, approvals, stage/output decisions, commands, counters, usage, and elapsed execution time. It reconciles uncertain effects before continuing. |

`Resume` validates saved loop settings and reattached dependencies before claiming the checkpoint. A failed restore leaves stored status, execution identity and revision untouched. Propagated middleware setup failures remain unfinished so a later resume can retry mapping or child construction; `RecordError` deliberately completes and retains an accepted failure.

Canceling the execution context stops that handle. GAI saves known outcomes and `StatusCanceled`, and the run remains resumable; a tool whose handler stopped because the execution was canceled or timed out keeps an unknown outcome and needs a recovery resolution, because it may have partly run its effect. A handler that hits its own tool timeout while the run continues is an ordinary tool error. A durable run reports `StatusSucceeded` only when every stage has completed. To end a logical run for good, record that decision in the application's control inbox so a queue cannot revive the checkpoint. For worker shutdown, prefer `RequestPause`, which lets started tools finish; for ownership loss, fence writes and cancel the context.

## Tool identity and uncertain outcomes

`loop.InvocationFromContext(ctx)` exposes `RunID`, `ExecutionID`, `StageID`, `GenerationID`, `InvocationID`, and the physical dispatch `Attempt`. Use `RunID + ":" + InvocationID` as an external idempotency key when the receiving system supports it. Provider `ai.ToolCall.ID` stays unchanged for transcript pairing. Identical arguments in different model plans are different invocations.

| Last acknowledged state | Recovery |
| --- | --- |
| Planned, never dispatched | Run under current policy and approval rules. |
| Processed result committed | Reuse it; do not call the handler or processor again. |
| Dispatch committed without usable result | Require an explicit recovery resolution. This includes a handler interrupted because the execution stopped. A dispatch canceled before its handler starts is reverted to planned. |
| Denied or canceled before dispatch | Reuse the paired non-execution result. |

`Resume` returns `*loop.RecoveryRequiredError` (also matching `loop.ErrRecoveryRequired`) with unresolved invocation records. Supply a command with `ToolResolutions` using one of:

- `UseVerifiedResult`: provide a canonical, already processed/redacted `ai.ToolResult` and `ToolExecution` obtained from authoritative records. Preserve the original tool call ID/name. Handler execution and output-processing status are separate.
- `ReplayInvocation`: explicitly assert `SafeReplay: true`. GAI preserves invocation identity and increments its dispatch attempt. Current authorization is still checked. `ToolTraits.Idempotent` alone never authorizes replay. A replay cannot be combined with `CancelUnstartedTools` in the same command, because the steering would cancel the replay.
- `ConfirmNotExecuted`: assert that the operation did not happen, allowing normal admission. It is rejected when the saved state shows the handler ran, for example when only result processing failed; use `ReplayInvocation` or `UseVerifiedResult` instead.

Otherwise leave the run blocked. GAI cannot infer whether an external effect occurred, undo an effect whose output processor failed, or provide exactly-once behavior for an external service that ignores idempotency keys.

## Steering and approvals

Pause, then resume with an application-assigned stable `CommandID` and optional `StageID` (the active suspended stage by default). `Prompt.User` appends user content; `Prompt.Context` remains trusted machine context. Context parts must render purely and deterministically: validation and duplicate comparison can render them before the ownership claim.

`ContinuePendingTools` is the default. GAI finishes the accepted plan before appending steering and generating again. `CancelUnstartedTools` preserves completed calls and resolves only undispatched calls with `Not executed: canceled by steering.` before appending the new input. Outcome-unknown calls must first be reconciled. Provider call/result pairing is preserved.

The command is applied and recorded atomically: its effects go into the stage continuation, and `Checkpoint.Commands` keeps its ID with a SHA-256 digest of the materialized content. Exact redelivery is a no-op, including after its stage has completed; reusing an ID with different content returns `agent.ErrCommandConflict`. New commands cannot reopen completed stages. Persist pause requests in the application's durable inbox if they must survive process loss before acknowledgement.

Set `Durability.DeferredApprovals` explicitly to persist approval requests and pause without a waiting resolver. Resume with the same request IDs in `ResumeInput.Approvals` and a command ID. Requests bind an invocation to its exact arguments. Saved approval never overrides newly revoked permissions. Without deferred mode, the existing resolver/refusal behavior is unchanged.

## Definitions, prompts and middleware

`DefinitionRevision` is the application's compatibility contract for the model configuration, tools, policies, processors, prompt rules and middleware. Use the same value only for compatible definitions; mismatches fail before execution. Schema version 1 is validated strictly; automatic migrations are not provided.

Durable runs require the standard `modelinput.Builder`, including the built-in history-compaction wrapper. GAI freezes the materialized request prefix and conversation after context selection. Resume restores these values without rerunning the prompt factory, sources or compaction. Refreshing context requires explicitly supplied continuation input or a new run. Restored context parts preserve semantic content and names, not original Go implementation types.

Register live dependencies and retry policy on `agent.Definition`; per-run overrides of model, tools, policy, resolver, processor, counter or retry policy are rejected in durable mode. Serializable scalar execution overrides are supported and retained. Metadata must be JSON-serializable and is normalized at the first durable `NewRun` to JSON types; numbers become lossless `json.Number`, and structs become maps. The same representation is delivered after restoration.

Built-in `AgentMiddleware` stages require unique stable `AgentMiddlewareConfig.ID` values, separate from names and event indexes. `MapInput` and `ShouldRun` must be pure and deterministic: a crash before committing their outputs can reevaluate them. Once materialized, input, decision, child continuation and completed output decision are reused. Preserve/append/replace decisions survive recovery; completed stages never rerun.

Durable v1 supports a flat ordered list of built-in agent stages. Custom streaming middleware and nested middleware definitions are rejected before primary execution. A paused or recoverable failed stage blocks downstream stages. `RecordError` may finish a failed stage only when no uncertain tool outcome remains; recovery-required failures propagate so the application can resolve them.

This changes when `ShouldRun` runs. A stage is decided only after every earlier stage has completed, either successfully or with an error kept by `RecordError`. When the primary agent or an earlier stage fails, is canceled or pauses, later stages emit `EventStageStart` and `EventStageFinish` with `StageSkipped` and reason `upstream_unfinished`, and no decision is saved; they are decided after a resumed execution completes the earlier stage. A predicate written to run a fallback after a primary failure therefore never sees that failure in durable mode.

## Events, output and usage

Use the existing workflow event stream. Durable events include run/execution/stage identity, generation/invocation identity where relevant, and a sequence local to this execution. `(ExecutionID, Sequence)` is the live event identity; checkpoint revision is a separate counter. `EventResumed` carries the immutable acknowledged resume-claim snapshot, `EventCheckpointCommitted` reports loop commits, and `EventPaused` carries the acknowledged paused checkpoint. Agent-level stage/terminal commits are reflected in checkpoint snapshots rather than generating a second stream.

On `EventResumed`, discard abandoned provisional output and reconstruct accepted output from the checkpoint's completed generations and stage output decisions. Pending assistant text is provisional: it is re-emitted with its saved generation identity as the pending plan continues. Completed output is restored internally and is not redelivered as fresh output. This provides a canonical rebuild boundary; the live channel is not a transactional delivery guarantee. Use an application outbox when checkpoint persistence and event delivery must commit atomically.

`AgentResult.Iterations` keeps full live diagnostics, such as typed tool errors and request budgets, for iterations accepted in the current execution. Iterations restored from a checkpoint carry the canonical messages and tool outcomes only; a restored tool error is plain text.

Only provider-reported usage is saved, deduplicated by generation identity, including known discarded/retried attempts. `Continuation.UsageIncomplete` marks unavailable accounting. The usage ledger's `Reported` flag also distinguishes an in-flight unreported attempt from a reported zero. Request-budget estimates are not serialized as billed usage. A process lost before usage is received or saved can incur unrecorded cost; recovery does not invent an exact value.
