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

Checkpoints contain selected prompts, user content, tool arguments and processed results. Apply the application's access, encryption and retention rules. Raw output rejected by the tool result processor is not checkpointed. Error text in saved stage failures is deliberately reduced to safe classifications.

## Pause and resume

`RequestPause()` is cooperative and non-blocking. Continue draining `RunEvents` until it closes. GAI finishes an in-flight model response, stops admitting new tools, waits for already-started tool pipelines, and commits the paused state. Long-running tools may delay pause; retry backoff is interrupted.

On an acknowledged pause, `Wait()` returns nil error with `Complete == true`, `Status == agent.StatusPaused`, and an authoritative `Checkpoint`. Complete describes the execution handle, not the logical run. That handle remains single-use; repeated `Wait` calls return the same result.

```go
checkpoint, err := store.Load(ctx, "run-123")
// Construct the same revision-compatible agent definition in a new process.
resumed, err := a.Resume(ctx, checkpoint, agent.ResumeInput{}, durability)
result, err := resumed.Run(ctx)
```

The logical run ID stays the same; each handle gets a new execution ID. A checkpoint whose status is `StatusSucceeded` or `StatusCanceled` cannot be resumed. A failed execution can recover from its last acknowledged continuation, subject to limits and tool reconciliation. Total timeout uses an absolute logical deadline; pause and recovery do not grant additional time. Interrupted model attempts consume retry budget and get new generation identities. Accepted tool plans are never regenerated.

Persist explicit user cancellation in the application's control inbox so a queue cannot revive an older checkpoint. A canceled context alone cannot guarantee durable cancellation. For worker shutdown, request pause; for ownership loss, fence writes and cancel the context.

## Tool identity and uncertain outcomes

`loop.InvocationFromContext(ctx)` exposes `RunID`, `ExecutionID`, `StageID`, `GenerationID`, `InvocationID`, and the physical dispatch `Attempt`. Use `RunID + ":" + InvocationID` as an external idempotency key when the receiving system supports it. Provider `ai.ToolCall.ID` stays unchanged for transcript pairing. Identical arguments in different model plans are different invocations.

| Last acknowledged state | Recovery |
| --- | --- |
| Planned, never dispatched | Run under current policy and approval rules. |
| Processed result committed | Reuse it; do not call the handler or processor again. |
| Dispatch committed without usable result | Require an explicit recovery resolution. |
| Denied or canceled before dispatch | Reuse the paired non-execution result. |

`Resume` returns `*loop.RecoveryRequiredError` (also matching `loop.ErrRecoveryRequired`) with unresolved invocation records. Supply a command with `ToolResolutions` using one of:

- `UseVerifiedResult`: provide a canonical, already processed/redacted `ai.ToolResult` and `ToolExecution` obtained from authoritative records. Preserve the original tool call ID/name. Handler execution and output-processing status are separate.
- `ReplayInvocation`: explicitly assert `SafeReplay: true`. GAI preserves invocation identity and increments its dispatch attempt. Current authorization is still checked. `ToolTraits.Idempotent` alone never authorizes replay.
- `ConfirmNotExecuted`: assert that the operation did not happen, allowing normal admission.

Otherwise leave the run blocked. GAI cannot infer whether an external effect occurred, undo an effect whose output processor failed, or provide exactly-once behavior for an external service that ignores idempotency keys.

## Steering and approvals

Pause, then resume with an application-assigned stable `CommandID` and optional `StageID` (the active suspended stage by default). `Prompt.User` appends user content; `Prompt.Context` remains trusted machine context. Context parts must render purely and deterministically: validation and duplicate comparison can render them before the ownership claim.

`ContinuePendingTools` is the default. GAI finishes the accepted plan before appending steering and generating again. `CancelUnstartedTools` preserves completed calls and resolves only undispatched calls with `Not executed: canceled by steering.` before appending the new input. Outcome-unknown calls must first be reconciled. Provider call/result pairing is preserved.

The accepted command payload and ID are checkpointed atomically. Exact redelivery is a no-op, including after its stage has completed; conflicting reuse returns `agent.ErrCommandConflict`. New commands cannot reopen completed stages. Persist pause requests in the application's durable inbox if they must survive process loss before acknowledgement.

Set `Durability.DeferredApprovals` explicitly to persist approval requests and pause without a waiting resolver. Resume with the same request IDs in `ResumeInput.Approvals` and a command ID. Requests bind an invocation to its exact arguments. Saved approval never overrides newly revoked permissions. Without deferred mode, the existing resolver/refusal behavior is unchanged.

## Definitions, prompts and middleware

`DefinitionRevision` is the application's compatibility contract for the model configuration, tools, policies, processors, prompt rules and middleware. Use the same value only for compatible definitions; mismatches fail before execution. Schema version 1 is validated strictly; automatic migrations are not provided.

Durable runs require the standard `modelinput.Builder`, including the built-in history-compaction wrapper. GAI freezes the materialized request prefix and conversation after context selection. Resume restores these values without rerunning the prompt factory, sources or compaction. Refreshing context requires explicitly supplied continuation input or a new run. Restored context parts preserve semantic content and names, not original Go implementation types.

Register live dependencies and retry policy on `agent.Definition`; per-run overrides of model, tools, policy, resolver, processor, counter or retry policy are rejected in durable mode. Serializable scalar execution overrides are supported and retained. Metadata must be JSON-serializable and is normalized at the first durable `NewRun` to JSON types; numbers become lossless `json.Number`, and structs become maps. The same representation is delivered after restoration.

Built-in `AgentMiddleware` stages require unique stable `AgentMiddlewareConfig.ID` values, separate from names and event indexes. `MapInput` and `ShouldRun` must be pure and deterministic: a crash before committing their outputs can reevaluate them. Once materialized, input, decision, child continuation and completed output decision are reused. Preserve/append/replace decisions survive recovery; completed stages never rerun.

Durable v1 supports a flat ordered list of built-in agent stages. Custom streaming middleware and nested middleware definitions are rejected before primary execution. A paused or recoverable failed stage blocks downstream stages. `RecordError` may finish a failed stage only when no uncertain tool outcome remains; recovery-required failures propagate so the application can resolve them.

## Events, output and usage

Use the existing workflow event stream. Durable events include run/execution/stage identity, generation/invocation identity where relevant, and a sequence local to this execution. `(ExecutionID, Sequence)` is the live event identity; checkpoint revision is a separate counter. `EventResumed` carries the immutable acknowledged resume-claim snapshot, `EventCheckpointCommitted` reports loop commits, and `EventPaused` carries the acknowledged paused checkpoint. Agent-level stage/terminal commits are reflected in checkpoint snapshots rather than generating a second stream.

On `EventResumed`, discard abandoned provisional output and reconstruct accepted output from the checkpoint's completed generations and stage output decisions. Pending assistant text is provisional: it is re-emitted with its saved generation identity as the pending plan continues. Completed output is restored internally and is not redelivered as fresh output. This provides a canonical rebuild boundary; the live channel is not a transactional delivery guarantee. Use an application outbox when checkpoint persistence and event delivery must commit atomically.

Only provider-reported usage is saved, deduplicated by generation identity, including known discarded/retried attempts. `Continuation.UsageIncomplete` marks unavailable accounting. The usage ledger's `Reported` flag also distinguishes an in-flight unreported attempt from a reported zero. Request-budget estimates are not serialized as billed usage. A process lost before usage is received or saved can incur unrecorded cost; recovery does not invent an exact value.
