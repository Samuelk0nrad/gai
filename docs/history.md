# History persistence and compaction

`history.NewHistory(sessionID, reader)` loads a detached snapshot and selects the
recent complete turns that fit the allocated history budget. Its source functions
and ordinary `Builder.BuildContext` calls are read-only. Selection may omit turns from a prompt without deleting
them from storage. Existing summary-first selection and tool-result previews are
preserved.

## Explicit compaction

```go
compactor, err := history.NewCompactor(sessionID, store, history.CompactorDefinition{
    Model: model,
    TokenCounter: counter, // nil selects ai.TextTokenEstimator
    Amount: 0.7,          // oldest fraction; zero selects this default
    SummaryMaxTokens: 512,
})
if err != nil {
    return err
}
result, err := compactor.Compact(ctx, historyBudget)
```

`historyBudget` is the allocation for history, not the model context window.
Account for system instructions, current input, tools, and output reserve in the
application's request policy. The request-budget checker never invokes compaction.

The compactor loads once, checks pressure, creates a detached summary candidate,
counts its projection, and attempts one CAS. It does not hold a database transaction
while calling a model. `Changed` means the CAS succeeded; `Revision` is the new
revision or the loaded revision for a no-op. `PressureRemaining` reports that some
working history still cannot fit, including an oversized summary with no remaining
turns. Compaction does not promise to eliminate pressure. Errors return a zero
result; do not consume an uncommitted candidate or automatically retry it.

Only completed turns belong in this working history. Text summarization rejects
media and opaque provider content before generation rather than losing it. History
is a compacted working representation: retain an archival transcript separately
if original messages must remain available.

## Automatic compaction in agents

Enable the agent wrapper with one flag:

```go
assistant := agent.New(agent.Definition{
    Model: model,
    AutoCompactHistory: true,
    RequestBudget: &ai.RequestBudgetConfig{Limit: 128000, OutputReserve: 4096},
    Prompt: func(ctx context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
        return gaictx.New(gaictx.Definition{
            ContextSources: []gaictx.ContextSource{history.NewHistory(sessionID, store)},
        }), nil
    },
})
workflow, err := assistant.NewRun(ctx, input)
if err != nil {
    return err
}
result, err := workflow.Run(ctx) // compacts if needed, then generates the answer
```

Here `gaictx` is `github.com/lace-ai/gai/context`, and `store` implements
`history.HistoryStore`. Return a fresh builder and history source for each run.
Read-only readers are insufficient for this opt-in. Custom builders currently
return `agent.ErrHistoryCompactionNotConfigurable`; they can orchestrate explicit
compaction themselves.

`NewRun` does not summarize or write. At `Run`/`RunEvents` start, the agent wraps
the standard builder's explicit preparation path. Immediately before each history
source is selected, it calls `CompactHistory` with that source's remaining local
allocation and the effective run model/counter. The allocation already accounts
for instructions, current input, fixed context, output reserve, request options,
native tools, safety margin, and preceding sources. Each source runs once; later
sources retain their usual declaration-order priority. Per-run execution overrides
are respected. `ExecutionOverrides.AutoCompactHistory` is an optional `*bool`:
nil inherits, true enables, and false disables automatic compaction for that run.

Defaults match `NewCompactor`: summarize the oldest 70% of completed turns, at
least one, only under pressure. Summary output uses the summarizer's default
limit; the main answer's output limit is not a summary limit. Use explicit
`NewCompactor` orchestration for a different summary model, amount, output limit,
or remaining-pressure policy. Disabled budgets skip automatic compaction. A zero
remaining allocation is still a real budget. Required input or earlier context
that already exceeds the window stops preparation; compacting history cannot fix it.

The agent attempts at most one compaction per history source during the initial
build. It does not compact again for model retries or tool iterations. No-op
compaction performs no summary call or write. Conflicts and summary failures stop
before the main answer. `history.ErrHistoryPressureRemaining` also stops the run;
in this case a smaller summary may already have been committed. Compaction is not
rolled back if the later run fails. Final request-budget checks, including optional
provider counting, still apply and do not trigger another compaction.

This flag compacts existing working history only. Applications still own session
serialization and persistence of accepted new turns. Compaction can advance the
revision during `Run`: do not capture a revision before the run and assume it is
the revision used for the answer, or reload only at commit time to accept a stale
answer. Keep explicit compaction when the application needs to load and pin a
snapshot for accepted-turn CAS, as in the session-lifecycle example below.

## Store contract

`HistoryReader.LoadHistory` returns `HistorySnapshot{Revision, State}`. Snapshots
and accepted write inputs must not alias store-owned mutable data. For an in-memory
store, clone under its lock. Cloning a live pointer after releasing the lock can
race with writers. `HistoryState.Clone` covers nested canonical payloads.

`HistoryStore.CompareAndSwapHistory` must atomically compare the expected revision
and replace the state. A stale revision returns `*history.RevisionConflictError`
with no change. In a database, a conditional update and its affected-row check
belong to one atomic operation; initial creation needs a unique session key. A
legacy load followed by an unconditional save does not satisfy the contract.

Revisions are opaque, store-generated equality tokens. Every accepted write uses a
fresh nonempty revision, including append/edit paths outside this interface.
Empty expected revision means create-only. Initialized empty history still has a
revision. Nil replacements are invalid; clear with an empty `HistoryState`. Stores
that offer deletion preserve a non-reused generation/tombstone revision. Neither a
turn count nor `HistoryState.SchemaVersion` is a concurrency revision.

## Conflicts and conversation ordering

```go
var conflict *history.RevisionConflictError
if errors.As(err, &conflict) {
    // This write was rejected. Reload/reconsider, defer compaction, or queue it.
}
```

CAS protects stored state. Linear chats also need application-owned session
serialization spanning optional compaction, snapshot load, prompt construction,
model/tools, and accepted-turn persistence. Persist against the exact revision
used to build the answer. A lock around individual saves does not establish this
ordering. Different sessions may still run concurrently; multiple processes need
shared coordination rather than an in-process mutex alone.

Do not reload solely to acquire a new revision and append an answer generated from
old context. Do not replay an entire agent run after conflict: tools may already
have performed effects and users may already have seen output. A definite conflict
means no commit; a network error or cancellation during a commit may leave its
outcome unknown. Applications requiring recovery across such errors need durable
operation IDs or equivalent idempotency semantics.

## Migration

This is a breaking pre-v1 change; no legacy unconditional-save adapter is provided.

1. Store revision metadata alongside existing canonical history records. The
   `HistoryState` JSON schema remains unchanged.
2. Replace every store/test double's load/save pair with detached `LoadHistory` and
   atomic `CompareAndSwapHistory`. Update all other writers to the same revision
   scheme and coordinate their deployment.
3. Replace `history.New(..., *SummarizerDefinition)` with `NewHistory` and a separate
   `NewCompactor(..., CompactorDefinition)`. Remove `Enabled`; calling `Compact`
   explicitly selects when model work and persistence happen.
4. Handle conflicts in application orchestration. Preserve the canonical transcript
   and accepted output; token estimates and streamed attempt events are not history.

Compaction has separate `context.history.compact` observations, including
`history_compactor_conflict` and `history_compactor_finished`. A generated summary
is not a committed summary: inspect `changed` on successful completion. Selection
events carry `selection_stage: loaded` or `selection_stage: candidate`; candidate
events describe the proposed state before CAS and do not imply persistence. Build
observations describe selection only.

## Runnable session lifecycle

See [the offline history-session example](../examples/history-session) for atomic
in-memory CAS, cancellation-aware per-session ownership, a prompt pinned to its
loaded revision, accepted-turn persistence, and deterministic concurrency tests.
It also demonstrates explicit pre-run compaction and an application policy for
remaining pressure. The example is process-local; production persistence and
multi-process ownership remain application responsibilities.
