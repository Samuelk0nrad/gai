# History persistence and compaction

`history.NewHistory(sessionID, reader)` loads a detached snapshot and selects the
recent complete turns that fit the allocated history budget. Its source functions
and ordinary `Builder.BuildContext` calls are read-only. Selection may omit turns from a prompt without deleting
them from storage. Existing summary-first selection and tool-result previews are
preserved. The summary is the oldest history, so it is trimmed first: when it does
not fit, selection skips it and still selects the newest turns that fit.

## Explicit compaction

```go
compactor, err := history.NewCompactor(sessionID, store, history.CompactorDefinition{
    Model: model,          // selects the continuation summarizer
    TokenCounter: counter, // nil selects ai.TextTokenEstimator
    Amount: 0.7,           // oldest fraction; zero selects this default
    SummaryMaxTokens: 512, // zero selects history.DefaultSummaryMaxTokens (2048)
})
if err != nil {
    return err
}
result, err := compactor.Compact(ctx, historyBudget,
    history.WithPrefix(run.Primary.Transcript), // the run that produced the newest turn
    history.WithExpectedRevision(committed),    // the revision that turn was committed with
    history.WithRequestBudget(requestBudget),   // the model's request budget
)
```

`historyBudget` is the allocation for history, not the model context window.
Account for system instructions, current input, tools, and output reserve in the
application's request policy. The request-budget checker never invokes compaction.

The compactor loads once, checks pressure, creates a detached summary candidate,
counts its projection, and attempts one CAS. It does not hold a database transaction
while calling a model. `Changed` means the CAS succeeded; `Revision` is the new
revision or the loaded revision for a no-op. `PressureRemaining` reports that some
working history still cannot fit, including an oversized summary with no remaining
turns. Compaction does not promise to eliminate pressure. `SummarizedTurns` and
`Usage` describe the summary call; that usage is not part of any agent run's
usage. Errors return a zero result except for `Early` and `Usage`: a summary call
that ran before the failure was still billed, so its usage is reported. Do not
consume an uncommitted candidate or automatically retry it.

Only completed turns belong in this working history. History is a compacted
working representation: retain an archival transcript separately if original
messages must remain available.

### Continuation summaries

With only a `Model`, the compactor uses `history.NewContinuationSummarizer(model)`.
It summarizes by continuing a request that already contains the history, so the
provider can serve the shared prefix from its prompt cache:

- The request passed to `WithPrefix` is sent unchanged: messages, tools, tool
  choice and reasoning. One user message is appended. It tells the model not to
  call tools, which messages to summarize, and to merge the earlier
  `Conversation summary:` message. It quotes the start of the first kept user
  message to mark where the summary stops.
- `CompactorDefinition.ResponseFormat` selects what happens to a structured
  output setting: `SummaryResponseFormatOmit` (default) drops it, so the summary
  is plain text; `Keep` sends it unchanged, so the summary is whatever the model
  returns in that format; `Reject` fails before the model call. Changing the
  format costs the cache: Anthropic invalidates the conversation's prompt cache
  when `output_config.format` changes. Agents without structured output are
  unaffected.
- Tool choice and reasoning settings stay unchanged, because changing either
  invalidates Anthropic's messages cache. A completed run's request has its
  required tool choice relaxed to auto; a prefix that still requires a tool call
  is sent with auto, since a required call cannot produce a summary.
- Before any model call, the compactor checks that the prefix contains the stored
  summary and every turn to fold, in order. Turns match by meaning: JSON
  whitespace, key order and escaping may differ, and older tool results may be
  history previews. A mismatch returns `history.ErrSummaryPrefixMismatch`.
- With `WithRequestBudget`, a request that does not fit drops its cached prefix
  after the last folded turn and summarizes all of it, else folds the most turns
  that fit, at least one. `history.ErrSummaryRequestTooLarge` means not even one turn fits.
- A reply that calls a tool, is empty, or stops at the output limit returns
  `history.ErrInvalidSummaryReply`. The call is one generation with no retries.
- `SummaryMaxTokens` limits the summary output and is the output reserved in the
  budget check. Zero selects `history.DefaultSummaryMaxTokens` (2048): the
  continued request's own limit is the answer's, not the summary's, and the limit
  does not affect the cache. A reasoning token budget
  (`ai.ReasoningConfig.BudgetTokens`) is added on top; reasoning controlled only by
  effort is not, so leave it room.

Calling `Compact` without `WithPrefix`, or with a nil request, returns
`history.ErrSummaryPrefixRequired`, even when no compaction is needed, so the
missing prefix shows up on the first call. To summarize without a prefix, configure the
separate-prompt text summarizer from `agent/summary`:

```go
history.CompactorDefinition{Summarizer: summary.New(model)}
```

It renders the turns as one text message under its own system prompt, so it shares
no cached prefix with the conversation. It rejects media and drops provider replay
state, which has no text form.

`AgentResult.Transcript` is the request the run's next generation would have
sent: the full prompt, every accepted message including the final answer, and the
final settings. Compacting right after persisting the turn reuses the cache while
it is still valid, and the request only grows by the answer and the instruction,
so it usually fits the window that the run's last request fit. This is the most
reliable cache-reusing path. Use the same model as the run, because replayed
provider state such as reasoning signatures belongs to it. The transcript is nil
unless the stage completed in that execution; `WithPrefix` takes the pointer, so a
failed, canceled or paused run returns `ErrSummaryPrefixRequired` instead of
panicking.

A transcript only contains the turns the run selected. If an earlier compaction
failed and history selection then dropped the oldest turns or the summary, the
transcript no longer contains them and every later continuation fails with
`ErrSummaryPrefixMismatch`. Recover with a compaction that does not depend on the
transcript: the agent's run-start policy, which sends the full stored history, or
the text summarizer below.

## Automatic compaction in agents

Enable the agent wrapper with a policy:

```go
assistant := agent.New(agent.Definition{
    Model: model,
    HistoryCompaction: &agent.HistoryCompactionPolicy{
        OnFailure: agent.HistoryCompactionContinue, // default: HistoryCompactionStop
        Threshold: 0.85,                            // default when zero
    },
    RequestBudget: &ai.RequestBudgetConfig{Limit: 128000, OutputReserve: 4096},
    Prompt: func(ctx context.Context, input agent.RunInput) (modelinput.PromptBuilder, error) {
        return modelinput.New(modelinput.Definition{
            ContextSources: []modelinput.ContextSource{history.NewHistory(sessionID, store)},
        }), nil
    },
})
workflow, err := assistant.NewRun(ctx, input)
if err != nil {
    return err
}
result, err := workflow.Run(ctx) // compacts if needed, then generates the answer
```

Here `modelinput` is `github.com/Samuelk0nrad/gai/modelinput`, `history` is
`github.com/Samuelk0nrad/gai/modelinput/history`, and `store` implements
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
are respected. `ExecutionOverrides.HistoryCompaction` is an `Optional`: unset
inherits, a nil value disables, and a policy replaces the definition's policy.

The summary continues the run's own prompt: the system prompt, the context before
the history source, and the full stored history, with the run's tools, tool choice
and reasoning, followed by the summary instruction. That repeats the previous run's
final request plus its answer, so it reuses the prompt cache when the next run
starts within the cache lifetime and nothing before the history changed. A required
tool choice is sent as the auto choice the previous run ended with. The run's
request budget limits the summary request, as described above. Set
`HistoryCompactionPolicy.Summarizer` to use another summarizer; `Amount`,
`SummaryMaxTokens` and `ResponseFormat` mean the same as in `CompactorDefinition`.
The run's answer limit is not the summary limit. The system prompt and earlier
context are rendered for the summary only when compaction runs.

Compaction runs once the history exceeds `Threshold` of its allocation, 0.85 by
default. Starting before the allocation is full matters for the cache: the summary
request is the previous request plus its answer and the instruction, so it fits
the request budget only while the history leaves that much room. Once the history
outgrows its allocation, the summary falls back to a request cut after the folded
turns, which misses the cache. `Threshold: 1` compacts only history that no longer
fits. Pressure left after compaction is measured against the whole allocation.
Disabled budgets skip compaction. A zero remaining allocation is still a real
budget. Required input or earlier context that already exceeds the window stops
preparation; compacting history cannot fix it.

The agent attempts at most one compaction per history source during the initial
build. It does not compact again for model retries or tool iterations. No-op
compaction performs no summary call or write. With `HistoryCompactionStop`,
conflicts, summary failures and `history.ErrHistoryPressureRemaining` stop before
the main answer; with remaining pressure a smaller summary may already have been
committed. With `HistoryCompactionContinue`, these failures are reported and the
run builds its prompt from the history as stored, which selection trims to its
allocation. A compaction that started early, while the history still fit, does
not stop the run when it fails: `CompactionResult.Early` is set and the failure is
reported as continued. A revision conflict still follows `OnFailure`, because
another writer changed the history, which may no longer fit. Cancellation and configuration errors, such as a read-only store,
stop the run in all cases. Cancellation is the run's context ending or a
`context.Canceled` error; a deadline that the summarizer or store sets for itself
is an ordinary compaction failure; when it ends the run, the run fails with
`agent.ErrHistoryCompactionTimeout` rather than being reported as canceled. Compaction is not rolled back if the later run fails.
Final request-budget checks, including optional provider counting, still apply and
do not trigger another compaction.

`AgentResult.HistoryCompaction` reports each compaction: the result, including
the committed revision and the summary call's usage, the error, and whether the
run continued past it. The `agent_history_compaction` observation carries the same
fields.

This policy compacts existing working history only. Applications still own session
serialization and persistence of accepted new turns. Compaction can advance the
revision during `Run`: commit the accepted turn against the reported revision,
not one captured before the run, and do not reload only at commit time to accept a
stale answer. Keep explicit compaction when the application needs to load and pin
a snapshot for accepted-turn CAS, as in the session-lifecycle example below.

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
4. `CompactorDefinition{Model: model}` now selects the continuation summarizer and
   needs `WithPrefix`. Keep the previous behavior with
   `CompactorDefinition{Summarizer: summary.New(model)}`. Replace
   `agent.Definition.AutoCompactHistory: true` with
   `HistoryCompaction: &agent.HistoryCompactionPolicy{}`, and the
   `ExecutionOverrides.AutoCompactHistory` pointer with the `HistoryCompaction`
   `Optional`. Checkpoints written with `auto_compact_history` still resume.
   The policy now compacts at 0.85 of the allocation instead of only when history
   no longer fits; set `Threshold: 1` for the previous trigger.
   `Builder.BuildContextWithPreparation` now passes one `modelinput.SourcePreparation`
   instead of separate arguments: replace
   `func(ctx, source ContextSource, remaining int, counter ai.TokenCounter) error`
   with `func(ctx context.Context, prep modelinput.SourcePreparation) error` and read
   `prep.Source`, `prep.Remaining` and `prep.Counter`. `prep.Preceding` renders the
   system prompt and the sources before this one.
   History selection now skips a summary that does not fit and keeps the newest
   turns that do, instead of selecting nothing.
5. Handle conflicts in application orchestration. Preserve the canonical transcript
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
