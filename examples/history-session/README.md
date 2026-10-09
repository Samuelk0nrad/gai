# Ordered history sessions

Run the offline example from the repository root:

```sh
go run ./examples/history-session
go test -race ./examples/history-session
```

No API keys or network calls are needed. The streaming models are deterministic
stand-ins so the lifecycle can be exercised in CI. The demo starts with a large
completed turn, explicitly compacts it, and answers two new questions. Its second
answer sees the first accepted answer:

```text
Answer based on 0 previous accepted assistant messages.
Answer based on 1 previous accepted assistant messages.
Stored revision 4: summary=true, completed turns=2
```

## Lifecycle

The service owns a session across:

```text
acquire session gate
  → calculate history allocation
  → optionally compact through CAS
  → load one revisioned snapshot
  → build and run the agent against that snapshot
  → persist the accepted turn using the same revision
release session gate
```

The example uses one token counter for compaction and generation, counts its fixed
request, and reserves output capacity before passing the remaining history
allocation to `Compact`. It has no system instructions,
tools, or structured request options; applications adding these must include their
cost too. The finalized-request budget guard still runs before generation.

The application policy is explicit: a compaction conflict or remaining pressure
stops the run before answering. It does not repeatedly summarize until something
fits. A compaction can have committed successfully even when remaining pressure
prevents the subsequent run. Disabling compaction uses ordinary budgeted selection.

`fixedHistory` implements only `HistoryReader` and pins the prompt to the snapshot
used for the final CAS. If another writer changes history after that snapshot is
loaded, the CAS fails.
The service returns the completed workflow result along with the persistence
error, allowing the application to handle already-emitted output/effects without
replaying the run. It never reloads just to get a revision that accepts a stale
answer.

`Primary.Messages` is the accepted canonical transcript for the run and already
contains its user message. `completedTurn` stores that user once and retains the
assistant/tool messages in order. Attempted tokens and rejected retries are not
persisted as accepted history. Turn counts continue from both the summary boundary
and the remaining tail.

## Scope of the example

The gate and store are application code, not a coordination runtime added to GAI.
The store is in-memory and has no deletion or archival-transcript API. Loads and
writes clone nested content under synchronization; CAS compares and replaces
atomically and issues a new revision. Idle gates are removed after the last owner
or waiter leaves.

All writers in one process must share the same service to get linear ordering.
Multiple processes require shared session coordination and a durable store with
atomic CAS. CAS by itself prevents lost updates; it cannot make one answer see
another concurrent answer. Production recovery after an ambiguous commit needs
application operation IDs or equivalent idempotency handling. Keep an archival
transcript separately if compaction must not remove original details.

Tests cover cancellation while waiting, ordered accepted context without duplicate
user messages, parallel sessions, stale-answer conflicts without replay, explicit
compaction and remaining-pressure policy, failed generation, and nested canonical
store/input ownership.
