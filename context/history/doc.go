// Package history provides read-only conversation context and explicit persistent
// compaction. NewHistory accepts a HistoryReader; its build methods only load,
// select complete recent turns, and return detached canonical messages. Building
// context never runs a model or writes storage. Tool-result previews affect only
// the current prompt projection; stored content is unchanged.
//
// HistoryState is working history: a summary plus an unsummarized tail of completed
// turns. Compaction replaces older details in that state. Applications needing a
// full transcript must archive it separately. Do not publish in-progress turns
// to this compactor: there is no partial-turn/protected-turn policy.
//
// NewCompactor creates an explicit load/summarize/CAS operation. Compact takes the
// history allocation, not the model's full request budget, and makes at most one
// summarization attempt and one CAS. It returns without writing when no compaction
// is needed or no turns remain. PressureRemaining can be true after a successful
// commit. Built-in text summarization rejects media or opaque provider state
// rather than silently dropping it. Configured summarizer retries remain explicit
// summarizer policy; a revision conflict never restarts the compaction or agent.
//
// HistoryReader returns coherent detached snapshots. HistoryStore compares and
// writes atomically, returning a fresh opaque Revision or *RevisionConflictError.
// Use errors.As to detect wrapped conflicts. All writers must participate in the
// same revision scheme. A load-then-unconditional-save adapter is not CAS. Store
// revisions belong to HistorySnapshot, never canonical messages or SchemaVersion.
// Missing history has an empty revision; initialized history has a nonempty one.
// Stores with deletion retain tombstone/generation identity to avoid revision reuse.
//
// CAS prevents lost updates but does not make concurrent answers causally ordered.
// For a linear conversation, applications serialize each whole session workflow:
// optional compaction, load/build, model/tools, and accepted-turn persistence.
// Commit against the revision used to build the answer. Locking only a save, or
// reloading just to append an answer generated from stale context, is insufficient.
// Different sessions can run concurrently. Multi-process services need shared
// coordination; a process-local mutex is not a distributed ownership mechanism.
//
// Conflicts are definite rejected writes. Other commit errors, including transport
// cancellation, can have an unknown outcome. Applications decide whether to reload,
// defer, or retry; operations that require ambiguous-commit recovery need their own
// durable operation identity. Never blindly replay tool effects or streamed runs.
//
// Token counts are build-local estimates over the same previewed messages emitted
// in the prompt, including summary prefix and framing. Semantic values contain no
// mutable token caches. HistorySource hands the count to Builder without a second
// pass. The finalized-request budget guard remains separate from compaction.
// Each builder owns its HistorySource configuration; shared counters, models,
// sinks, and stores must support their application's concurrency.
//
// Pre-v1 migration: replace GetLastHistoryState/SaveHistoryState with LoadHistory/
// CompareAndSwapHistory on every writer. Replace New(..., SummarizerDefinition)
// with NewHistory and a separately configured NewCompactor. Remove Enabled;
// invoking Compact is the opt-in. Existing canonical HistoryState JSON keeps its
// schema version; store revision metadata outside that payload. Migrate writers
// together so an older unconditional writer cannot bypass revision checks.
package history
