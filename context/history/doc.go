// Package history provides persisted conversation history as a GAI context
// source.
//
// HistorySource loads turns through a HistoryStore, selects the newest history
// that fits the available prompt budget, and exposes canonical messages through
// a ConversationPart. Tool-result previews are applied to that shared projection
// before native mapping or fallback rendering; stored content is unchanged. An
// optional summarizer can compact older turns when the complete history no
// longer fits. Only a changed summary state is saved; selection never writes. Persisted history remains canonical; token-budget trimming only
// changes the prompt projection produced for the current run. JSON persistence
// requires versioned HistoryState and context.StoredMessage envelopes with
// canonical messages; old and unversioned formats are rejected.
//
// Token counts are calculated locally on demand and belong only to the current
// build and its observations. HistoryStore persists semantic state, never token
// caches. Selection and Part.Tokens count the same previewed messages, including
// the summary prefix. Plain summaries are counted as their prefixed text;
// opaque summary content retains its original part and metadata. HistorySource
// implements context.ContextSourceWithTokenCount so the builder consumes the
// selection total without a second counting pass. These are text estimates
// rather than complete provider request costs. Repeated builds recount the
// selected messages. Applications choosing an encoding counter pay its
// tokenization cost on each build; the
// generic ai.TextTokenEstimator avoids that cost. No runtime cache is implicit.
//
// Counting does not mutate messages, turns, parts, or summaries. Shared content
// must remain read-only, and concurrent counting requires a concurrency-safe
// counter. Each builder owns its HistorySource configuration; stores must support
// their application's concurrent load/save operations. Provider-reported usage
// belongs to execution observations rather than history state.
package history
