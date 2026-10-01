// Package history provides persisted conversation history as a GAI context
// source.
//
// HistorySource loads turns through a HistoryStore, selects the newest history
// that fits the available prompt budget, and exposes canonical messages through
// a ConversationPart. Tool-result previews are applied to that shared projection
// before native mapping or fallback rendering; stored content is unchanged. An
// optional summarizer can compact older turns when the complete history no
// longer fits. Persisted history remains canonical; token-budget trimming only
// changes the prompt projection produced for the current run. JSON persistence
// requires versioned HistoryState and context.StoredMessage envelopes with
// canonical messages; old and unversioned formats are rejected.
package history
