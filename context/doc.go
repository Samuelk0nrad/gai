// Package context builds the model-facing context used by GAI agents.
//
// A Builder combines system instructions, dynamic ContextSource values, the
// current user input, and canonical ai.Message values from a Conversation.
// BuildRequest is the sole assembly path. Arbitrary context parts are lowered
// through a Renderer; ConversationPart values retain their ordered content and
// roles. Text transport uses ai.RenderMessages on the assembled request.
// Optional token budgets reserve output space and limit dynamic context.
//
// StoredMessage adds IDs and a schema version around ai.Message.
// JSON readers require the current schema version and canonical content.
// Inputs use PromptInput.User with []ai.ContentPart.
//
// New defaults to ai.TextTokenEstimator when no local TokenCounter is supplied.
// Agents inject their resolved counter, and compatible history/context sources
// receive it through TokenCounterSetter. Custom Part.Tokens implementations take
// ai.TokenCounter, the local budgeting dependency.
// Sources implementing ContextSourceWithTokenCount can return their selection
// total for the builder to consume without counting the same snapshot again.
// That total is local to the invocation; later builds count fresh content.
// SystemInstructionsTokens now returns (int, error), and BuildContext propagates
// counting errors when budgeting is enabled. With a non-positive TokenBudget,
// the builder skips its own counts; context sources still run and may perform
// their own counting. Source and rendering errors remain errors. TextPart,
// NamedPart, and MessagePart count without hidden mutable caches, so immutable
// parts can be shared between fresh builders. StoredMessage and Turn also count
// on demand without caching or persisting calculated counts. Counters used
// concurrently must support concurrent calls; shared semantic content must stay
// read-only during counting. Build-local totals belong to the builder. Counts
// remain text-level estimates or encoding counts, not complete provider request costs.
//
// Pre-v1 cache migration: remove TokenCount fields from StoredMessage and Turn.
// Replace Turn.Tokenize(ctx, counter, store) with Turn.Tokens(ctx, counter).
// HistoryStore no longer requires TurnTokenStore or UpdateTurnTokens. History
// parts and summaries have no count maps or mutators. Replace
// Summary.TokenCount(counter) with Summary.Tokens(ctx, counter). Versioned
// canonical records with old token-count fields can
// still be read, but those fields are ignored and omitted from new writes.
//
// This package is commonly imported with an alias such as gaictx to distinguish
// it from the standard library context package.
package context
