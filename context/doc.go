// Package context builds the model-facing context used by GAI agents.
//
// A Builder combines system instructions, dynamic ContextSource values, the
// current user input, and canonical ai.Message values from a Conversation.
// BuildRequest is the sole assembly path. Arbitrary context parts are lowered
// through a Renderer; ConversationPart values retain their ordered content and
// roles. BuildPrompt applies ai.RenderMessages to that same canonical request.
// Optional token budgets reserve output space and limit dynamic context.
//
// StoredMessage adds IDs, token caches, and a schema version around ai.Message.
// JSON readers upgrade legacy text messages but reject legacy tool records that
// lack call IDs. Content implementations remain only as deprecated legacy
// ingestion helpers. New inputs use PromptInput.User with []ai.ContentPart.
//
// New defaults to ai.TextTokenEstimator when no local TokenCounter is supplied.
// Agents inject their resolved counter, and compatible history/context sources
// receive it through TokenCounterSetter. Custom Part.Tokens implementations take
// ai.TokenCounter; legacy ai.Tokenizer is no longer the budgeting dependency.
// SystemInstructionsTokens now returns (int, error), and BuildContext propagates
// counting errors when budgeting is enabled. With a non-positive TokenBudget,
// the builder skips its own counts; context sources still run and may perform
// their own counting. Source and rendering errors remain errors. TextPart,
// NamedPart, and MessagePart count without hidden mutable caches, so immutable
// parts can be shared between fresh builders. History cache removal is separate.
// Counts remain
// text-level estimates or encoding counts, not complete provider request costs.
//
// This package is commonly imported with an alias such as gaictx to distinguish
// it from the standard library context package.
package context
