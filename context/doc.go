// Package context builds the model-facing context used by GAI agents.
//
// A Builder combines system instructions, dynamic ContextSource values, the
// current user prompt, and messages from a Conversation. Parts first produce a
// renderer-neutral RenderNode tree, which a Renderer converts into the final
// prompt string. Optional token budgets reserve space for model output and limit
// how much dynamic context is included.
//
// New defaults to ai.TextTokenEstimator when no local TokenCounter is supplied.
// Agents inject their resolved counter, and compatible history/context sources
// receive it through TokenCounterSetter. Custom Part.Tokens implementations take
// ai.TokenCounter; legacy ai.Tokenizer is no longer the budgeting dependency.
// SystemInstructionsTokens now returns (int, error), and BuildContext propagates
// counting errors instead of continuing with an uncounted part. TextPart,
// NamedPart, and MessagePart count without hidden mutable caches, so immutable
// parts can be shared between fresh builders. History cache removal is separate.
// Counts remain
// text-level estimates or encoding counts, not complete provider request costs.
//
// This package is commonly imported with an alias such as gaictx to distinguish
// it from the standard library context package.
package context
