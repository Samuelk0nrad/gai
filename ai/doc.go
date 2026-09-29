// Package ai defines provider-independent interfaces and value types for text
// generation.
//
// A Provider exposes named Models. A Model accepts an AIRequest and streams
// Tokens; ModelGenerator optionally supplies synchronous generation.
// ModelRepository selects models by provider and model name.
//
// Provider-specific implementations live in subpackages such as anthropic,
// gemini, mistral, and openai.
//
// # Local text counting
//
// TokenCounter is the runtime's count-only, local capability. Its ID includes an
// algorithm/data version, and Fidelity distinguishes an exact text encoding
// from an estimate. Neither describes full request overhead or billed usage.
// Agent selects a supplied counter, then a model's optional TokenCounterProvider,
// then TextTokenEstimator. OpenAI uses an exact local encoding where known;
// Anthropic, Gemini, Mistral, and unknown mappings use the generic estimate.
// Counter selection and counting must not make network requests. Legacy
// Tokenizer methods remain available explicitly on concrete built-in models;
// some of these APIs can make provider requests or download tokenizer data.
// They are never consulted for automatic budgeting.
//
// # Pre-v1 migration
//
// Model implementations need only GenerateStream. Code that calls Generate or
// Name through an ai.Model uses ModelGenerator or ModelNamer respectively;
// ModelName returns an empty diagnostic name when naming is unavailable.
// Resource cleanup remains caller-owned through io.Closer where supported.
//
// Agent definitions, per-run execution overrides, context parts/builders, and
// history sources now accept TokenCounter instead of Tokenizer. Migrate fields,
// getters and setters to TokenCounter/SetTokenCounter, and summary configuration
// to WithTokenCounter. Custom counters implement CountTokens, ID and Fidelity;
// no token-splitting method is required. Clearing an agent counter override with
// Optional[TokenCounter]{Set: true} restores automatic local selection.
//
// Text counts and the remaining history caches do not represent a full model
// request budget. Request framing, complete input/conversation budgeting, and
// cache ownership remain separate work.
package ai
