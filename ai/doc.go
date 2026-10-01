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
// # Portable and native capabilities
//
// The shared request supports portable messages, function tools, response formats,
// reasoning, and normalized output/usage. Concrete provider packages expose
// TypedModel constructors and typed request options; OpenAI, Anthropic, and Gemini
// also expose configured SDKClient access for full native request/response types.
// Mistral exposes a NativeClient HTTP escape hatch. Native calls do not pass
// through GAI preflight, normalization, observations, or loop retry policies.
//
// ModelDescriptor describes portable adapter capabilities, not a native feature
// catalog. Unknown facts do not reject requests. NativeTools covers definitions
// and tool history. Token counting fidelity belongs to TokenCounter. Catalog
// caches, descriptor merging, and discovery locks are internal implementation
// details; ModelRepository remains an optional public application utility.
//
// # Local text counting
//
// TokenCounter is the runtime's count-only, local capability. Its ID includes an
// algorithm/data version, and Fidelity distinguishes an exact text encoding
// from an estimate. Neither describes full request overhead or billed usage.
// Agent selects a supplied counter, then a model's optional TokenCounterProvider,
// then TextTokenEstimator. OpenAI uses its upstream local tokenizer on blocks of
// at most 10,000 Unicode code points, checking cancellation between blocks.
// Block boundaries can change counts, so this has estimated fidelity and a
// distinct counter ID. Anthropic, Gemini, Mistral, and unknown mappings use the
// generic estimate.
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
