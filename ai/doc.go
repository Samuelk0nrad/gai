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
// # Canonical conversations
//
// Message is the conversation value shared by requests, provider responses, and
// persisted history. Its ordered ContentPart values retain text, reasoning,
// tool calls/results, JSON, media, and provider extensions. Message.Role uses
// RoleSystem, RoleUser, RoleAssistant, or RoleTool. Storage metadata such as
// session IDs belongs to storage envelopes outside Message.
//
// AIRequest requires canonical Messages. Copy and Clone snapshot mutable
// payloads, including nested results, media bytes, raw JSON, schemas, and opaque
// provider state.
//
// Message.Validate checks tagged payloads and role constraints. Request
// validation matches tool results to preceding calls by ID and name. IDs must
// be unique among outstanding calls and may be reused after completion. Media and
// unknown namespaced extensions can be persisted even when a selected adapter
// cannot replay them. RenderMessages preserves supported roles, part order,
// and tool identity, and returns an UnsupportedContentError for media or opaque
// continuity state that has no faithful text representation.
//
// AIResponse.Message owns output; Text(), Reasoning(), and ToolCalls() derive
// views directly from it. Token carries exactly one canonical Part, error, or
// completion payload. Type(), Text(), and ToolCall() derive stream views. Errors
// and completion accounting remain execution metadata outside messages.
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
// catalog. Unknown facts do not reject requests. NativeTools controls explicit
// native tool definitions; canonical history can also be rendered as text.
// Token counting fidelity belongs to TokenCounter. Catalog caches, descriptor
// merging, and discovery locks are internal implementation
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
// Counter selection and counting must not make network requests. Token-splitting
// APIs have been removed; openai.NewTokenCounter provides an explicit local count.
//
// # Request budgets
//
// RequestBudgetConfig sets total-window and input-only limits, output capacity,
// safety margin, and opt-in accurate mode. The loop checks finalized requests
// before every generation attempt. Accepted reported input usage anchors a
// run-owned checkpoint; unchanged bases count only appended messages locally.
// Changed messages/model/options/tools invalidate reuse. Counts are never written
// into semantic history. RequestBudgetResult travels through existing execution
// events/results and observations and remains distinct from billed attempt usage.
//
// InputTokenCounter is the optional complete-request preflight capability. Only
// explicit RequestCountAccurate selects it; unsupported/failed preflight stops
// generation. Anthropic shares its generation mapper with native count_tokens.
// Provider preflight may still be estimated; reported usage remains authoritative.
//
// # Pre-v1 migration
//
// Replace RequestMessage with Message and its Text/ToolCalls/ToolResult fields
// with ordered ContentPart values. TextMessage and TextParts construct text
// payloads. Use ToolCall directly for calls; ToolResult carries the call ID,
// tool name, ordered result parts, and IsError. Replace RequestMessageRole values
// with Role values. Preserve provider extensions on their original message,
// part, or call instead of converting opaque data to prompt text.
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
// Replace openai.NewTokenizer with openai.NewTokenCounter and concrete Tokenizer
// methods with TokenCounter or explicit InputTokenCounter. Calculated text counts
// are not stored on messages, turns, parts, or summaries. Complete request budgets
// belong to the loop; local text fidelity does not imply exact request accuracy.
package ai
