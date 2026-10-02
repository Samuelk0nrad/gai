# Mistral provider

Package `github.com/lace-ai/gai/ai/mistral` adapts the Mistral
[Chat Completions API](https://docs.mistral.ai/api/endpoint/chat) to GAI's
canonical `ai.Message` model. Portable requests (`ai.AIRequest`) cover the
controls that every adapter shares. Mistral-only controls use typed options on
the concrete `*mistral.Model`. Anything else stays available through
`Provider.NativeClient`.

```go
provider := mistral.New(os.Getenv("MISTRAL_API_KEY"), nil)
model, err := provider.TypedModel(mistral.MistralSmallLatest)
```

## Capabilities

| Feature | Portable adapter | Notes |
|---|---|---|
| Text, system, and user messages | Supported | Text-only content is sent as a plain string. |
| Function tools, tool choice, parallel calls | Supported | Provider-issued tool call IDs are preserved. |
| JSON object / JSON Schema output | Supported | `ai.ResponseFormat`. |
| Streaming, usage, finish reason | Supported | A terminal `ai.Completion` is emitted for each stream. |
| Reasoning (`thinking` chunks) | Supported | Mapped to `ai.ContentReasoning` parts and `TokenTypeThought` stream events. |
| Reasoning effort | Supported | `none` and `high` for documented adjustable-reasoning models. Other values pass through for models without known facts. |
| Reasoning replay | Supported | Thinking chunks and their signatures are replayed in later turns. |
| Image input | Supported | User messages only. Accepts http(s) URLs, base64 `data:` URIs, or inline bytes. |
| Typed native options | Supported | See [Native options](#native-options). |
| Hosted tools (web search, code interpreter, image generation, document library, connectors) | Deferred | Use `NativeClient`. |
| Citation, reference, tool-reference, document, file, and audio chunks | Deferred | Responses containing them fail explicitly instead of being flattened. |
| Multiple completions (`n > 1`) | Deferred | Use `NativeClient`. |
| `prompt_mode`, `guardrails`, `service_tier`, `metadata` | Deferred | Use `NativeClient`. |

See [Deferred features](#deferred-features).

### Capability descriptors

`Model.Descriptor()` reports what the adapter maps, narrowed by known model
facts and the provider catalog:

- **Adjustable-reasoning models.** `mistral-small-latest` and
  `mistral-medium-3-5` report reasoning and reasoning effort as supported. The
  only allowed efforts are `none` and `high`. Local preflight rejects other
  efforts before any request is sent.
- **Other models.** Reasoning is `Unknown`. Requests pass local preflight and
  the Mistral API decides.
- **Catalog `reasoning: true`.** After `ListModelDescriptors` runs, a model
  marked `reasoning: true` in `/v1/models` reports reasoning as supported. Its
  reasoning-effort values stay unknown. A `false` flag is not treated as a
  limitation, because adjustable-reasoning models can report it.
- **Catalog `vision: false`.** Image input to that model fails before the
  request. The adapter only reads the cached catalog and never triggers
  discovery during generation.

## Reasoning

```go
response, err := model.Generate(ctx, ai.AIRequest{
	Messages:  []ai.Message{ai.TextMessage(ai.RoleUser, "Is 1013 prime?")},
	Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortHigh},
})
fmt.Println("thinking:", response.Reasoning())
fmt.Println("answer:", response.Text())
```

| `ai.ReasoningConfig` | `reasoning_effort` sent |
|---|---|
| zero value | omitted, so the model default applies |
| `Effort: none` | `"none"` |
| `Effort: minimal/low/medium/high/xhigh` | the same value |
| `Enabled` or `IncludeThoughts`, no effort | `"high"`, the only documented value that returns thinking |
| `Effort: max` | rejected with `ai.ErrUnsupportedCapability` |
| `BudgetTokens > 0` | rejected, because Mistral has no token budget |
| `Effort: none` with `Enabled` or `IncludeThoughts` | rejected as contradictory |

Mistral always returns thinking when reasoning is on, so `IncludeThoughts` does
not hide it. Mistral documents `reasoning_effort` only for adjustable-reasoning
models. For other models, including ones the catalog marks as reasoning models,
the API decides whether to accept it. Leave `ReasoningConfig` empty for models
that always reason.

Responses may contain a plain string or an ordered array of `thinking` and
`text` chunks. Streams may move from thinking arrays to mixed arrays and then
to plain strings. Both shapes become ordered canonical parts:

```text
reasoning("…")  text("…")  tool_call(...)
```

The reasoning part's text is readable thinking. When Mistral returns a thinking
`signature`, it is stored on that part as a non-required extension with
namespace `mistral` and type `thinking_signature`. Mistral
[recommends](https://docs.mistral.ai/studio/conversations/reasoning) replaying
the full assistant message. Appending `response.Message` to the next request's
history does that, including the thinking chunk, its signature, text, and tool
calls:

```go
history = append(history, response.Message)
history = append(history, toolResultMessage)
next, err := model.Generate(ctx, ai.AIRequest{Messages: history, Tools: tools})
```

Mistral keeps content and tool calls in separate fields, so text or reasoning
placed after a tool call in one assistant message is rejected rather than
reordered.

## Images

Image parts use the shared `ai.MediaPart`. Text and images keep their order
inside a user message:

```go
message := ai.Message{Role: ai.RoleUser, Parts: []ai.ContentPart{
	{Kind: ai.ContentText, Text: "What is in this picture?"},
	{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", URI: "https://example.com/cat.png"}},
	{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/jpeg", Data: jpegBytes}},
}}
```

- **Supported MIME types.** `image/jpeg`, `image/png`, `image/webp`, and
  `image/gif`.
- **Inline bytes** are sent as a base64 `data:` URL.
- **Rejected inputs.** Other MIME types, other URI schemes such as `file:` or
  `gs:`, and images in system, assistant, or tool messages fail with
  `*ai.UnsupportedContentError`. No request is sent.
- **Vision support.** Choose a vision-capable model. See
  [Vision](https://docs.mistral.ai/studio/conversations/vision).

Broader multimedia work is tracked in #67 and #68.

## Native options

`ChatCompletionOptions` holds typed Mistral controls. Pointer fields keep an
explicit zero or `false` distinct from "not set":

```go
temperature, parallel := 0.0, false
model, err := provider.TypedModel(mistral.MistralSmallLatest,
	mistral.WithChatCompletionOptions(mistral.ChatCompletionOptions{
		Temperature:       &temperature, // sends "temperature": 0
		ParallelToolCalls: &parallel,    // sends "parallel_tool_calls": false
		Stop:              []string{"\n\n"},
	}))

// Per-call variation without mutating the shared model:
key := "tenant-42"
perCall := model.With(mistral.WithChatCompletionOptions(mistral.ChatCompletionOptions{
	PromptCacheKey: &key,
}))
```

| Field | JSON | Validation |
|---|---|---|
| `Temperature` | `temperature` | finite, at least 0 |
| `TopP` | `top_p` | finite, between 0 and 1 |
| `RandomSeed` | `random_seed` | at least 0 |
| `SafePrompt` | `safe_prompt` | none |
| `Stop` | `stop` | entries must be non-empty |
| `PresencePenalty` | `presence_penalty` | finite |
| `FrequencyPenalty` | `frequency_penalty` | finite |
| `ParallelToolCalls` | `parallel_tool_calls` | none |
| `Prediction` | `prediction` (`{"type":"content","content":…}`) | content must be non-empty |
| `PromptCacheKey` | `prompt_cache_key` | non-blank |

**Precedence.** The portable `ai.AIRequest` alone controls `model`,
`messages`, `max_tokens`, `tools`, `tool_choice`, `response_format`,
`reasoning_effort`, and streaming. `ChatCompletionOptions` has no fields for
these, so the two sets never overlap. Options are validated on every
`Generate` and `GenerateStream` call. Invalid values return
`mistral.ErrInvalidChatCompletionOptions` before any request is sent.

`WithChatCompletionOptions` replaces the model's whole option set. `Model.With`
returns an independent copy, so build per-call options from the complete set
of values you want.

## Native access

`Provider.NativeClient()` returns an authenticated HTTP client restricted to the
configured origin. Use it for the deferred features above and for other Mistral
endpoints. These include embeddings, OCR, audio, moderation, batch, and
agents/conversations, which are outside the chat adapter. Native calls bypass
GAI preflight, normalization, observations, and retries.

## Deferred features

These chat features are not mapped by the portable adapter yet:

- Hosted tools: web search, code interpreter, image generation, document
  library, and connectors.
- Citation, reference, tool-reference, document, file, and audio response
  chunks. These currently fail with `ai.ErrUnsupportedCapability`.
- Multiple completions (`n > 1`).
- `prompt_mode`, `guardrails`, `service_tier`, and `metadata`.
