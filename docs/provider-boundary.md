# Portable models and native provider access

GAI's shared `ai` package describes operations that applications and agent loops
can use across providers. Provider packages expose richer types without adding
provider-specific fields to `ai.AIRequest`.

| Portable API | Provider-native API |
| --- | --- |
| `ai.Model.GenerateStream` and optional `ai.ModelGenerator.Generate` | SDK client methods, request types, responses, and streaming events |
| Text conversation history and function tool calls/results | Native content blocks, hosted tools, multimodal inputs, and provider endpoints |
| Tool choice, JSON/schema output, reasoning hints | Provider request/configuration fields |
| Normalized tokens, completion metadata, and usage | Complete native responses and metadata |
| Optional local descriptors and token counters | Explicit native discovery/counting and other service APIs |

A field belongs in `ai` when it has useful portable semantics for applications or
orchestration. New provider features should first use the native path. Native
availability does not imply that the portable adapter can normalize every event
or represent every input.

## Select a concrete model

Every built-in provider has `TypedModel(name, ...ModelOption) (*Model, error)`.
It validates local configuration and the name without fetching a model catalog.
The returned model also implements `ai.Model`, so it can be assigned directly to
`agent.Definition.Model` or passed to the loop. `Provider.Model(name)` retains its
portable `(ai.Model, error)` signature.

```go
provider := openai.New(apiKey, sink)
model, err := provider.TypedModel(modelName,
    openai.WithChatCompletionParams(func(p *sdk.ChatCompletionNewParams) error {
        p.Temperature = sdk.Float(0.2)
        return nil
    }),
)
if err != nil {
    return err
}
var portable ai.Model = model
_ = portable // Or agent.Definition{Model: model}.
```

Here `openai` is `github.com/lace-ai/gai/ai/openai` and `sdk` is
`github.com/openai/openai-go`.

Provider hooks run after portable request mapping for both synchronous and
streaming calls:

- OpenAI: `WithChatCompletionParams` and `WithResponsesParams`. Each hook runs
  only for its selected transport; use `WithResponsesTransport` for Responses.
- Anthropic: `WithMessageParams` with the SDK's `MessageNewParams`.
- Gemini: `WithGenerateContentConfig` with `genai.GenerateContentConfig`.
- Mistral: `WithChatCompletionOptions` snapshots typed temperature, top-p,
  random-seed, and safe-prompt settings, including explicit zero/false values.

Hooks execute in registration order on fresh mapped values. Errors stop the
request before network transport and appear as ordinary generation errors or
stream error tokens. GAI restores the model name and its required OpenAI
streaming-usage setting after hooks. SDK streaming methods control streaming.
Hooks must preserve portable conversation/tool protocol semantics; native hosted
tools and events requiring different handling belong on the native client path.
Preflight checks the portable request, not arbitrary native hook additions.

Hooks must be safe to call concurrently, must not retain parameter/config
pointers, and must not insert shared mutable data that they later modify. Closure
captures and custom transports remain caller-owned. Construction options should
be applied only while building the provider/model; do not apply them to a running
instance. Mistral value options are copied when the option is created and when it
is applied to a model.

## Use full native request and response types

OpenAI and Anthropic expose `SDKClient() (*sdk.Client, error)`; Gemini exposes
`SDKClient(ctx) (*genai.Client, error)`. These create independent client values
using the same explicit API key, base URL, and HTTP transport as the provider.
`WithBaseURL` and `WithHTTPClient` configure discovery, portable generation, and
native access together. `WithHTTPClient` shallow-copies the HTTP client; its
transport, cookie jar, and callbacks are still shared dependencies.

```go
client, err := provider.SDKClient()
if err != nil {
    return err
}
response, err := client.Chat.Completions.New(ctx, sdk.ChatCompletionNewParams{
    Model: sdk.ChatModel(modelName),
    Messages: []sdk.ChatCompletionMessageParamUnion{
        sdk.SystemMessage("Use the provider's native message representation."),
        sdk.UserMessage("Hello"),
    },
    Temperature: sdk.Float(0.2),
})
if err != nil {
    return err
}
fmt.Println(response.SystemFingerprint) // Native SDK response field.
```

Use the returned SDK's native streaming API to retain all SDK stream events.
OpenAI Responses and Anthropic native/beta services remain accessible through
those clients. Gemini callers can pass native content parts and configuration
and receive native grounding/citation metadata. SDK version upgrades can expose
new features without expanding `ai`.

Native calls bypass GAI descriptor preflight, token normalization, observations,
and loop retry policies. OpenAI and Anthropic SDK retries are disabled by the
configured client; native callers may explicitly change SDK call options.
Gemini uses the existing SDK's retry/transport behavior. OpenAI native access has
no whole-response HTTP timeout; use context deadlines. Anthropic and Gemini
retain the configured HTTP timeout and SDK timeout behavior. Consumers own the
native stream/body lifecycle and must close streams where the SDK requires it.
Provider factories do not take ownership of a caller's transport or close it.

## Mistral native HTTP access

GAI's Mistral adapter uses HTTP rather than a Go SDK. `NativeClient()` provides an
explicit HTTP escape hatch, not a claim of full SDK coverage. Encode your own
typed structs for native fields/endpoints and read the unchanged response or SSE
body. No generic configuration map or growing shared request type is required.

```go
client, err := provider.NativeClient()
if err != nil {
    return err
}
// nativePayload is JSON encoded from an application-owned native request struct.
request, err := http.NewRequestWithContext(ctx, http.MethodPost,
    "v1/chat/completions", bytes.NewReader(nativePayload))
if err != nil {
    return err
}
request.Header.Set("Content-Type", "application/json")
response, err := client.Do(request)
if err != nil {
    return err
}
defer response.Body.Close()
// Check response.StatusCode and decode the full native response or stream.
```

Relative URLs resolve against the configured base URL, including an optional
base path. Absolute URLs and redirects must use the same scheme and host/port;
userinfo and mismatched Host overrides are rejected before authentication is
sent. Request headers and URL are copied; body ownership follows `http.Client`.
Native HTTP errors retain their original status, headers, and response body.
There are no GAI retries or whole-body timeouts; use a context deadline.
The supported sampling fields follow the [Mistral Chat API](https://docs.mistral.ai/api/endpoint/chat).

## Descriptors and discovery

`ModelDescriptor` describes the portable adapter's effective capabilities,
not every feature exposed by the native service. It retains model/provider
identity, native conversation/function-tool support, tool-choice modes,
JSON/schema output, reasoning/effort support, and usage/finish metadata facts.
`Unknown` is distinct from `Unsupported`: preflight rejects known unsupported
features; unknown facts do not reject a request. Unknown native-tool support
still selects the existing text-tool compatibility fallback in the agent.
`Descriptor()` remains local; optional discovery is explicit and context-aware.

`ModelRepository` stays public for application-level model lookup. A custom
provider needs only `Name` and `Model`; a custom model needs only `GenerateStream`.
Optional `ProviderValidator`, `ModelLister`, and `ModelCatalogProvider` support
validation and discovery. Missing discovery is an explicit unsupported-capability
error when listing; it does not prevent registration or named model lookup.

## Pre-v1 migration

- Replace descriptor `ToolCalling` with `NativeTools`, which covers both function
  tool definitions and tool-call/result history.
- `Multimodal` is removed: the portable request has no general multimodal content
  contract. Use native content types where needed.
- `Tokenizer`, `TokenizerDescriptor`, and `TokenizerFidelity` are removed from
  descriptor APIs. Local counting exposes fidelity through `TokenCounter`;
  legacy concrete `Tokenizer()` methods remain explicitly available.
- `ContextMutex`, `ModelCatalogCache`, `IntersectModelDescriptors`, and
  `OverrideModelDescriptor` are implementation details under `internal`.
  Custom providers can return their own independent descriptor values without
  implementing or importing GAI's catalog/cache machinery.
- `Copy`, descriptor validation, optional capability interfaces, and
  `ModelRepository` remain public.

External-package tests in `ai/provider_native_test.go` exercise native SDK
response preservation and hooks through concurrent portable synchronous/streaming
calls. `ai/mistral/native_test.go` covers raw payload preservation, authentication
boundaries, request ownership, cancellation, and stream lifetime.
