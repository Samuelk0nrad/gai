# Anthropic provider

Package `github.com/Samuelk0nrad/gai/ai/anthropic` adapts the Anthropic
[Messages API](https://platform.claude.com/docs/en/api/messages) to GAI's
canonical `ai.Message` model. Portable requests (`ai.AIRequest`) cover the
controls that every adapter shares. Anthropic-only controls use provider and
model options. Anything else stays available through `WithMessageParams` hooks
and `Provider.SDKClient`.

```go
provider := anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), nil,
	anthropic.WithPromptCaching(anthropic.PromptCacheTTL5m),
	anthropic.WithRefusalFallbacks(),
)
model, err := provider.TypedModel(anthropic.ClaudeOpus5_5)
```

## Model facts

GAI keeps local facts for every model in its catalog, so preflight works
without loading the live model catalog. After `ListModelDescriptors` runs, the
catalog's thinking and effort data override these facts. A catalog entry that
lists `low`, `medium` and `high` but predates `xhigh` and `max` is read as
supporting only the levels it lists. GAI sends effort only with adaptive
thinking, so a model that accepts only a thinking budget never advertises
effort, whatever the catalog reports.

| Models | Thinking | Effort levels | Forced tool choice | Mid-conversation system messages |
|---|---|---|---|---|
| Opus 5.5, Sonnet 5.5, Fable 5.1, Fable 5.2, Mythos 5.1, Mythos 5.2 | Adaptive only | low to max | Rejected | Accepted |
| Opus 5, Opus 4.8, Haiku 5.5, Fable 5, Mythos 5 | Adaptive only | low to max | Accepted | Accepted |
| Sonnet 5, Opus 4.7 | Adaptive only | low to max | Accepted | Rejected |
| Opus 4.6, Sonnet 4.6 | Adaptive, or a budget | low, medium, high, max | Accepted | Rejected |
| Opus 4.5, Sonnet 4.5, Haiku 4.5, Opus 4.1 | Budget only | None | Accepted | Rejected |

"low to max" means `low`, `medium`, `high`, `xhigh` and `max`.

## Reasoning

`ai.ReasoningConfig` maps to `thinking` and `output_config.effort`:

- Setting `Enabled`, `IncludeThoughts` or `Effort` without `BudgetTokens` sends
  `thinking: {type: "adaptive"}`. `IncludeThoughts` selects
  `display: "summarized"`; otherwise thinking is `"omitted"`.
- `BudgetTokens` sends `{type: "enabled", budget_tokens: N}` on models that
  accept a budget. On adaptive-only models it fails locally with
  `ai.ErrUnsupportedCapability`, because the API returns a 400. Set `Effort`
  instead.
- An effort a model does not list fails locally. For example, `xhigh` on Opus
  4.6 fails.
- Leaving `Reasoning` empty sends no `thinking` field. On Opus 5 and 5.5,
  Sonnet 5 and 5.5, Haiku 5.5 and the Fable and Mythos models thinking still
  runs, because it is on by default there. On Opus 4.7 and 4.8 the model then
  runs without thinking.

**Effort defaults differ.** Opus 5.5 and Haiku 5.5 default to `medium`. Every
other model that accepts effort defaults to `high`. Set `Effort` explicitly when you move a
workload between models.

Thinking and forced tool choice cannot be combined. A request with both fails
locally.

## Tool choice

`ai.ToolChoiceRequired` maps to `tool_choice` `any`, or `tool` when one name is
given. Opus 5.5, Sonnet 5.5, Fable 5.1, Fable 5.2, Mythos 5.1 and Mythos 5.2
reject both forms, so their descriptors list only `auto` and `none`, and
preflight rejects `Required` before a request is sent. On these models, use
`auto` with a prompt that names the tool, or structured output.

## System messages

Leading system messages become the top-level `system` field. On models that
accept them, a later system message is sent as a `role: "system"` entry in
`messages`. Use it for per-turn instructions such as the current time; it
leaves the cached prefix intact. Consecutive system messages merge into one
entry. The API's placement rule is checked locally: each system entry must
follow a user or tool message, and must be last or followed by an assistant
message. Models without support reject non-leading system messages locally.

## Prompt caching

`WithPromptCaching(ttl)` sets the top-level `cache_control` on every
generation and token count request. The API places one breakpoint on the last
cacheable block and moves it forward as the conversation grows, so in a tool
loop each iteration reads the tools, system prompt and earlier turns from the
cache. Use `PromptCacheTTL5m` (also the empty value) or `PromptCacheTTL1h`. An
unknown TTL makes `Provider.Validate` fail.

Cache reads and writes are reported in `ai.Usage.CachedTokens` and
`ai.Usage.CacheCreationTokens`; `InputTokens` includes both. Notes:

- The automatic breakpoint counts toward the limit of four per request, so
  `WithMessageParams` hooks may add at most three more.
- Prefixes shorter than the model's minimum cacheable length are not cached.
- Anything that changes the prefix, such as a timestamp in the system prompt,
  invalidates the cache for everything after it. Put volatile content in a
  mid-conversation system message instead.

## Refusals and fallbacks

A response with `stop_reason: "refusal"` returns `*anthropic.RefusalError`. It
carries `Category`, `Explanation` and `RecommendedModel` from `stop_details`.
Any of them can be empty. The error unwraps to `*ai.TerminalError` with reason
`refusal`, so `errors.Is(err, ai.ErrUnsuccessfulGeneration)` still matches.
Refusals are not retried.

`WithRefusalFallbacks()` turns on server-side fallbacks, with
`fallbacks: "default"` and the beta header `server-side-fallback-2026-07-01`.
A declined request is then re-served on Anthropic's recommended fallback model
within the same call. The provider option applies only to models with
server-side fallbacks: Opus 5, Opus 5.5, Sonnet 5.5, Fable 5.1, Fable 5.2,
Mythos 5.1 and Mythos 5.2. Haiku 5.5, for example, has none.

The model option `WithModelRefusalFallbacks(models...)` turns fallbacks on for
one model, including models outside GAI's catalog. With names, each must be in
that model's allowed fallback models; with none, it uses `"default"`. An empty
name makes `TypedModel` fail. `WithoutRefusalFallbacks()` turns the
provider-wide option off for one model. A `WithMessageParams` hook that sets
`fallbacks` wins: a value turns fallbacks on and GAI adds the beta header for
it, and `nil` turns them off and removes the field.

When a fallback runs:

- The response names the fallback model (`ai.Completion.Model` in streams).
- A `fallback` content block marks each switch. GAI keeps it as an
  `anthropic/fallback` extension part and replays it where it appeared.
- Tool calls from the declined attempt are dropped, including malformed ones,
  which then do not fail the response. In a stream with fallbacks on, parts
  after the first tool call are held until the message ends, the stream ends,
  or a fallback block arrives, so a tool call from the declined attempt is
  never emitted.
- Thinking and text from the declined attempt stay in the response, in both
  `Generate` and streams, because a stream cannot hold them back without
  losing streaming. Thinking before the last fallback block is dropped on
  replay, because the API rejects it.
- A request whose history replays a fallback block always carries the beta
  header, including token count requests.

## SDK version

The adapter uses `github.com/anthropics/anthropic-sdk-go` v1.84.0, the latest
v1 release. Everything above is available in v1. Version 2 was not adopted
here for three reasons:

- No feature in this adapter needs it. Its non-beta `MessageNewParams` also
  lacks `fallbacks`, so fallbacks would still be set on the serialized body.
- The adapter's public API exposes SDK types (`WithMessageParams` takes
  `*sdk.MessageNewParams`, and `SDKClient` returns `*sdk.Client`). Moving to
  the `/v2` import path breaks every caller of those functions.
- v2 sends `anthropic-version: 2026-10-21` instead of `2023-06-01`, which
  changes request validation (for example, `temperature`, `top_p` and `top_k`
  are rejected). That migration should be its own change.
