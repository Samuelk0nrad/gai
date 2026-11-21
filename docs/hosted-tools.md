# Hosted tools and citations

Some providers run tools themselves inside one generation, such as web
search, web fetch or code execution. GAI does not define these tools. Turn
them on with the provider's request hook. GAI keeps what they produce on the
assistant message, sends it back on later requests, and lets callers read it
in one shape for every provider.

| Provider | Kept as hosted tool calls | Replayed | Citations |
|---|---|---|---|
| OpenAI (Responses transport) | `web_search_call`, `file_search_call`, `code_interpreter_call`, `image_generation_call`, `mcp_call`, `mcp_list_tools` | yes | `output_text` annotations, on their text part |
| Anthropic | `server_tool_use`, and the result blocks (`web_search_tool_result`, `web_fetch_tool_result`, the code execution results, `tool_search_tool_result`, `container_upload`) | yes | text block `citations`, on their text part |
| Gemini | server-side `toolCall` and `toolResponse` parts, `executableCode` and `codeExecutionResult` | yes | grounding metadata, on an extension part at the end |
| Mistral | none: the chat completions API has no hosted tools | - | - |

OpenAI's Chat Completions transport has no hosted tools either.

## Reading them

```go
for _, call := range response.Message.HostedToolCalls() {
	// call.Provider, call.Kind (ai.HostedToolWebSearch, ...), call.Name,
	// call.Status, call.Query, call.URL, call.Raw
}
for _, citation := range response.Message.Citations() {
	// citation.URL, citation.Title, citation.Text, citation.Source
}
```

`HostedToolCall.Kind` is normalized across providers, so callers can count
calls per kind, for example to bill web searches. `Name` keeps the provider's
own tool or item type, and `Raw` holds the provider's item.

Only calls are listed, not their results. Anthropic and Gemini return a
separate result block or part for each call; those are kept for replay only.

Gemini models that do not return server-side tool parts report their
searches only in the grounding metadata. GAI then lists one call per search
query, with an empty `Raw`. These calls are not replayed.

`ContentPart.Citations()` returns the citations of one part. On a text part,
`Start` and `End` are rune offsets into that part's `Text`, and `Text` is the
text they cover. In a stream, text deltas merge into one part, so GAI shifts
the offsets to stay correct in the merged text. Anthropic cites whole text
blocks, so its citations cover the cited block. Gemini citations sit on an
extension part, so they have no offsets, but `Text` holds the supported
segment.

## Storage and replay

A hosted tool call is an extension part with type `ai.ExtensionHostedToolCall`
and the provider's namespace (`openai`, `anthropic` or `google`). Citations
are an `ai.ExtensionCitations` extension. Anthropic and Gemini results use a
`hosted_tool_result` extension in their namespace. Gemini also keeps the raw
grounding metadata as `google`/`grounding_metadata`, which includes the
search entry point that Google asks apps to display.

Each provider replays only its own items, in their original order among
reasoning, text and tool calls. None of these extensions is required, so
other providers skip them, and history compaction leaves them out of the
summary prompt. `ai.RenderMessages` still rejects them, as it rejects any
extension.

## Turning them on

OpenAI, with openai-go v1.12.0:

```go
model, err := provider.TypedModel("gpt-5.6-terra", openai.WithResponsesParams(func(p *responses.ResponseNewParams) error {
	p.Tools = append(p.Tools, responses.ToolUnionParam{
		OfWebSearchPreview: &responses.WebSearchToolParam{Type: responses.WebSearchToolTypeWebSearchPreview},
	})
	return nil
}))
```

Anthropic takes server tools through `anthropic.WithMessageParams`, and
Gemini through `gemini.WithGenerateContentConfig`, for example by appending
`&genai.Tool{GoogleSearch: &genai.GoogleSearch{}}` to `config.Tools`.

## Paused Anthropic turns

Anthropic can stop a long server tool turn with `stop_reason: "pause_turn"`.
The Anthropic adapter then sends the paused turn back and continues it in a
new request, in `Generate` and in streams, up to 10 times per generation. The
caller sees one generation: the parts of every request in order, one
completion with the final stop reason, and usage summed over the requests.
A turn that is still paused after the last continuation fails with an
`ai.TerminalError` whose reason is `pause_turn`.

## Gemini grounding in streams

Gemini normally sends grounding metadata with the final stream chunk. When
several chunks carry it, GAI merges them: search queries, sources (by URI)
and supported segments are each kept once. So a query that the model really
ran twice in one turn is counted once.
