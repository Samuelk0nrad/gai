# OpenAI provider

Package `github.com/Samuelk0nrad/gai/ai/openai` adapts the OpenAI Chat
Completions API and, with `WithResponsesTransport`, the Responses API to GAI's
canonical `ai.Message` model.

## Hosted tools on the Responses transport

OpenAI runs hosted tools, such as `web_search`, inside one response. GAI does
not define these tools. Add them with a `WithResponsesParams` hook:

```go
provider := openai.New(os.Getenv("OPENAI_API_KEY"), nil, openai.WithResponsesTransport())
model, err := provider.TypedModel("gpt-5.6-terra", openai.WithResponsesParams(func(p *responses.ResponseNewParams) error {
	p.Tools = append(p.Tools, responses.ToolUnionParam{
		OfWebSearchPreview: &responses.WebSearchToolParam{Type: responses.WebSearchToolTypeWebSearchPreview},
	})
	return nil
}))
```

GAI keeps what these tools produce on the assistant message, in both
`Generate` and `GenerateStream`:

- Each hosted tool item (`web_search_call`, `file_search_call`,
  `code_interpreter_call`, `image_generation_call`, `mcp_call` and
  `mcp_list_tools`) becomes an extension part with namespace `openai` and type
  `responses_hosted_tool`. It holds the item JSON as OpenAI returned it, in its
  place among the reasoning, text and function call parts.
- The `output_text` annotations, such as `url_citation`, stay on their text
  part as an `openai`/`responses_annotations` extension. Their offsets are
  relative to that text part.

On the next request, GAI sends the hosted tool items back as input items in
their original order. So when the model searches in one round and answers in
a later round, OpenAI still has the search results and returns resolved
citations. Neither extension is required, so other providers and the Chat
Completions transport skip them.

Read them without parsing raw JSON:

```go
for _, call := range openai.HostedToolCalls(response.Message) {
	// call.Type == "web_search_call", call.Status, call.Action.Query
}
for _, citation := range openai.Citations(response.Message) {
	// citation.URL, citation.Title
}
```

`openai.TextAnnotations(part)` returns every annotation type of one text part,
with offsets into that part's text.

Other output items that need the caller to act, such as `computer_call`, are
still skipped in a stream and rejected by `Generate` with
`ai.ErrUnsupportedCapability`.

Like reasoning items, these extension parts have no text form, so
`ai.RenderMessages`, which history compaction uses, rejects messages that
carry them.
