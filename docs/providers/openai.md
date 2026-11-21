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

GAI keeps the hosted tool items and the `output_text` annotations on the
assistant message and sends the items back on the next request. See
[Hosted tools and citations](../hosted-tools.md) for how to read them and
how every provider stores them.
