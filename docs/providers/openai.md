# OpenAI provider

Package `github.com/Samuelk0nrad/gai/ai/openai` adapts the OpenAI Chat
Completions API and, with `WithResponsesTransport`, the Responses API to GAI's
canonical `ai.Message` model.

## Images, PDFs and text files

On the Responses transport, user messages and tool results accept media parts
(`ai.ContentMedia` with an `ai.MediaPart`). Text and media keep their order: a
user message with media becomes one `message` item with a content list.

```go
provider := openai.New(os.Getenv("OPENAI_API_KEY"), nil, openai.WithResponsesTransport())
message := ai.Message{Role: ai.RoleUser, Parts: []ai.ContentPart{
	{Kind: ai.ContentText, Text: "What does this receipt say?"},
	{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/jpeg", Data: jpegBytes}},
	{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "application/pdf", URI: presignedURL, Name: "invoice.pdf"}},
}}
```

| Media | Sent as |
| --- | --- |
| `image/jpeg`, `image/png`, `image/webp`, `image/gif` | `input_image` with `detail: "auto"`. An `http(s)` URL or base64 `data:` URL is sent as `image_url`; inline bytes become a base64 data URL. |
| `application/pdf` | `input_file`. An `http(s)` URL is sent as `file_url`; inline bytes or a base64 data URL are sent as `file_data` with `filename` set to `MediaPart.Name`, or `file.pdf`. |
| `text/*`, `application/json`, `+json` types | `input_text` holding `<file name="…" mime_type="…">` and the UTF-8 text, because OpenAI has no text-file input. Only inline bytes or a base64 data URL are accepted. |

- **Tool results.** A tool result that contains media is sent as a
  `function_call_output` whose `output` is a content list of `input_text`,
  `input_image` and `input_file` items. A failed result sends its text first,
  as `{"error": "…"}`, then its media. Text-only results keep the plain string
  `output`.
- **Rejected inputs.** Other MIME types, other URI schemes (such as `s3:` or
  `file:`), text files given by URL, text that is not UTF-8, and media in
  system or assistant messages fail with `*ai.UnsupportedContentError`
  (`errors.Is(err, ai.ErrUnsupportedCapability)`). No request is sent.
- **Chat Completions.** The Chat Completions transport does not send media and
  returns the same typed error.
- **Model support.** Choose a model that accepts image and file input. See
  [Images and vision](https://platform.openai.com/docs/guides/images-vision)
  and [File inputs](https://platform.openai.com/docs/guides/pdf-files).
- **Budgets.** Request budgets do not count media bytes. See
  [Request budgets](../request-budgets.md#media).

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
