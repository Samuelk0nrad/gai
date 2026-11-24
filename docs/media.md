# Media

Messages and tool results carry files as `ai.ContentMedia` parts holding an
`ai.MediaPart`: a MIME type, exactly one of `URI` (an `http(s)` URL or a base64
`data:` URI) or `Data` (inline bytes), and an optional `Name`.

```go
message := ai.Message{Role: ai.RoleUser, Parts: []ai.ContentPart{
	{Kind: ai.ContentText, Text: "What changed in this contract?"},
	{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "application/pdf", URI: presignedURL, Name: "contract.pdf"}},
}}
```

GAI never resizes, compresses or re-encodes media. Bytes reach the provider
unchanged, base64 encoded where the API needs it.

## What each provider accepts

Each adapter sends every type its API takes, in user messages and in tool
results:

| Media | OpenAI (Responses) | Anthropic | Gemini | Mistral |
| --- | --- | --- | --- | --- |
| JPEG, PNG, WebP | yes | yes | yes | yes |
| GIF | yes | yes | no | yes |
| PDF | yes | yes | yes | by URL only |
| Text, CSV, JSON (`text/*`, `+json`) | inline | inline | inline | inline |
| .docx, .pptx, .xlsx | yes (text only) | no | no | no |
| Audio | no | no | wav, mp3, aiff, aac, ogg, flac | mp3, wav, flac, ogg, aac, m4a, webm |
| Video | no | no | mp4, mpeg, mov, avi, flv, mpg, webm, wmv, 3gpp | no |

"Inline" means bytes or a `data:` URI; no provider fetches text files by URL.
The OpenAI Chat Completions transport sends no media. Mistral images also need
a vision model. See each provider guide for the wire format:
[OpenAI](providers/openai.md), [Anthropic](providers/anthropic.md),
[Gemini](providers/gemini.md), [Mistral](providers/mistral.md).

Every adapter implements `ai.MediaAcceptor`. `AcceptsMedia(media, placement)`
answers locally, without I/O, whether the model sends a part:

```go
if acceptor, ok := model.(ai.MediaAcceptor); ok {
	err := acceptor.AcceptsMedia(part, ai.MediaInUserMessage) // nil, or *ai.UnsupportedContentError
}
```

## Converting what a provider cannot read

To send any file to any model, give the agent (or `loop.Loop`) a media policy
with a converter. GAI calls it for each part the model does not accept:

```go
def.Media = ai.MediaPolicy{
	Converter: ai.MediaConverterFunc(func(ctx context.Context, media ai.MediaPart, accepts func(ai.MediaPart) error) ([]ai.ContentPart, error) {
		switch {
		case isOffice(media.MIMEType):
			pdf, err := officeToPDF(ctx, media) // for example with LibreOffice
			if err != nil {
				return nil, err
			}
			candidate := ai.MediaPart{MIMEType: "application/pdf", Data: pdf, Name: media.Name + ".pdf"}
			if accepts(candidate) == nil {
				return []ai.ContentPart{{Kind: ai.ContentMedia, Media: &candidate}}, nil
			}
			text, err := officeToText(ctx, media)
			return ai.TextParts(text), err
		case strings.HasPrefix(media.MIMEType, "audio/"):
			transcript, err := transcribe(ctx, media)
			return ai.TextParts("Transcript of " + media.Name + ":\n" + transcript), err
		}
		return nil, ai.ErrMediaNotConverted
	}),
	Fallback: ai.MediaFallbackPlaceholder,
}
```

- **Converter.** Returns replacement parts: text, JSON, or media the model
  accepts. `accepts` checks a candidate against the model, so one converter
  can pick a target per provider. Return `ai.ErrMediaNotConverted` to decline;
  any other error is a `*ai.MediaConversionError`. GAI ships no converters,
  because they need tools such as LibreOffice, ffmpeg or a speech-to-text
  service.
- **Fallback.** For media no converter handled. `ai.MediaFallbackError` (the
  default) keeps the media, so the request fails with the adapter's
  `*ai.UnsupportedContentError`. `ai.MediaFallbackPlaceholder` replaces it
  with text such as `[file: report.docx, application/vnd.openxmlformats-...]`.
- **Tool results** are adapted once, right after `ToolResultProcessor`.
  Converted text is appended to the result text and converted media replaces
  the original, so events, checkpoints and history keep the converted form.
  Media that still cannot be sent becomes a tool error matching
  `loop.ErrToolMediaUnsupported`: the model sees which file it could not read,
  and the run continues instead of failing on every later request.
- **User and history messages** are adapted in every request, before the
  request budget check. Each distinct file is converted once per run and the
  result reused, so a rebuilt conversation does not convert it again.
- **Models without `ai.MediaAcceptor`**, such as custom test models, are sent
  media unchanged.

`ai.NewMediaAdapter(model, policy)` applies a policy outside the loop, for
example before storing a conversation.

## Budgets

Request budgets never count media bytes as text. Each part adds a fixed or
size-based allowance; see [Request budgets](request-budgets.md#media).
