# Gemini

Package `github.com/Samuelk0nrad/gai/ai/gemini` adapts the Gemini
`generateContent` API to GAI's canonical `ai.Message` model.

## Images, PDFs and text files

User messages and tool results accept media parts (`ai.ContentMedia` with an
`ai.MediaPart`). Text and media keep their order.

| Media | Sent as |
| --- | --- |
| `image/png`, `image/jpeg`, `image/webp`, `application/pdf`, audio (`audio/wav`, `audio/mpeg`, `audio/aiff`, `audio/aac`, `audio/ogg`, `audio/flac`), video (`video/mp4`, `video/mpeg`, `video/quicktime`, `video/x-msvideo`, `video/x-flv`, `video/mpg`, `video/webm`, `video/x-ms-wmv`, `video/3gpp`) | `inlineData` for inline bytes and base64 `data:` URIs; `fileData` for an `http(s)` URL, such as a Files API URI or a YouTube link. |
| `text/*`, `application/json`, `+json` types | Text part holding `<file name="…" mime_type="…">` and the UTF-8 text. Only inline bytes or a base64 `data:` URI. |

- **Tool results.** Images and PDFs in a tool result are sent as
  `functionResponse.parts` (multimodal function responses). Gemini 1.x and
  2.x models do not accept those parts, so for models named `gemini-1…` or
  `gemini-2…` the media follows the `functionResponse` as sibling parts of
  the same content. Audio and video always follow as sibling parts. Text files are appended to the response's `output` (or
  `error`) text.
- **Rejected inputs.** GIF and HEIC images, Office files, other MIME types, other URI schemes (such as `gs:` or
  `file:`), text files by URL, text that is not UTF-8, and media in system or
  model messages fail with `*ai.UnsupportedContentError`. No request is sent.
- **Inline size.** Gemini limits the total inline request size. Upload large
  files with the Files API and pass the file URI.
