package openai

import (
	"cmp"
	"encoding/base64"
	"encoding/xml"
	"fmt"
	"net/url"
	"strings"
	"unicode/utf8"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
)

// Media input on the Responses transport.
//
//   - Images (image/jpeg, image/png, image/webp, image/gif) become input_image
//     with detail "auto". An http(s) URL or base64 data URL is sent as is;
//     inline bytes become a base64 data URL.
//   - PDFs (application/pdf) become input_file. An http(s) URL is sent as
//     file_url; inline bytes or a base64 data URL are sent as file_data with a
//     filename, MediaPart.Name or "file.pdf".
//   - Text media (text/*, application/json, +json) given as bytes or a base64
//     data URL becomes input_text wrapped in a <file> element, because OpenAI
//     has no text-file input.
//
// Other MIME types, other URI schemes and text media by URL return
// *ai.UnsupportedContentError. Nothing is dropped.

const (
	mimePDF         = "application/pdf"
	defaultFilename = "file.pdf"
)

var openAIImageMIMETypes = map[string]bool{
	"image/jpeg": true,
	"image/png":  true,
	"image/webp": true,
	"image/gif":  true,
}

func unsupportedMedia(format string, args ...any) error {
	return &ai.UnsupportedContentError{Provider: "openai", Kind: ai.ContentMedia, Detail: fmt.Sprintf(format, args...)}
}

// normalizeMIME lowercases a MIME type, drops parameters, and maps the
// image/jpg alias to image/jpeg.
func normalizeMIME(mimeType string) string {
	mimeType, _, _ = strings.Cut(mimeType, ";")
	mimeType = strings.ToLower(strings.TrimSpace(mimeType))
	if mimeType == "image/jpg" {
		return "image/jpeg"
	}
	return mimeType
}

func partsContainMedia(parts []ai.ContentPart) bool {
	for _, part := range parts {
		if part.Kind == ai.ContentMedia {
			return true
		}
	}
	return false
}

// mediaInputContent maps one media part to a Responses input content item.
func mediaInputContent(media *ai.MediaPart) (responses.ResponseInputContentUnionParam, error) {
	mimeType := normalizeMIME(media.MIMEType)
	switch {
	case openAIImageMIMETypes[mimeType]:
		imageURL, err := mediaURL(media, mimeType, "image")
		if err != nil {
			return responses.ResponseInputContentUnionParam{}, err
		}
		return responses.ResponseInputContentUnionParam{OfInputImage: &responses.ResponseInputImageParam{
			Detail:   responses.ResponseInputImageDetailAuto,
			ImageURL: param.NewOpt(imageURL),
		}}, nil
	case mimeType == mimePDF:
		fileURL, err := mediaURL(media, mimeType, "PDF")
		if err != nil {
			return responses.ResponseInputContentUnionParam{}, err
		}
		file := &responses.ResponseInputFileParam{}
		if strings.HasPrefix(fileURL, "data:") {
			file.FileData = param.NewOpt(fileURL)
			file.Filename = param.NewOpt(cmp.Or(strings.TrimSpace(media.Name), defaultFilename))
		} else {
			file.FileURL = param.NewOpt(fileURL)
			if name := strings.TrimSpace(media.Name); name != "" {
				file.Filename = param.NewOpt(name)
			}
		}
		return responses.ResponseInputContentUnionParam{OfInputFile: file}, nil
	case ai.IsTextMIME(mimeType):
		text, err := textMediaInput(media, mimeType)
		if err != nil {
			return responses.ResponseInputContentUnionParam{}, err
		}
		return responses.ResponseInputContentParamOfInputText(text), nil
	default:
		return responses.ResponseInputContentUnionParam{}, unsupportedMedia("MIME type %q (supported: image/jpeg, image/png, image/webp, image/gif, application/pdf, text/*, application/json)", media.MIMEType)
	}
}

// mediaURL returns an http(s) URL or a base64 data URL with the canonical
// MIME type for an image or PDF.
func mediaURL(media *ai.MediaPart, mimeType, what string) (string, error) {
	if len(media.Data) > 0 {
		return "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(media.Data), nil
	}
	parsed, err := url.Parse(media.URI)
	if err != nil {
		return "", unsupportedMedia("invalid %s URI", what)
	}
	switch strings.ToLower(parsed.Scheme) {
	case "http", "https":
		if parsed.Host == "" {
			return "", unsupportedMedia("%s URL requires a host", what)
		}
		return media.URI, nil
	case "data":
		uriMIME, payload, ok := parseBase64DataURI(media.URI)
		if !ok || normalizeMIME(uriMIME) != mimeType {
			return "", unsupportedMedia("%s data URI must be base64 encoded and match the part MIME type", what)
		}
		return "data:" + mimeType + ";base64," + payload, nil
	default:
		return "", unsupportedMedia("%s URI scheme %q (use http, https, data, or inline bytes)", what, parsed.Scheme)
	}
}

// parseBase64DataURI splits "data:<mime>;base64,<payload>".
func parseBase64DataURI(uri string) (mimeType, payload string, ok bool) {
	if len(uri) < len("data:") || !strings.EqualFold(uri[:len("data:")], "data:") {
		return "", "", false
	}
	header, payload, found := strings.Cut(uri[len("data:"):], ",")
	if !found {
		return "", "", false
	}
	mimeType, isBase64 := strings.CutSuffix(strings.ToLower(header), ";base64")
	if !isBase64 {
		return "", "", false
	}
	return mimeType, payload, true
}

// textMediaInput renders text media as input text. The file name and MIME
// type are kept so the model can tell the file apart from the message.
func textMediaInput(media *ai.MediaPart, mimeType string) (string, error) {
	data := media.Data
	if len(data) == 0 {
		uriMIME, payload, ok := parseBase64DataURI(media.URI)
		if !ok {
			return "", unsupportedMedia("text media %q must be inline data or a base64 data URI; OpenAI does not fetch text files by URL", mimeType)
		}
		if normalizeMIME(uriMIME) != mimeType {
			return "", unsupportedMedia("text data URI must match the part MIME type")
		}
		decoded, err := base64.StdEncoding.DecodeString(payload)
		if err != nil {
			return "", unsupportedMedia("text data URI is not valid base64")
		}
		data = decoded
	}
	if !utf8.Valid(data) {
		return "", unsupportedMedia("text media %q is not valid UTF-8", mimeType)
	}
	var out strings.Builder
	out.WriteString("<file")
	if name := strings.TrimSpace(media.Name); name != "" {
		out.WriteString(` name="`)
		_ = xml.EscapeText(&out, []byte(name))
		out.WriteString(`"`)
	}
	out.WriteString(` mime_type="`)
	_ = xml.EscapeText(&out, []byte(mimeType))
	out.WriteString("\">\n")
	out.Write(data)
	out.WriteString("\n</file>")
	return out.String(), nil
}

// toolResultContent maps a tool result that contains media to the array form
// of function_call_output. Text and JSON parts keep their order as input_text;
// an error result sends its text first as {"error": ...}, then its media.
func toolResultContent(result *ai.ToolResult) (responses.ResponseInputMessageContentListParam, error) {
	var content responses.ResponseInputMessageContentListParam
	var text strings.Builder
	flush := func() {
		if text.Len() > 0 {
			content = append(content, responses.ResponseInputContentParamOfInputText(text.String()))
			text.Reset()
		}
	}
	var media responses.ResponseInputMessageContentListParam
	for _, part := range result.Parts {
		if err := rejectRequiredExtensions(part.Extensions); err != nil {
			return nil, err
		}
		switch part.Kind {
		case ai.ContentText:
			text.WriteString(part.Text)
		case ai.ContentJSON:
			text.Write(part.JSON)
		case ai.ContentMedia:
			item, err := mediaInputContent(part.Media)
			if err != nil {
				return nil, err
			}
			if result.IsError {
				media = append(media, item)
				continue
			}
			flush()
			content = append(content, item)
		default:
			return nil, fmt.Errorf("%w: tool-result content %q", ai.ErrUnsupportedCapability, part.Kind)
		}
	}
	if result.IsError {
		content = append(content, responses.ResponseInputContentParamOfInputText(toolErrorText(text.String())))
		return append(content, media...), nil
	}
	flush()
	return content, nil
}

// functionCallOutputWithContent builds a function_call_output item whose
// output is a content array. The pinned SDK models output as a string only,
// so the array replaces it through the item's extra fields.
func functionCallOutputWithContent(callID string, content responses.ResponseInputMessageContentListParam) responses.ResponseInputItemUnionParam {
	item := &responses.ResponseInputItemFunctionCallOutputParam{CallID: callID}
	item.SetExtraFields(map[string]any{"output": content})
	return responses.ResponseInputItemUnionParam{OfFunctionCallOutput: item}
}
