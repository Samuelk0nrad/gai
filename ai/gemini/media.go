package gemini

import (
	"fmt"
	"strings"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/mediainput"
	"google.golang.org/genai"
)

// Media input on generateContent.
//
//   - Images (image/png, image/jpeg, image/webp), PDFs (application/pdf),
//     audio (wav, mp3, aiff, aac, ogg, flac) and video (mp4, mpeg, mov, avi,
//     flv, mpg, webm, wmv, 3gpp) given as inline bytes or a base64 data URI
//     become inlineData parts; an http(s) URL, such as a Files API URI or a
//     YouTube link, becomes a fileData part.
//   - Text media (text/*, application/json, +json) given as bytes or a base64
//     data URI becomes a text part wrapped in a <file> element.
//
// Tool results with images or PDFs use multimodal function responses: the
// media becomes functionResponse.parts. Gemini 1.x and 2.x models do not
// accept those parts, and function responses take no audio or video, so that
// media follows the functionResponse as sibling parts of the same content.
// Text media is appended to the function response's output text.
//
// User messages and tool results accept media; system and model messages do
// not. Other MIME types, URI schemes and text by URL return
// *ai.UnsupportedContentError.

var (
	geminiImageTypes = mediainput.Types{"image/png", "image/jpeg", "image/webp"}
	geminiAudioTypes = mediainput.Types{"audio/wav", "audio/mpeg", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac"}
	geminiVideoTypes = mediainput.Types{
		"video/mp4", "video/mpeg", "video/quicktime", "video/mov", "video/avi", "video/x-msvideo",
		"video/x-flv", "video/mpg", "video/webm", "video/wmv", "video/x-ms-wmv", "video/3gpp",
	}
)

const geminiSupportedTypes = "image/png, image/jpeg, image/webp, application/pdf, audio (wav, mp3, aiff, aac, ogg, flac), video (mp4, mpeg, mov, avi, flv, mpg, webm, wmv, 3gpp), text/*, application/json"

// AcceptsMedia reports whether the generateContent mapping sends media. User
// messages and tool results take the same types.
func (m *Model) AcceptsMedia(media ai.MediaPart, _ ai.MediaPlacement) error {
	_, err := resolveGeminiMedia(&media)
	return err
}

func resolveGeminiMedia(media *ai.MediaPart) (mediainput.Source, error) {
	source, err := mediainput.Resolve("gemini", media)
	if err != nil {
		return source, err
	}
	switch {
	case geminiImageTypes.Has(source.MIMEType), source.Kind == mediainput.KindPDF,
		geminiAudioTypes.Has(source.MIMEType), geminiVideoTypes.Has(source.MIMEType):
		return source, nil
	case source.Kind == mediainput.KindText:
		return source, source.CheckText("gemini")
	default:
		return source, mediainput.Unsupported("gemini", "MIME type %q (supported: %s)", media.MIMEType, geminiSupportedTypes)
	}
}

// siblingToolResultMedia reports whether model predates multimodal function
// responses. It reads the model ID from resource names such as
// models/<id> and projects/.../publishers/google/models/<id>.
func siblingToolResultMedia(model string) bool {
	name := strings.ToLower(strings.TrimSpace(model))
	if i := strings.LastIndex(name, "models/"); i >= 0 {
		name = name[i+len("models/"):]
	}
	return strings.HasPrefix(name, "gemini-1") || strings.HasPrefix(name, "gemini-2")
}

// mediaPart maps media in a user message, or a sibling of a function
// response.
func mediaPart(media *ai.MediaPart) (*genai.Part, error) {
	source, err := resolveGeminiMedia(media)
	if err != nil {
		return nil, err
	}
	return sourcePart(source)
}

func sourcePart(source mediainput.Source) (*genai.Part, error) {
	if source.Kind == mediainput.KindText {
		text, err := source.TextFile("gemini")
		if err != nil {
			return nil, err
		}
		return &genai.Part{Text: text}, nil
	}
	if !source.Inline() {
		return &genai.Part{FileData: &genai.FileData{FileURI: source.URL, MIMEType: source.MIMEType}}, nil
	}
	data, err := source.Bytes("gemini")
	if err != nil {
		return nil, err
	}
	return &genai.Part{InlineData: &genai.Blob{MIMEType: source.MIMEType, Data: data}}, nil
}

// functionResponseParts maps a canonical tool result to a functionResponse
// part and the media parts that follow it.
func functionResponseParts(r *ai.ToolResult, siblingMedia bool) ([]*genai.Part, error) {
	var text strings.Builder
	var nested []*genai.FunctionResponsePart
	var siblings []*genai.Part
	for _, part := range r.Parts {
		if err := rejectRequiredGeminiExtensions(part.Extensions); err != nil {
			return nil, err
		}
		switch part.Kind {
		case ai.ContentText:
			text.WriteString(part.Text)
		case ai.ContentJSON:
			text.Write(part.JSON)
		case ai.ContentExtension:
			// Optional metadata has no model-visible content.
		case ai.ContentMedia:
			source, err := resolveGeminiMedia(part.Media)
			if err != nil {
				return nil, err
			}
			switch {
			case source.Kind == mediainput.KindText:
				file, err := source.TextFile("gemini")
				if err != nil {
					return nil, err
				}
				if text.Len() > 0 {
					text.WriteString("\n")
				}
				text.WriteString(file)
			case siblingMedia || source.Kind == mediainput.KindAudio || source.Kind == mediainput.KindVideo:
				sibling, err := sourcePart(source)
				if err != nil {
					return nil, err
				}
				siblings = append(siblings, sibling)
			case !source.Inline():
				nested = append(nested, &genai.FunctionResponsePart{FileData: &genai.FunctionResponseFileData{FileURI: source.URL, MIMEType: source.MIMEType}})
			default:
				data, err := source.Bytes("gemini")
				if err != nil {
					return nil, err
				}
				nested = append(nested, &genai.FunctionResponsePart{InlineData: &genai.FunctionResponseBlob{MIMEType: source.MIMEType, Data: data}})
			}
		default:
			return nil, fmt.Errorf("%w: Gemini result %q", ai.ErrUnsupportedCapability, part.Kind)
		}
	}
	key := "output"
	if r.IsError {
		key = "error"
	}
	response := &genai.FunctionResponse{ID: r.ToolCallID, Name: r.Name, Response: map[string]any{key: text.String()}, Parts: nested}
	return append([]*genai.Part{{FunctionResponse: response}}, siblings...), nil
}
