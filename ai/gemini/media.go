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
//   - Images (image/jpeg, image/png, image/webp, image/gif) and PDFs
//     (application/pdf) given as inline bytes or a base64 data URI become
//     inlineData parts; an http(s) URL, such as a Files API URI, becomes a
//     fileData part.
//   - Text media (text/*, application/json, +json) given as bytes or a base64
//     data URI becomes a text part wrapped in a <file> element.
//
// Tool results with images or PDFs use multimodal function responses: the
// media becomes functionResponse.parts. Gemini 1.x and 2.x models do not
// accept those parts, so for them the media follows the functionResponse as
// sibling parts of the same content. Text media is appended to the
// function response's output text.
//
// User messages and tool results accept media; system and model messages do
// not. Other MIME types, URI schemes and text by URL return
// *ai.UnsupportedContentError.

// siblingToolResultMedia reports whether model predates multimodal function
// responses.
func siblingToolResultMedia(model string) bool {
	name := strings.ToLower(strings.TrimPrefix(strings.TrimSpace(model), "models/"))
	return strings.HasPrefix(name, "gemini-1") || strings.HasPrefix(name, "gemini-2")
}

// mediaPart maps media in a user message, or a sibling of a legacy function
// response.
func mediaPart(media *ai.MediaPart) (*genai.Part, error) {
	source, err := mediainput.Resolve("gemini", media)
	if err != nil {
		return nil, err
	}
	if source.Kind == mediainput.KindText {
		text, err := source.TextFile("gemini")
		if err != nil {
			return nil, err
		}
		return &genai.Part{Text: text}, nil
	}
	if source.URL != "" {
		return &genai.Part{FileData: &genai.FileData{FileURI: source.URL, MIMEType: source.MIMEType}}, nil
	}
	data, err := source.Bytes("gemini")
	if err != nil {
		return nil, err
	}
	return &genai.Part{InlineData: &genai.Blob{MIMEType: source.MIMEType, Data: data}}, nil
}

// functionResponseParts maps a canonical tool result to a functionResponse
// part and, for legacy models, the media parts that follow it.
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
			source, err := mediainput.Resolve("gemini", part.Media)
			if err != nil {
				return nil, err
			}
			if source.Kind == mediainput.KindText {
				file, err := source.TextFile("gemini")
				if err != nil {
					return nil, err
				}
				if text.Len() > 0 {
					text.WriteString("\n")
				}
				text.WriteString(file)
				continue
			}
			if siblingMedia {
				sibling, err := mediaPart(part.Media)
				if err != nil {
					return nil, err
				}
				siblings = append(siblings, sibling)
				continue
			}
			if source.URL != "" {
				nested = append(nested, &genai.FunctionResponsePart{FileData: &genai.FunctionResponseFileData{FileURI: source.URL, MIMEType: source.MIMEType}})
				continue
			}
			data, err := source.Bytes("gemini")
			if err != nil {
				return nil, err
			}
			nested = append(nested, &genai.FunctionResponsePart{InlineData: &genai.FunctionResponseBlob{MIMEType: source.MIMEType, Data: data}})
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
