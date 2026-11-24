package anthropic

import (
	"fmt"
	"strings"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/mediainput"
	antropic "github.com/anthropics/anthropic-sdk-go"
)

// Media input on the Messages API.
//
//   - Images (image/jpeg, image/png, image/webp, image/gif) become image
//     blocks: an http(s) URL as a url source, inline bytes or a base64 data URI
//     as a base64 source.
//   - PDFs (application/pdf) become document blocks with a url or base64
//     source, titled with MediaPart.Name when it is set.
//   - Text media (text/*, application/json, +json) given as bytes or a base64
//     data URI becomes a document block with a plain-text source, titled with
//     MediaPart.Name.
//
// User messages and tool results accept media; system and assistant messages
// do not. Other MIME types, URI schemes and text by URL return
// *ai.UnsupportedContentError.

var anthropicImageTypes = mediainput.Types{"image/jpeg", "image/png", "image/webp", "image/gif"}

const anthropicSupportedTypes = "image/jpeg, image/png, image/webp, image/gif, application/pdf, text/*, application/json"

// AcceptsMedia reports whether the Messages API mapping sends media. The
// placement does not matter: user messages and tool results take the same
// types.
func (m *Model) AcceptsMedia(media ai.MediaPart, _ ai.MediaPlacement) error {
	_, err := resolveAnthropicMedia(&media)
	return err
}

func resolveAnthropicMedia(media *ai.MediaPart) (mediainput.Source, error) {
	source, err := mediainput.Resolve("anthropic", media)
	if err != nil {
		return source, err
	}
	switch {
	case anthropicImageTypes.Has(source.MIMEType), source.Kind == mediainput.KindPDF:
		return source, nil
	case source.Kind == mediainput.KindText:
		return source, source.CheckText("anthropic")
	default:
		return source, mediainput.Unsupported("anthropic", "MIME type %q (supported: %s)", media.MIMEType, anthropicSupportedTypes)
	}
}

// mediaContent is an image or document block for one media part. Exactly one
// field is set.
type mediaContent struct {
	image    *antropic.ImageBlockParam
	document *antropic.DocumentBlockParam
}

func mapMedia(media *ai.MediaPart) (mediaContent, error) {
	source, err := resolveAnthropicMedia(media)
	if err != nil {
		return mediaContent{}, err
	}
	switch source.Kind {
	case mediainput.KindImage:
		image := &antropic.ImageBlockParam{}
		if source.Inline() {
			image.Source.OfBase64 = &antropic.Base64ImageSourceParam{Data: source.Base64(), MediaType: antropic.Base64ImageSourceMediaType(source.MIMEType)}
		} else {
			image.Source.OfURL = &antropic.URLImageSourceParam{URL: source.URL}
		}
		return mediaContent{image: image}, nil
	case mediainput.KindPDF:
		document := &antropic.DocumentBlockParam{}
		if source.Inline() {
			document.Source.OfBase64 = &antropic.Base64PDFSourceParam{Data: source.Base64()}
		} else {
			document.Source.OfURL = &antropic.URLPDFSourceParam{URL: source.URL}
		}
		setDocumentTitle(document, source.Name)
		return mediaContent{document: document}, nil
	default: // mediainput.KindText
		text, err := source.Text("anthropic")
		if err != nil {
			return mediaContent{}, err
		}
		document := &antropic.DocumentBlockParam{}
		document.Source.OfText = &antropic.PlainTextSourceParam{Data: text}
		setDocumentTitle(document, source.Name)
		return mediaContent{document: document}, nil
	}
}

func setDocumentTitle(document *antropic.DocumentBlockParam, name string) {
	if name != "" {
		document.Title = antropic.String(name)
	}
}

// mediaBlock maps media in a user message.
func mediaBlock(media *ai.MediaPart) (antropic.ContentBlockParamUnion, error) {
	content, err := mapMedia(media)
	if err != nil {
		return antropic.ContentBlockParamUnion{}, err
	}
	return antropic.ContentBlockParamUnion{OfImage: content.image, OfDocument: content.document}, nil
}

// toolResultBlock maps a canonical tool result. A result without media keeps
// its single text block; with media, text runs and media blocks keep their
// order.
func toolResultBlock(r *ai.ToolResult) (antropic.ContentBlockParamUnion, error) {
	var content []antropic.ToolResultBlockParamContentUnion
	var text strings.Builder
	flush := func() {
		if text.Len() > 0 {
			content = append(content, antropic.ToolResultBlockParamContentUnion{OfText: &antropic.TextBlockParam{Text: text.String()}})
			text.Reset()
		}
	}
	hasMedia := mediainput.ContainsMedia(r.Parts)
	for _, part := range r.Parts {
		if err := unsupportedExtensions(part.Extensions); err != nil {
			return antropic.ContentBlockParamUnion{}, err
		}
		switch part.Kind {
		case ai.ContentText:
			text.WriteString(part.Text)
		case ai.ContentJSON:
			text.Write(part.JSON)
		case ai.ContentMedia:
			media, err := mapMedia(part.Media)
			if err != nil {
				return antropic.ContentBlockParamUnion{}, err
			}
			flush()
			content = append(content, antropic.ToolResultBlockParamContentUnion{OfImage: media.image, OfDocument: media.document})
		default:
			return antropic.ContentBlockParamUnion{}, fmt.Errorf("%w: Anthropic tool result %q", ai.ErrUnsupportedCapability, part.Kind)
		}
	}
	if !hasMedia {
		return antropic.NewToolResultBlock(r.ToolCallID, text.String(), r.IsError), nil
	}
	flush()
	return antropic.ContentBlockParamUnion{OfToolResult: &antropic.ToolResultBlockParam{
		ToolUseID: r.ToolCallID,
		Content:   content,
		IsError:   antropic.Bool(r.IsError),
	}}, nil
}
