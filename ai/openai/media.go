package openai

import (
	"fmt"
	"strings"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/mediainput"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
)

// Media input on the Responses transport.
//
//   - Images (image/jpeg, image/png, image/webp, image/gif) become input_image
//     with detail "auto". An http(s) URL or base64 data URL is sent as is;
//     inline bytes become a base64 data URL.
//   - PDFs and Office files (.docx, .pptx, .xlsx) become input_file. An
//     http(s) URL is sent as file_url; inline bytes or a base64 data URL are
//     sent as file_data with a filename, MediaPart.Name or "file.<ext>".
//     OpenAI reads only the text of Office files.
//   - Text media (text/*, application/json, +json) given as bytes or a base64
//     data URL becomes input_text wrapped in a <file> element, because OpenAI
//     has no text-file input.
//
// Audio, video, other MIME types, other URI schemes and text media by URL
// return *ai.UnsupportedContentError. Nothing is dropped. The Chat
// Completions transport sends no media.

var (
	openAIImageTypes = mediainput.Types{"image/jpeg", "image/png", "image/webp", "image/gif"}
	// openAIFileExtensions maps input_file MIME types to the file extension
	// OpenAI uses to detect the type.
	openAIFileExtensions = map[string]string{
		mediainput.MIMEPDF: ".pdf",
		"application/vnd.openxmlformats-officedocument.wordprocessingml.document":   ".docx",
		"application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
		"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":         ".xlsx",
	}
)

const openAISupportedTypes = "image/jpeg, image/png, image/webp, image/gif, application/pdf, .docx, .pptx and .xlsx files, text/*, application/json"

func unsupportedMedia(format string, args ...any) error {
	return mediainput.Unsupported("openai", format, args...)
}

// AcceptsMedia reports whether the model's transport sends media. Only the
// Responses transport sends media; see the package media notes.
func (m *Model) AcceptsMedia(media ai.MediaPart, _ ai.MediaPlacement) error {
	if m.provider.transport != TransportResponses {
		return unsupportedMedia("media input requires the Responses transport (WithResponsesTransport)")
	}
	_, err := resolveOpenAIMedia(&media)
	return err
}

// resolveOpenAIMedia resolves media and checks its type and source.
func resolveOpenAIMedia(media *ai.MediaPart) (mediainput.Source, error) {
	source, err := mediainput.Resolve("openai", media)
	if err != nil {
		return source, err
	}
	switch {
	case openAIImageTypes.Has(source.MIMEType), openAIFileExtensions[source.MIMEType] != "":
		return source, nil
	case source.Kind == mediainput.KindText:
		return source, source.CheckText("openai")
	default:
		return source, unsupportedMedia("MIME type %q (supported: %s)", media.MIMEType, openAISupportedTypes)
	}
}

// mediaInputContent maps one media part to a Responses input content item.
func mediaInputContent(media *ai.MediaPart) (responses.ResponseInputContentUnionParam, error) {
	source, err := resolveOpenAIMedia(media)
	if err != nil {
		return responses.ResponseInputContentUnionParam{}, err
	}
	if openAIImageTypes.Has(source.MIMEType) {
		imageURL := source.URL
		if source.Inline() {
			imageURL = source.DataURL()
		}
		return responses.ResponseInputContentUnionParam{OfInputImage: &responses.ResponseInputImageParam{
			Detail:   responses.ResponseInputImageDetailAuto,
			ImageURL: param.NewOpt(imageURL),
		}}, nil
	}
	if ext := openAIFileExtensions[source.MIMEType]; ext != "" {
		file := &responses.ResponseInputFileParam{}
		if source.Inline() {
			file.FileData = param.NewOpt(source.DataURL())
			file.Filename = param.NewOpt(source.Filename(ext))
		} else {
			file.FileURL = param.NewOpt(source.URL)
			if source.Name != "" {
				file.Filename = param.NewOpt(source.Name)
			}
		}
		return responses.ResponseInputContentUnionParam{OfInputFile: file}, nil
	}
	text, err := source.TextFile("openai")
	if err != nil {
		return responses.ResponseInputContentUnionParam{}, err
	}
	return responses.ResponseInputContentParamOfInputText(text), nil
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
