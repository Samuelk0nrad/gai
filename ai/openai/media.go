package openai

import (
	"cmp"
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
//   - PDFs (application/pdf) become input_file. An http(s) URL is sent as
//     file_url; inline bytes or a base64 data URL are sent as file_data with a
//     filename, MediaPart.Name or "file.pdf".
//   - Text media (text/*, application/json, +json) given as bytes or a base64
//     data URL becomes input_text wrapped in a <file> element, because OpenAI
//     has no text-file input.
//
// Other MIME types, other URI schemes and text media by URL return
// *ai.UnsupportedContentError. Nothing is dropped.

const defaultFilename = "file.pdf"

func unsupportedMedia(format string, args ...any) error {
	return mediainput.Unsupported("openai", format, args...)
}

// mediaInputContent maps one media part to a Responses input content item.
func mediaInputContent(media *ai.MediaPart) (responses.ResponseInputContentUnionParam, error) {
	source, err := mediainput.Resolve("openai", media)
	if err != nil {
		return responses.ResponseInputContentUnionParam{}, err
	}
	switch source.Kind {
	case mediainput.KindImage:
		imageURL := source.URL
		if imageURL == "" {
			imageURL = source.DataURL()
		}
		return responses.ResponseInputContentUnionParam{OfInputImage: &responses.ResponseInputImageParam{
			Detail:   responses.ResponseInputImageDetailAuto,
			ImageURL: param.NewOpt(imageURL),
		}}, nil
	case mediainput.KindPDF:
		file := &responses.ResponseInputFileParam{}
		if source.URL == "" {
			file.FileData = param.NewOpt(source.DataURL())
			file.Filename = param.NewOpt(cmp.Or(source.Name, defaultFilename))
		} else {
			file.FileURL = param.NewOpt(source.URL)
			if source.Name != "" {
				file.Filename = param.NewOpt(source.Name)
			}
		}
		return responses.ResponseInputContentUnionParam{OfInputFile: file}, nil
	default: // mediainput.KindText; Resolve rejects other kinds.
		text, err := source.TextFile("openai")
		if err != nil {
			return responses.ResponseInputContentUnionParam{}, err
		}
		return responses.ResponseInputContentParamOfInputText(text), nil
	}
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
