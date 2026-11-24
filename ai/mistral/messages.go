package mistral

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/mediainput"
)

// toolResultContent maps a canonical tool result to tool message content. A
// result without media is one text chunk, sent as a plain string. With media,
// text runs and media chunks keep their order; a failed result sends its text
// as {"error": ...} first, then its media.
func toolResultContent(result *ai.ToolResult) (chatContent, error) {
	if !mediainput.ContainsMedia(result.Parts) {
		text, err := canonicalResultText(result)
		if err != nil {
			return nil, err
		}
		return textContent(text), nil
	}
	var content, media chatContent
	var text strings.Builder
	flush := func() {
		if text.Len() > 0 {
			content = append(content, chatContentChunk{Type: chunkText, Text: text.String()})
			text.Reset()
		}
	}
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
			chunk, err := mediaChunk(part.Media)
			if err != nil {
				return nil, err
			}
			if result.IsError {
				media = append(media, chunk)
				continue
			}
			flush()
			content = append(content, chunk)
		default:
			return nil, fmt.Errorf("%w: tool-result content %q", ai.ErrUnsupportedCapability, part.Kind)
		}
	}
	if result.IsError {
		encoded, _ := json.Marshal(map[string]string{"error": text.String()})
		return append(chatContent{{Type: chunkText, Text: string(encoded)}}, media...), nil
	}
	flush()
	return content, nil
}

func canonicalResultText(result *ai.ToolResult) (string, error) {
	var out strings.Builder
	for _, part := range result.Parts {
		if err := rejectRequiredExtensions(part.Extensions); err != nil {
			return "", err
		}
		switch part.Kind {
		case ai.ContentText:
			out.WriteString(part.Text)
		case ai.ContentJSON:
			out.Write(part.JSON)
		default:
			return "", fmt.Errorf("%w: tool-result content %q", ai.ErrUnsupportedCapability, part.Kind)
		}
	}
	content := out.String()
	if result.IsError {
		encoded, _ := json.Marshal(map[string]string{"error": content})
		content = string(encoded)
	}
	return content, nil
}

func rejectRequiredExtensions(extensions []ai.Extension) error {
	for _, ext := range extensions {
		if ext.Required {
			return fmt.Errorf("%w: Mistral extension %s/%s", ai.ErrUnsupportedCapability, ext.Namespace, ext.Type)
		}
	}
	return nil
}
