package mistral

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/lace-ai/gai/ai"
)

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
