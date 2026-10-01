package openai

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
			return fmt.Errorf("%w: OpenAI extension %s/%s", ai.ErrUnsupportedCapability, ext.Namespace, ext.Type)
		}
	}
	return nil
}

func chatGoogleSignature(extensions []ai.Extension) ([]byte, error) {
	var signature []byte
	for _, ext := range extensions {
		if ext.Namespace == "google" && ext.Type == "thought_signature" {
			if err := json.Unmarshal(ext.Data, &signature); err != nil {
				return nil, err
			}
		} else if ext.Required {
			return nil, fmt.Errorf("%w: OpenAI chat extension %s/%s", ai.ErrUnsupportedCapability, ext.Namespace, ext.Type)
		}
	}
	return signature, nil
}

func chatGoogleExtensions(raw string) []ai.Extension {
	var payload struct {
		ExtraContent struct {
			Google struct {
				ThoughtSignature json.RawMessage `json:"thought_signature"`
			} `json:"google"`
		} `json:"extra_content"`
	}
	if json.Unmarshal([]byte(raw), &payload) != nil || len(payload.ExtraContent.Google.ThoughtSignature) == 0 {
		return nil
	}
	return []ai.Extension{{Namespace: "google", Type: "thought_signature", Data: append(json.RawMessage(nil), payload.ExtraContent.Google.ThoughtSignature...), Required: true}}
}
