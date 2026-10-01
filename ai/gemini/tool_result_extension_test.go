package gemini

import (
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestToolResultsApplyExtensionRequirements(t *testing.T) {
	for _, namespace := range []string{"example", "google"} {
		for _, placement := range []string{"wrapper", "text", "json", "extension_only"} {
			for _, required := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/required_%t", namespace, placement, required), func(t *testing.T) {
					data, _ := json.Marshal([]byte("opaque-state"))
					extension := ai.Extension{Namespace: namespace, Type: "unknown", Data: data, Required: required}
					if namespace == "google" {
						extension.Type = "thought_signature"
					}
					result := ai.ContentPart{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{
						ToolCallID: "call_1", Name: "lookup", IsError: true,
						Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "result"}, {Kind: ai.ContentJSON, JSON: json.RawMessage(`{"ok":false}`)}},
					}}
					switch placement {
					case "wrapper":
						result.Extensions = []ai.Extension{extension}
					case "text":
						result.ToolResult.Parts[0].Extensions = []ai.Extension{extension}
					case "json":
						result.ToolResult.Parts[1].Extensions = []ai.Extension{extension}
					case "extension_only":
						result.ToolResult.Parts = append(result.ToolResult.Parts, ai.ContentPart{Kind: ai.ContentExtension, Extensions: []ai.Extension{extension}})
					}
					req := ai.AIRequest{Messages: []ai.Message{
						{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "call_1", Type: "function", Name: "lookup", Args: json.RawMessage(`{}`)}}}},
						{Role: ai.RoleTool, Parts: []ai.ContentPart{result}},
					}}
					contents, err := nativeContents(req)
					if required {
						if !errors.Is(err, ai.ErrUnsupportedCapability) {
							t.Fatalf("error = %v, want unsupported capability", err)
						}
						return
					}
					if err != nil {
						t.Fatal(err)
					}
					if len(contents) != 2 || len(contents[1].Parts) != 1 || contents[1].Parts[0].FunctionResponse == nil {
						t.Fatalf("contents = %#v", contents)
					}
					response := contents[1].Parts[0].FunctionResponse
					if response.ID != "call_1" || response.Name != "lookup" || response.Response["error"] != `result{"ok":false}` {
						t.Fatalf("function response = %#v", response)
					}
				})
			}
		}
	}
}
