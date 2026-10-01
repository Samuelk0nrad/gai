package gemini

import (
	"encoding/json"
	"testing"

	"github.com/lace-ai/gai/ai"
	"google.golang.org/genai"
)

func TestCanonicalGeminiRoundTripPreservesOrderSignaturesAndRepeatedNames(t *testing.T) {
	response := &genai.GenerateContentResponse{Candidates: []*genai.Candidate{{Content: &genai.Content{Parts: []*genai.Part{
		{Text: "reason", Thought: true, ThoughtSignature: []byte("thinking-signature")},
		{FunctionCall: &genai.FunctionCall{ID: "one", Name: "lookup", Args: map[string]any{"x": 1}}, ThoughtSignature: []byte("call-signature")},
		{Text: "between"},
		{FunctionCall: &genai.FunctionCall{ID: "two", Name: "lookup", Args: map[string]any{"x": 2}}},
	}}}}}
	message, err := mapCanonicalResponse(response)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(message)
	var restored ai.Message
	if err := json.Unmarshal(raw, &restored); err != nil {
		t.Fatal(err)
	}
	req := ai.AIRequest{Prompt: "obsolete", Messages: []ai.Message{ai.TextMessage(ai.RoleSystem, "rules"), restored,
		{Role: ai.RoleTool, Parts: []ai.ContentPart{
			{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "one", Name: "lookup", Parts: ai.TextParts("first")}},
			{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "two", Name: "lookup", Parts: ai.TextParts("second"), IsError: true}},
		}},
	}}
	config, err := buildGenerateContentConfig(req)
	if err != nil {
		t.Fatal(err)
	}
	contents, err := nativeContents(req)
	if err != nil {
		t.Fatal(err)
	}
	if config.SystemInstruction.Parts[0].Text != "rules" || len(contents) != 2 || len(contents[0].Parts) != 4 || string(contents[0].Parts[0].ThoughtSignature) != "thinking-signature" || contents[0].Parts[1].FunctionCall.ID != "one" || contents[0].Parts[2].Text != "between" || contents[0].Parts[3].FunctionCall.ID != "two" || contents[1].Parts[1].FunctionResponse.ID != "two" || contents[1].Parts[1].FunctionResponse.Response["error"] != "second" {
		t.Fatalf("contents = %#v", contents)
	}
}
