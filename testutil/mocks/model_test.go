package mocks_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestMockModelStreamsOrderedCanonicalParts(t *testing.T) {
	message := ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{
		{Kind: ai.ContentReasoning, Text: "think", Extensions: []ai.Extension{{Namespace: "provider", Type: "signature", Data: json.RawMessage(`"opaque"`), Required: true}}},
		{Kind: ai.ContentText, Text: "before"},
		{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "first", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"input"}`)}},
		{Kind: ai.ContentJSON, JSON: json.RawMessage(`{"ok":true}`)},
		{Kind: ai.ContentText, Text: "after"},
	}}
	model := &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Message: message, ReasoningTokens: 3}}}}
	var tokens []ai.Token
	var response ai.AIResponse
	for token := range model.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "question")}}) {
		tokens = append(tokens, token)
		response.AppendToken(token)
	}
	if !reflect.DeepEqual(response.Message, message) {
		t.Fatalf("mock changed canonical message: %#v", response.Message)
	}
	if response.ReasoningTokens != 3 {
		t.Fatalf("reasoning usage=%d, want 3", response.ReasoningTokens)
	}
	tokens[0].Part.Extensions[0].Data[1] = 'X'
	tokens[2].Part.ToolCall.Args[9] = 'X'
	tokens[3].Part.JSON[2] = 'X'
	if string(message.Parts[0].Extensions[0].Data) != `"opaque"` || string(message.Parts[2].ToolCall.Args) != `{"text":"input"}` || string(message.Parts[3].JSON) != `{"ok":true}` {
		t.Fatal("mock token aliases scripted response")
	}
}
