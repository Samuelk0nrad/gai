package ollama

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestBuildChatRequestMapsCanonicalToolRoundTripAndOptions(t *testing.T) {
	model := mustModel(t, WithToolSupport(ai.FeatureSupportSupported), WithOptions(Options{NumCtx: intPtr(8192)}))
	tool, err := ai.NewToolDefinition("weather", "Get the weather", json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`))
	if err != nil {
		t.Fatal(err)
	}
	call := ai.ToolCall{ID: "call-1", Type: "function", Name: "weather", Args: json.RawMessage(`{"city":"Tokyo"}`)}
	req := ai.AIRequest{
		MaxTokens: 123,
		Messages: []ai.Message{
			ai.TextMessage(ai.RoleSystem, "Be concise."),
			ai.TextMessage(ai.RoleUser, "Weather?"),
			{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "Checking."}, {Kind: ai.ContentToolCall, ToolCall: &call}}},
			{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "call-1", Name: "weather", Parts: []ai.ContentPart{{Kind: ai.ContentJSON, JSON: json.RawMessage(`{"celsius":21}`)}}}}}},
		},
		Tools:      []ai.ToolDefinition{tool},
		ToolChoice: ai.ToolChoice{Mode: ai.ToolChoiceAuto},
	}

	payload, err := model.chatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if payload.Model != "test-model" || payload.Stream == nil || !*payload.Stream || payload.Think == nil || payload.Think.Value != false {
		t.Fatalf("request identity = %#v", payload)
	}
	if !reflect.DeepEqual(payload.Options, map[string]any{"num_ctx": 8192, "num_predict": 123}) {
		t.Fatalf("options = %#v", payload.Options)
	}
	if len(payload.Messages) != 4 {
		t.Fatalf("messages = %#v", payload.Messages)
	}
	assistant := payload.Messages[2]
	arguments, err := json.Marshal(assistant.ToolCalls[0].Function.Arguments)
	if err != nil {
		t.Fatal(err)
	}
	if assistant.Content != "Checking." || len(assistant.ToolCalls) != 1 || assistant.ToolCalls[0].ID != "call-1" || string(arguments) != `{"city":"Tokyo"}` {
		t.Fatalf("assistant = %#v", assistant)
	}
	result := payload.Messages[3]
	if result.Role != "tool" || result.ToolName != "weather" || result.ToolCallID != "call-1" || result.Content != `{"celsius":21}` {
		t.Fatalf("result = %#v", result)
	}
	parameters, err := json.Marshal(payload.Tools[0].Function.Parameters)
	if err != nil {
		t.Fatal(err)
	}
	var gotParameters, wantParameters any
	if err := json.Unmarshal(parameters, &gotParameters); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(tool.Parameters, &wantParameters); err != nil {
		t.Fatal(err)
	}
	if len(payload.Tools) != 1 || payload.Tools[0].Function.Name != "weather" || !reflect.DeepEqual(gotParameters, wantParameters) {
		t.Fatalf("tools = %#v", payload.Tools)
	}
}

func TestBuildChatRequestRejectsUnsupportedOrInvalidControls(t *testing.T) {
	tests := []struct {
		name  string
		model *Model
		req   ai.AIRequest
	}{
		{"required tool choice", mustModel(t, WithToolSupport(ai.FeatureSupportSupported)), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}, Tools: []ai.ToolDefinition{{Name: "x", Description: "x", Parameters: json.RawMessage(`{"type":"object"}`)}}, ToolChoice: ai.ToolChoice{Mode: ai.ToolChoiceRequired}}},
		{"reasoning", mustModel(t), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}, Reasoning: ai.ReasoningConfig{Enabled: true}}},
		{"negative max tokens", mustModel(t), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}, MaxTokens: -1}},
		{"invalid num ctx", mustModel(t, WithOptions(Options{NumCtx: intPtr(0)})), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.model.chatRequest(tt.req); err == nil {
				t.Fatal("expected error")
			} else if tt.name == "reasoning" && !errors.Is(err, ai.ErrUnsupportedCapability) {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func mustModel(t *testing.T, options ...ModelOption) *Model {
	t.Helper()
	model, err := New(nil).TypedModel("test-model", options...)
	if err != nil {
		t.Fatal(err)
	}
	return model
}
