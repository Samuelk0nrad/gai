package ai_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func canonicalCall(id string) ai.ContentPart {
	return ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{
		ID: id, Type: "function", Name: "search", Args: json.RawMessage(`{"q":"x"}`),
	}}
}

func canonicalResult(id, name string) ai.ContentPart {
	return ai.ContentPart{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{
		ToolCallID: id, Name: name, Parts: ai.TextParts("ok"),
	}}
}

func TestAIRequestMatchesRepeatedToolNamesByCallID(t *testing.T) {
	req := ai.AIRequest{Messages: []ai.Message{
		ai.TextMessage(ai.RoleUser, "search twice"),
		{Role: ai.RoleAssistant, Parts: []ai.ContentPart{canonicalCall("first"), canonicalCall("second")}},
		// Results can arrive in a different order; names alone cannot identify them.
		{Role: ai.RoleTool, Parts: []ai.ContentPart{canonicalResult("second", "search"), canonicalResult("first", "search")}},
	}}
	if err := req.ValidateMessages(); err != nil {
		t.Fatal(err)
	}
}

func TestAIRequestRejectsInvalidToolReferences(t *testing.T) {
	tests := []struct {
		name  string
		parts []ai.Message
	}{
		{"mismatched name", []ai.Message{
			{Role: ai.RoleAssistant, Parts: []ai.ContentPart{canonicalCall("first")}},
			{Role: ai.RoleTool, Parts: []ai.ContentPart{canonicalResult("first", "other")}},
		}},
		{"duplicate result", []ai.Message{
			{Role: ai.RoleAssistant, Parts: []ai.ContentPart{canonicalCall("first")}},
			{Role: ai.RoleTool, Parts: []ai.ContentPart{canonicalResult("first", "search"), canonicalResult("first", "search")}},
		}},
		{"duplicate call id", []ai.Message{
			{Role: ai.RoleAssistant, Parts: []ai.ContentPart{canonicalCall("first"), canonicalCall("first")}},
		}},
		{"result precedes call", []ai.Message{
			{Role: ai.RoleTool, Parts: []ai.ContentPart{canonicalResult("first", "search")}},
			{Role: ai.RoleAssistant, Parts: []ai.ContentPart{canonicalCall("first")}},
		}},
		{"unknown id with known name", []ai.Message{
			{Role: ai.RoleAssistant, Parts: []ai.ContentPart{canonicalCall("first")}},
			{Role: ai.RoleTool, Parts: []ai.ContentPart{canonicalResult("missing", "search")}},
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := (ai.AIRequest{Messages: tt.parts}).ValidateMessages(); err == nil {
				t.Fatal("invalid tool reference accepted")
			}
		})
	}
}

func TestAIRequestCopyIsolatesAllMutableFields(t *testing.T) {
	ext := ai.Extension{Namespace: "future", Type: "state", Data: json.RawMessage(`{"v":1}`), Required: true}
	call := canonicalCall("first")
	call.ToolCall.ThoughtSignature = []byte{1, 2, 3}
	call.ToolCall.Extensions = []ai.Extension{ext}
	call.Extensions = []ai.Extension{ext}
	result := canonicalResult("first", "search")
	result.ToolResult.Parts = []ai.ContentPart{
		{Kind: ai.ContentJSON, JSON: json.RawMessage(`{"ok":true}`)},
		{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", Data: []byte{4, 5, 6}}, Extensions: []ai.Extension{ext}},
	}
	req := ai.AIRequest{
		Prompt: "legacy", MaxTokens: 128,
		Messages: []ai.Message{
			{Role: ai.RoleAssistant, Parts: []ai.ContentPart{call}, Extensions: []ai.Extension{ext}},
			{Role: ai.RoleTool, Parts: []ai.ContentPart{result}},
		},
		Tools:          []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice:     ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"search"}},
		ResponseFormat: ai.ResponseFormat{Type: ai.ResponseFormatJSONSchema, Name: "answer", Schema: json.RawMessage(`{"type":"object"}`)},
		Reasoning:      ai.ReasoningConfig{Enabled: true, IncludeThoughts: true, BudgetTokens: 10, Effort: ai.ReasoningEffortLow},
	}
	if err := req.Validate(); err != nil {
		t.Fatal(err)
	}
	before, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	copy := req.Copy()
	if !reflect.DeepEqual(req, copy) {
		t.Fatal("copy changed request contents")
	}
	copy.Messages[0].Role = ai.RoleUser
	copy.Messages[0].Extensions[0].Data[5] = '9'
	copy.Messages[0].Parts[0].Extensions[0].Namespace = "changed"
	copy.Messages[0].Parts[0].Extensions[0].Data[5] = '9'
	copy.Messages[0].Parts[0].ToolCall.ID = "changed"
	copy.Messages[0].Parts[0].ToolCall.Args[6] = 'y'
	copy.Messages[0].Parts[0].ToolCall.ThoughtSignature[0] = 9
	copy.Messages[0].Parts[0].ToolCall.Extensions[0].Data[5] = '9'
	copy.Messages[1].Parts[0].ToolResult.IsError = true
	copy.Messages[1].Parts[0].ToolResult.Parts[0].JSON[6] = 'f'
	copy.Messages[1].Parts[0].ToolResult.Parts[1].Media.MIMEType = "image/jpeg"
	copy.Messages[1].Parts[0].ToolResult.Parts[1].Media.Data[0] = 9
	copy.Messages[1].Parts[0].ToolResult.Parts[1].Extensions[0].Data[5] = '9'
	copy.Tools[0].Name = "changed"
	copy.Tools[0].Parameters[2] = 'T'
	copy.ToolChoice.Names[0] = "changed"
	copy.ResponseFormat.Schema[2] = 'T'
	after, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatalf("mutating copied request changed original:\nbefore %s\nafter %s", before, after)
	}
}

func TestAIRequestNormalizedUsesAuthoritativeMessages(t *testing.T) {
	req := ai.AIRequest{Prompt: "stale flattened prompt", Messages: []ai.Message{ai.TextMessage(ai.RoleSystem, "instructions"), ai.TextMessage(ai.RoleUser, "question")}, MaxTokens: 42}
	got, err := req.Normalized()
	if err != nil {
		t.Fatal(err)
	}
	if got.Prompt != "" || got.MaxTokens != 42 || !reflect.DeepEqual(got.Messages, req.Messages) {
		t.Fatalf("normalized request = %#v", got)
	}
	got.Messages[1].Parts[0].Text = "changed"
	if req.Messages[1].Text() != "question" || req.Prompt != "stale flattened prompt" {
		t.Fatal("normalization modified input")
	}
}

func TestAIRequestNormalizedLiftsLegacyPrompt(t *testing.T) {
	got, err := (ai.AIRequest{Prompt: "legacy", MaxTokens: 42}).Normalized()
	if err != nil {
		t.Fatal(err)
	}
	want := []ai.Message{ai.TextMessage(ai.RoleUser, "legacy")}
	if got.Prompt != "" || got.MaxTokens != 42 || !reflect.DeepEqual(got.Messages, want) {
		t.Fatalf("normalized request = %#v", got)
	}
}
