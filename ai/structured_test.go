package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/structured"
)

type structuredScriptModel struct {
	mu        sync.Mutex
	responses [][]ai.Token
	requests  []ai.AIRequest
}

func (m *structuredScriptModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	m.mu.Lock()
	m.requests = append(m.requests, req.Copy())
	var tokens []ai.Token
	if len(m.responses) > 0 {
		tokens = m.responses[0]
		m.responses = m.responses[1:]
	}
	m.mu.Unlock()
	out := make(chan ai.Token, len(tokens))
	for _, token := range tokens {
		out <- token.Clone()
	}
	close(out)
	return out
}

func textToken(text string) ai.Token {
	return ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: text}}
}

func TestGenerateStructuredRetriesInvalidOutputAndReturnsExactJSON(t *testing.T) {
	schemaJSON := json.RawMessage(`{"type":"object","properties":{"count":{"type":"integer","minimum":1}},"required":["count"],"additionalProperties":false}`)
	schema, err := structured.Raw("count_answer", schemaJSON)
	if err != nil {
		t.Fatal(err)
	}
	type answer struct {
		Count int `json:"count"`
	}
	model := &structuredScriptModel{responses: [][]ai.Token{
		{textToken(`{"count":0}`)},
		{textToken(" {\n  \"count\": 2\n} ")},
	}}
	req := ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "count")}}

	result, err := ai.GenerateStructured(t.Context(), model, req, structured.Spec[answer]{
		Schema: schema,
		Retry:  structured.ValidationRetryPolicy{MaxRetries: 1},
	})
	if err != nil {
		t.Fatalf("GenerateStructured: %v", err)
	}
	if result.Value.Count != 2 || string(result.JSON) != " {\n  \"count\": 2\n} " {
		t.Fatalf("result = %#v", result)
	}
	if len(model.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(model.requests))
	}
	for i, got := range model.requests {
		if got.ResponseFormat.Type != ai.ResponseFormatJSONSchema || got.ResponseFormat.Name != "count_answer" || string(got.ResponseFormat.Schema) != string(schemaJSON) {
			t.Fatalf("request %d response format = %#v", i, got.ResponseFormat)
		}
	}
	if len(model.requests[1].Messages) != 2 || model.requests[1].Messages[1].Role != ai.RoleUser {
		t.Fatalf("retry messages = %#v", model.requests[1].Messages)
	}
	if len(req.Messages) != 1 || req.ResponseFormat.Type != "" {
		t.Fatalf("caller request mutated: %#v", req)
	}
}

func TestGenerateStructuredDoesNotRetryByDefault(t *testing.T) {
	schema, err := structured.Raw("answer", json.RawMessage(`{"type":"object","required":["ok"],"properties":{"ok":{"type":"boolean"}}}`))
	if err != nil {
		t.Fatal(err)
	}
	model := &structuredScriptModel{responses: [][]ai.Token{{textToken(`{"ok":"yes"}`)}, {textToken(`{"ok":true}`)}}}

	_, err = ai.GenerateStructured(t.Context(), model, ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "answer")}}, structured.Spec[struct {
		OK bool `json:"ok"`
	}]{Schema: schema})
	var validation *structured.ValidationError
	if !errors.As(err, &validation) {
		t.Fatalf("error = %T %v, want ValidationError", err, err)
	}
	if len(model.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(model.requests))
	}
}

func TestGenerateStructuredAcceptsCanonicalJSONContent(t *testing.T) {
	schema, err := structured.Raw("answer", json.RawMessage(`{"type":"object","required":["ok"],"properties":{"ok":{"type":"boolean"}}}`))
	if err != nil {
		t.Fatal(err)
	}
	raw := json.RawMessage(`{"ok":true}`)
	model := &structuredScriptModel{responses: [][]ai.Token{{{
		Part: &ai.ContentPart{Kind: ai.ContentJSON, JSON: raw},
	}}}}

	result, err := ai.GenerateStructured(t.Context(), model, ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "answer")}}, structured.Spec[struct {
		OK bool `json:"ok"`
	}]{Schema: schema})
	if err != nil {
		t.Fatalf("GenerateStructured: %v", err)
	}
	if !result.Value.OK || string(result.JSON) != string(raw) {
		t.Fatalf("result = %#v", result)
	}
}

func TestGenerateStructuredReturnsStreamErrorAfterCompletion(t *testing.T) {
	schema, err := structured.Raw("answer", json.RawMessage(`{"type":"object"}`))
	if err != nil {
		t.Fatal(err)
	}
	streamErr := errors.New("stream failed")
	model := &structuredScriptModel{responses: [][]ai.Token{{
		textToken(`{}`),
		{Completion: &ai.Completion{UsageReported: true}},
		{Err: streamErr},
	}}}

	_, err = ai.GenerateStructured(t.Context(), model, ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "answer")}}, structured.Spec[map[string]any]{Schema: schema})
	if !errors.Is(err, streamErr) {
		t.Fatalf("error = %v, want stream error", err)
	}
}
