package gemini

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestGenerateRequiresCanonicalMessagesBeforeTransport(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { requests.Add(1); w.WriteHeader(http.StatusBadRequest) }))
	defer server.Close()
	provider := New("test-key", nil)
	provider.baseURL = server.URL
	provider.httpClient = server.Client()
	model, err := provider.Model("test-model")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{}); err == nil {
		t.Fatal("empty canonical request accepted")
	}
	var streamErr error
	for token := range model.GenerateStream(t.Context(), ai.AIRequest{}) {
		if err := token.Validate(); err != nil {
			t.Fatal(err)
		}
		if token.Err != nil {
			streamErr = token.Err
		}
	}
	if streamErr == nil || requests.Load() != 0 {
		t.Fatalf("stream error = %v, provider requests = %d", streamErr, requests.Load())
	}
}

func TestCanonicalLeadingSystemMessagesPreserveOrder(t *testing.T) {
	req := ai.AIRequest{Messages: []ai.Message{
		{Role: ai.RoleSystem, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "first"}, {Kind: ai.ContentJSON, JSON: []byte(`{"second":true}`)}}},
		ai.TextMessage(ai.RoleSystem, "third"),
		ai.TextMessage(ai.RoleUser, "question"),
		ai.TextMessage(ai.RoleAssistant, "answer"),
	}}
	config, err := buildGenerateContentConfig(req)
	if err != nil {
		t.Fatal(err)
	}
	if config == nil || config.SystemInstruction == nil {
		t.Fatal("system instruction missing")
	}
	parts := config.SystemInstruction.Parts
	if len(parts) != 3 || parts[0].Text != "first" || parts[1].Text != `{"second":true}` || parts[2].Text != "third" {
		t.Fatalf("system parts = %#v", parts)
	}
	contents, err := nativeContents(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(contents) != 2 || contents[0].Role != "user" || contents[1].Role != "model" || len(contents[0].Parts) != 1 || len(contents[1].Parts) != 1 || contents[0].Parts[0].Text != "question" || contents[1].Parts[0].Text != "answer" {
		t.Fatalf("conversation = %#v", contents)
	}
}

func TestGenerateRejectsNonLeadingSystemBeforeTransport(t *testing.T) {
	var requests atomic.Int32
	model := emptyPartModel(t, func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusBadRequest)
	})
	req := ai.AIRequest{Messages: []ai.Message{
		ai.TextMessage(ai.RoleSystem, "initial instructions"),
		ai.TextMessage(ai.RoleUser, "question"),
		ai.TextMessage(ai.RoleSystem, "later instructions"),
	}}
	if _, err := model.Generate(t.Context(), req); !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Errorf("Generate error = %v", err)
	}
	var streamErr error
	for token := range model.GenerateStream(t.Context(), req) {
		if err := token.Validate(); err != nil {
			t.Fatal(err)
		}
		if token.Err != nil {
			streamErr = token.Err
		}
	}
	if !errors.Is(streamErr, ai.ErrUnsupportedCapability) || requests.Load() != 0 {
		t.Fatalf("stream error = %v, provider requests = %d", streamErr, requests.Load())
	}
}
