package anthropic

import (
	"github.com/lace-ai/gai/ai"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
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
