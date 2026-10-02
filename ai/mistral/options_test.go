package mistral_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/ai/mistral"
)

func TestWithBaseURLTrailingSlashes(t *testing.T) {
	for _, prefix := range []string{"", "/proxy"} {
		for _, suffix := range []string{"", "/", "///"} {
			for _, operation := range []string{"generate", "stream"} {
				t.Run(prefix+suffix+"/"+operation, func(t *testing.T) {
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						if r.Method != http.MethodPost || r.URL.Path != prefix+"/v1/chat/completions" {
							t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
							http.Error(w, "unexpected method or path", http.StatusBadRequest)
							return
						}
						var body struct {
							Model    string `json:"model"`
							Messages []struct {
								Content string `json:"content"`
							} `json:"messages"`
							Stream bool `json:"stream"`
						}
						if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
							t.Errorf("decode request: %v", err)
							http.Error(w, "invalid body", http.StatusBadRequest)
							return
						}
						if body.Model != mistral.MistralSmallLatest || len(body.Messages) != 1 || body.Messages[0].Content != "hello" || body.Stream != (operation == "stream") {
							t.Errorf("unexpected body: %+v", body)
						}
						if operation == "stream" {
							w.Header().Set("Content-Type", "text/event-stream")
							_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")
							return
						}
						w.Header().Set("Content-Type", "application/json")
						_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":1}}`)
					}))
					defer server.Close()
					provider := mistral.New("test-key", nil, mistral.WithBaseURL(server.URL+prefix+suffix), mistral.WithHTTPClient(server.Client()))
					model, err := provider.TypedModel(mistral.MistralSmallLatest)
					if err != nil {
						t.Fatal(err)
					}
					switch operation {
					case "generate":
						response, err := model.Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}})
						if err != nil {
							t.Fatal(err)
						}
						if response.Text() != "ok" {
							t.Fatalf("unexpected response: %q", response.Text())
						}
					case "stream":
						var text string
						for token := range model.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
							if token.Err != nil {
								t.Error(token.Err)
							}
							if token.Type() == ai.TokenTypeText {
								text += token.Text()
							}
						}
						if text != "ok" {
							t.Fatalf("unexpected stream: %q", text)
						}

					}
				})
			}
		}
	}
}
