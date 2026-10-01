package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	asdk "github.com/anthropics/anthropic-sdk-go"
	aparam "github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/ai/anthropic"
	"github.com/lace-ai/gai/ai/gemini"
	"github.com/lace-ai/gai/ai/mistral"
	"github.com/lace-ai/gai/ai/openai"
	osdk "github.com/openai/openai-go"
	oparam "github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"google.golang.org/genai"
)

// These tests intentionally use external packages: native APIs must be usable
// without package-private test seams or assertions from ai.Model.
func configuredNativeModel(t *testing.T, provider, endpoint string, hookError error) ai.Model {
	t.Helper()
	var model ai.Model
	var err error
	switch provider {
	case "openai":
		model, err = openai.New("test-key", nil, openai.WithBaseURL(endpoint), openai.WithHTTPClient(&http.Client{})).TypedModel("test-model",
			openai.WithChatCompletionParams(func(p *osdk.ChatCompletionNewParams) error {
				if p.Temperature.Valid() {
					return fmt.Errorf("parameters reused")
				}
				p.Temperature = oparam.NewOpt(0.25)
				p.Model = "must-be-restored"
				p.StreamOptions.IncludeUsage = oparam.NewOpt(false)
				return hookError
			}))
	case "responses":
		model, err = openai.New("test-key", nil, openai.WithBaseURL(endpoint), openai.WithResponsesTransport()).TypedModel("test-model",
			openai.WithResponsesParams(func(p *responses.ResponseNewParams) error {
				if p.Temperature.Valid() {
					return fmt.Errorf("parameters reused")
				}
				p.Temperature = oparam.NewOpt(0.25)
				p.Model = "must-be-restored"
				return hookError
			}))
	case "anthropic":
		model, err = anthropic.New("test-key", nil, anthropic.WithBaseURL(endpoint), anthropic.WithHTTPClient(&http.Client{})).TypedModel("test-model",
			anthropic.WithMessageParams(func(p *asdk.MessageNewParams) error {
				if p.Temperature.Valid() {
					return fmt.Errorf("parameters reused")
				}
				p.Temperature = aparam.NewOpt(0.25)
				p.Model = "must-be-restored"
				return hookError
			}))
	case "gemini":
		model, err = gemini.New("test-key", nil, gemini.WithBaseURL(endpoint), gemini.WithHTTPClient(&http.Client{})).TypedModel("test-model",
			gemini.WithGenerateContentConfig(func(p *genai.GenerateContentConfig) error {
				if p.Temperature != nil {
					return fmt.Errorf("config reused")
				}
				temperature := float32(0.25)
				p.Temperature = &temperature
				return hookError
			}))
	case "mistral":
		temperature, seed, safe := 0.25, 0, false
		option := mistral.WithChatCompletionOptions(mistral.ChatCompletionOptions{Temperature: &temperature, RandomSeed: &seed, SafePrompt: &safe})
		temperature = 0.9 // Construction of the option snapshots caller-owned values.
		model, err = mistral.New("test-key", nil, mistral.WithBaseURL(endpoint), mistral.WithHTTPClient(&http.Client{})).TypedModel("test-model", option)
	}
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func writeNativeFixture(w http.ResponseWriter, provider string, streaming bool) {
	w.Header().Set("Content-Type", "application/json")
	body := `{"id":"native-id","model":"test-model","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"system_fingerprint":"native-fingerprint","usage":{"prompt_tokens":1,"completion_tokens":2}}`
	switch provider {
	case "anthropic":
		body = `{"id":"native-id","type":"message","model":"test-model","role":"assistant","content":[{"type":"text","text":"ok"},{"type":"thinking","thinking":"thought","signature":"native-signature"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}}`
	case "gemini":
		body = `{"responseId":"native-id","modelVersion":"test-model","candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["native-query"]}}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2}}`
	case "responses":
		body = `{"id":"native-id","object":"response","status":"completed","model":"test-model","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":2}}`
	}
	if !streaming {
		_, _ = io.WriteString(w, body)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	switch provider {
	case "anthropic":
		_, _ = io.WriteString(w, `event: message_start
data: {"type":"message_start","message":{"id":"native-id","type":"message","role":"assistant","model":"test-model","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":"thought"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"native-signature"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}

`)
	case "gemini":
		_, _ = fmt.Fprintf(w, "data: %s\n\n", body)
	case "responses":
		_, _ = fmt.Fprintf(w, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":%s}\n\n", body)
	default:
		_, _ = io.WriteString(w, "data: {\"id\":\"native-id\",\"model\":\"test-model\",\"system_fingerprint\":\"native-fingerprint\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}
}

func TestNativeOptionsThroughPortableCalls(t *testing.T) {
	for _, provider := range []string{"openai", "responses", "anthropic", "gemini", "mistral"} {
		t.Run(provider, func(t *testing.T) {
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					w.WriteHeader(400)
					return
				}
				streaming, _ := body["stream"].(bool)
				settings := body
				if provider == "gemini" {
					streaming = strings.Contains(r.URL.Path, ":streamGenerateContent")
					settings, _ = body["generationConfig"].(map[string]any)
					if !strings.Contains(r.URL.Path, "test-model:") {
						t.Errorf("model path = %s", r.URL.Path)
					}
				} else if body["model"] != "test-model" {
					t.Errorf("model = %v", body["model"])
				}
				if settings["temperature"] != 0.25 {
					t.Errorf("native settings missing: %#v", body)
				}
				if provider == "mistral" && (body["random_seed"] != float64(0) || body["safe_prompt"] != false) {
					t.Errorf("explicit zero settings lost: %#v", body)
				}
				if provider == "openai" && !streaming && body["stream_options"] != nil {
					t.Error("synchronous request contains stream_options")
				}
				if provider == "openai" && streaming {
					options, _ := body["stream_options"].(map[string]any)
					if options["include_usage"] != true {
						t.Error("streaming usage invariant lost")
					}
				}
				writeNativeFixture(w, provider, streaming)
			}))
			defer server.Close()
			model := configuredNativeModel(t, provider, server.URL, nil)
			var wg sync.WaitGroup
			for i := 0; i < 8; i++ {
				wg.Add(1)
				go func(streaming bool) {
					defer wg.Done()
					req := ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}
					if !streaming {
						result, err := model.(ai.ModelGenerator).Generate(t.Context(), req)
						if err != nil {
							t.Error(err)
							return
						}
						if result.Text() != "ok" {
							t.Errorf("text = %q", result.Text())
						}
						return
					}
					var collected ai.AIResponse
					for token := range model.GenerateStream(t.Context(), req) {
						if token.Err != nil {
							t.Error(token.Err)
						}
						collected.AppendToken(token)
					}
					if collected.Text() != "ok" {
						t.Errorf("stream text = %q", collected.Text())
					}
				}(i%2 == 0)
			}
			wg.Wait()
			if requests.Load() != 8 {
				t.Errorf("requests = %d; generation must not discover models", requests.Load())
			}
		})
	}
}

func TestNativeHookErrorsStopBeforeTransport(t *testing.T) {
	sentinel := errors.New("native hook failed")
	for _, provider := range []string{"openai", "responses", "anthropic", "gemini"} {
		t.Run(provider, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { t.Error("unexpected request"); w.WriteHeader(500) }))
			defer server.Close()
			model := configuredNativeModel(t, provider, server.URL, sentinel)
			if _, err := model.(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}); !errors.Is(err, sentinel) {
				t.Errorf("error = %v", err)
			}
			found := false
			for token := range model.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
				if errors.Is(token.Err, sentinel) {
					found = true
				}
			}
			if !found {
				t.Error("missing stream hook error")
			}
		})
	}
}

func TestProviderModelErrorsReturnNilInterface(t *testing.T) {
	for _, p := range []ai.Provider{openai.New("key", nil), anthropic.New("key", nil), gemini.New("key", nil), mistral.New("key", nil)} {
		if m, err := p.Model(" "); m != nil || !errors.Is(err, ai.ErrModelNotFound) {
			t.Errorf("%s: model=%v err=%v", p.Name(), m, err)
		}
	}
}

func TestConfiguredSDKClientsPreserveNativeTypes(t *testing.T) {
	for _, provider := range []string{"openai", "anthropic", "gemini"} {
		t.Run(provider, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					w.WriteHeader(400)
					return
				}
				if provider == "gemini" {
					config, _ := body["generationConfig"].(map[string]any)
					if config["temperature"] != 0.5 || r.Header.Get("x-goog-api-key") != "test-key" {
						t.Errorf("native Gemini request: %#v", body)
					}
				} else {
					if body["temperature"] != 0.5 {
						t.Errorf("native request: %#v", body)
					}
					if provider == "anthropic" && r.Header.Get("x-api-key") != "test-key" {
						t.Error("missing Anthropic auth")
					}
					if provider == "openai" && r.Header.Get("Authorization") != "Bearer test-key" {
						t.Error("missing OpenAI auth")
					}
				}
				streaming, _ := body["stream"].(bool)
				writeNativeFixture(w, provider, streaming || strings.Contains(r.URL.Path, ":streamGenerateContent"))
			}))
			defer server.Close()
			switch provider {
			case "openai":
				client, err := openai.New("test-key", nil, openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client())).SDKClient()
				if err != nil {
					t.Fatal(err)
				}
				result, err := client.Chat.Completions.New(t.Context(), osdk.ChatCompletionNewParams{Model: "native-model", Messages: []osdk.ChatCompletionMessageParamUnion{osdk.SystemMessage("native system message")}, Temperature: oparam.NewOpt(0.5)})
				if err != nil {
					t.Fatal(err)
				}
				if result.SystemFingerprint != "native-fingerprint" {
					t.Errorf("native response lost: %+v", result)
				}
				stream := client.Chat.Completions.NewStreaming(t.Context(), osdk.ChatCompletionNewParams{Model: "native-model", Messages: []osdk.ChatCompletionMessageParamUnion{osdk.UserMessage("hello")}, Temperature: oparam.NewOpt(0.5)})
				defer stream.Close()
				found := false
				for stream.Next() {
					if stream.Current().SystemFingerprint == "native-fingerprint" {
						found = true
					}
				}
				if stream.Err() != nil || !found {
					t.Errorf("native stream field missing: %v", stream.Err())
				}

			case "anthropic":
				client, err := anthropic.New("test-key", nil, anthropic.WithBaseURL(server.URL), anthropic.WithHTTPClient(server.Client())).SDKClient()
				if err != nil {
					t.Fatal(err)
				}
				result, err := client.Messages.New(t.Context(), asdk.MessageNewParams{Model: "native-model", MaxTokens: 10, Messages: []asdk.MessageParam{asdk.NewUserMessage(asdk.NewTextBlock("hello"))}, Temperature: aparam.NewOpt(0.5)})
				if err != nil {
					t.Fatal(err)
				}
				if len(result.Content) != 2 || result.Content[1].Signature != "native-signature" {
					t.Errorf("native response lost: %+v", result)
				}
				stream := client.Messages.NewStreaming(t.Context(), asdk.MessageNewParams{Model: "native-model", MaxTokens: 10, Messages: []asdk.MessageParam{asdk.NewUserMessage(asdk.NewTextBlock("hello"))}, Temperature: aparam.NewOpt(0.5)})
				defer stream.Close()
				found := false
				for stream.Next() {
					if stream.Current().Delta.Signature == "native-signature" {
						found = true
					}
				}
				if stream.Err() != nil || !found {
					t.Errorf("native signature event missing: %v", stream.Err())
				}

			case "gemini":
				client, err := gemini.New("test-key", nil, gemini.WithBaseURL(server.URL), gemini.WithHTTPClient(server.Client())).SDKClient(t.Context())
				if err != nil {
					t.Fatal(err)
				}
				temperature := float32(0.5)
				result, err := client.Models.GenerateContent(t.Context(), "native-model", genai.Text("hello"), &genai.GenerateContentConfig{Temperature: &temperature})
				if err != nil {
					t.Fatal(err)
				}
				if len(result.Candidates) != 1 || result.Candidates[0].GroundingMetadata.WebSearchQueries[0] != "native-query" {
					t.Errorf("native response lost: %+v", result)
				}
				found := false
				for event, err := range client.Models.GenerateContentStream(t.Context(), "native-model", genai.Text("hello"), &genai.GenerateContentConfig{Temperature: &temperature}) {
					if err != nil {
						t.Fatal(err)
					}
					if len(event.Candidates) > 0 && event.Candidates[0].GroundingMetadata != nil && len(event.Candidates[0].GroundingMetadata.WebSearchQueries) > 0 {
						found = event.Candidates[0].GroundingMetadata.WebSearchQueries[0] == "native-query"
					}
				}
				if !found {
					t.Error("native grounding event missing")
				}

			}
		})
	}
}

// Compile-time proof that adding native access does not expand the core contract.
type streamOnly struct{}

func (streamOnly) GenerateStream(context.Context, ai.AIRequest) <-chan ai.Token {
	c := make(chan ai.Token)
	close(c)
	return c
}

var _ ai.Model = streamOnly{}
