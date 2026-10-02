package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/lace-ai/gai/ai"
)

var _ ai.InputTokenCounter = (*Model)(nil)

func TestCountInputTokensMapsGenerationInputs(t *testing.T) {
	type capturedRequest struct {
		path   string
		method string
		header http.Header
		body   map[string]any
	}
	requests := make(chan capturedRequest, 2)
	model := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		requests <- capturedRequest{r.URL.Path, r.Method, r.Header.Clone(), decodeRequest(t, r)}
		switch r.URL.Path {
		case "/v1/messages/count_tokens":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"input_tokens":42}`))
		case "/v1/messages":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	})
	req := countInputTokensRequest()
	snapshot := req.Copy()
	if count, err := model.CountInputTokens(t.Context(), req); err != nil || count != 42 {
		t.Fatalf("CountInputTokens = %d, %v, want 42, nil", count, err)
	}
	for token := range model.GenerateStream(t.Context(), req) {
		if token.Err != nil {
			t.Fatal(token.Err)
		}
	}
	counted, generated := <-requests, <-requests
	if counted.path != "/v1/messages/count_tokens" || generated.path != "/v1/messages" {
		t.Fatalf("request paths = %q, %q", counted.path, generated.path)
	}
	for _, captured := range []capturedRequest{counted, generated} {
		if captured.method != http.MethodPost || captured.header.Get("x-api-key") != "test-key" || captured.header.Get("anthropic-version") != "2023-06-01" || captured.body["model"] != ClaudeSonnet4_6 {
			t.Fatalf("request metadata = %#v", captured)
		}
	}
	for _, field := range []string{"messages", "system", "tools", "tool_choice", "thinking", "output_config"} {
		if counted.body[field] == nil || !reflect.DeepEqual(counted.body[field], generated.body[field]) {
			t.Errorf("%s differs: count %#v, generation %#v", field, counted.body[field], generated.body[field])
		}
	}
	for _, field := range []string{"max_tokens", "stream", "temperature", "top_k", "top_p", "stop_sequences"} {
		if _, exists := counted.body[field]; exists {
			t.Errorf("count endpoint received generation-only field %q: %#v", field, counted.body[field])
		}
	}
	if generated.body["max_tokens"] != float64(4096) || generated.body["stream"] != true {
		t.Fatalf("generation defaults = %#v, want max_tokens 4096 and stream true", generated.body)
	}
	if !reflect.DeepEqual(req, snapshot) {
		t.Fatal("counting or generation mutated the canonical request")
	}
	messages := array(t, counted.body["messages"])
	if len(messages) != 4 {
		t.Fatalf("counted messages = %#v, want adjacent tool results grouped", messages)
	}
	assistant := array(t, object(t, messages[1])["content"])
	if len(assistant) != 4 || object(t, assistant[0])["signature"] != "signed" || object(t, assistant[3])["data"] != "opaque" {
		t.Fatalf("reasoning blocks lost opaque data: %#v", assistant)
	}
	results := array(t, object(t, messages[2])["content"])
	if len(results) != 2 || object(t, results[0])["tool_use_id"] != "call_1" || object(t, results[1])["tool_use_id"] != "call_2" || object(t, results[1])["is_error"] != true {
		t.Fatalf("grouped tool results = %#v", results)
	}
}

func TestCountInputTokensRequiresReportedNonnegativeCount(t *testing.T) {
	for _, tc := range []struct {
		name    string
		body    string
		want    int
		wantErr bool
	}{
		{name: "positive", body: `{"input_tokens":42}`, want: 42},
		{name: "reported zero", body: `{"input_tokens":0}`},
		{name: "missing", body: `{}`, wantErr: true},
		{name: "negative", body: `{"input_tokens":-1}`, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			model := testModel(t, func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				if r.URL.Path != "/v1/messages/count_tokens" {
					t.Errorf("path = %q", r.URL.Path)
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(tc.body))
			})
			got, err := model.CountInputTokens(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "question")}})
			if got != tc.want || (err != nil) != tc.wantErr || calls.Load() != 1 {
				t.Fatalf("CountInputTokens = %d, %v, calls %d; want %d, error %t, one call", got, err, calls.Load(), tc.want, tc.wantErr)
			}
		})
	}
}

func TestCountInputTokensPreservesProviderErrors(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		code   string
		kind   ai.ProviderErrorKind
	}{
		{"invalid request", http.StatusBadRequest, "invalid_request_error", ai.ProviderErrorInvalidRequest},
		{"authentication", http.StatusUnauthorized, "authentication_error", ai.ProviderErrorAuthentication},
		{"missing resource", http.StatusNotFound, "not_found_error", ai.ProviderErrorInvalidRequest},
		{"rate limit", http.StatusTooManyRequests, "rate_limit_error", ai.ProviderErrorRateLimited},
		{"server failure", http.StatusInternalServerError, "api_error", ai.ProviderErrorTransient},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			model := testModel(t, func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("request-id", "req_count")
				w.Header().Set("Retry-After", "3")
				w.WriteHeader(tc.status)
				_, _ = fmt.Fprintf(w, `{"type":"error","error":{"type":%q,"message":"count rejected"}}`, tc.code)
			})
			got, err := model.CountInputTokens(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "question")}})
			var providerErr *ai.ProviderError
			var sdkErr *sdk.Error
			if got != 0 || !errors.As(err, &providerErr) || providerErr.Kind != tc.kind || providerErr.StatusCode != tc.status || providerErr.Code != tc.code || providerErr.RequestID != "req_count" || providerErr.RetryAfter != 3*time.Second || !errors.As(err, &sdkErr) || calls.Load() != 1 {
				t.Fatalf("CountInputTokens = %d, %#v, provider %#v, SDK %#v, calls %d", got, err, providerErr, sdkErr, calls.Load())
			}
			if errors.Is(err, ai.ErrInputTokenCountUnsupported) {
				t.Fatalf("provider error incorrectly became unsupported: %v", err)
			}
		})
	}
}

func TestCountInputTokensCanceledBeforeTransport(t *testing.T) {
	var calls atomic.Int32
	model := testModel(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	})
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	got, err := model.CountInputTokens(ctx, ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "question")}})
	if got != 0 || !errors.Is(err, context.Canceled) || calls.Load() != 0 {
		t.Fatalf("CountInputTokens = %d, %v, calls %d, want canceled before transport", got, err, calls.Load())
	}
}

func TestCountInputTokensCancelsInFlightRequest(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	model := testModel(t, func(_ http.ResponseWriter, r *http.Request) {
		close(started)
		select {
		case <-r.Context().Done():
		case <-release:
		}
	})
	t.Cleanup(func() { close(release) })
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	type result struct {
		count int
		err   error
	}
	finished := make(chan result, 1)
	go func() {
		count, err := model.CountInputTokens(ctx, ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "question")}})
		finished <- result{count, err}
	}()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("count request did not reach the provider")
	}
	cancel()
	select {
	case got := <-finished:
		if got.count != 0 || !errors.Is(got.err, context.Canceled) {
			t.Fatalf("CountInputTokens = %d, %v, want cancellation", got.count, got.err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("count request ignored cancellation")
	}
}

func TestCountInputTokensRejectsInvalidCanonicalInputBeforeTransport(t *testing.T) {
	for _, tc := range []struct {
		name string
		req  ai.AIRequest
	}{
		{name: "missing messages"},
		{name: "unmatched tool result", req: ai.AIRequest{Messages: []ai.Message{{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "missing", Name: "lookup", Parts: ai.TextParts("result")}}}}}}},
		{name: "non-leading system", req: ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "question"), ai.TextMessage(ai.RoleSystem, "late")}}},
		{name: "unsigned reasoning", req: ai.AIRequest{Messages: []ai.Message{{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentReasoning, Text: "unsigned"}}}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			model := testModel(t, func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				w.WriteHeader(http.StatusInternalServerError)
			})
			if got, err := model.CountInputTokens(t.Context(), tc.req); got != 0 || err == nil || calls.Load() != 0 {
				t.Fatalf("CountInputTokens = %d, %v, calls %d, want validation failure before transport", got, err, calls.Load())
			}
		})
	}
}

func TestCountInputTokensRejectsURLMediaBeforeTransport(t *testing.T) {
	var calls atomic.Int32
	model := testModel(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	})
	req := ai.AIRequest{Messages: []ai.Message{{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", URI: "https://example.test/image.png"}}}}}}
	if err := req.Validate(); err != nil {
		t.Fatalf("fixture must be a valid canonical request: %v", err)
	}
	got, err := model.CountInputTokens(t.Context(), req)
	if got != 0 || !errors.Is(err, ai.ErrInputTokenCountUnsupported) || calls.Load() != 0 {
		t.Fatalf("CountInputTokens = %d, %v, calls %d, want unsupported before transport", got, err, calls.Load())
	}
}

func TestCountInputTokensRejectsNativeHooksWithoutExecutingThem(t *testing.T) {
	var requests, hooks atomic.Int32
	model := testModel(t, func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	})
	configured, err := model.client.TypedModel(model.Name(), WithMessageParams(func(*sdk.MessageNewParams) error {
		hooks.Add(1)
		return nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	got, err := configured.CountInputTokens(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "question")}})
	if got != 0 || !errors.Is(err, ai.ErrInputTokenCountUnsupported) || requests.Load() != 0 || hooks.Load() != 0 {
		t.Fatalf("CountInputTokens = %d, %v, requests %d, hooks %d, want unsupported without executing hooks", got, err, requests.Load(), hooks.Load())
	}
}

func countInputTokensRequest() ai.AIRequest {
	return ai.AIRequest{
		Messages: []ai.Message{
			{Role: ai.RoleSystem, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "rules"}, {Kind: ai.ContentJSON, JSON: json.RawMessage(`{"ordered":true}`)}}},
			ai.TextMessage(ai.RoleUser, "look up both values"),
			{Role: ai.RoleAssistant, Parts: []ai.ContentPart{
				{Kind: ai.ContentReasoning, Text: "reason", Extensions: []ai.Extension{{Namespace: "anthropic", Type: "thinking_signature", Data: json.RawMessage(`"signed"`), Required: true}}},
				{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "call_1", Type: "function", Name: "lookup", Args: json.RawMessage(`{"value":1,"enabled":true}`)}},
				{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "call_2", Type: "function", Name: "lookup", Args: json.RawMessage(`{"value":2,"enabled":false}`)}},
				{Kind: ai.ContentExtension, Extensions: []ai.Extension{{Namespace: "anthropic", Type: "redacted_thinking", Data: json.RawMessage(`"opaque"`), Required: true}}},
			}},
			{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "call_1", Name: "lookup", Parts: ai.TextParts("first")}}}},
			{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "call_2", Name: "lookup", Parts: []ai.ContentPart{{Kind: ai.ContentJSON, JSON: json.RawMessage(`{"error":true}`)}}, IsError: true}}}},
			ai.TextMessage(ai.RoleUser, "answer now"),
		},
		Tools:          []ai.ToolDefinition{{Type: "function", Name: "lookup", Description: "Lookup a value", Parameters: json.RawMessage(`{"type":"object","properties":{"value":{"type":"integer"},"enabled":{"type":"boolean"}},"required":["value","enabled"]}`)}},
		ToolChoice:     ai.ToolChoice{Mode: ai.ToolChoiceAuto},
		ResponseFormat: ai.ResponseFormat{Type: ai.ResponseFormatJSONSchema, Name: "answer", Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`)},
		Reasoning:      ai.ReasoningConfig{Enabled: true, BudgetTokens: 1024, Effort: ai.ReasoningEffortHigh},
	}
}
