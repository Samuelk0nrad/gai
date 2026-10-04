package exa_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop/tools/exa"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestSearchTool(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if got := r.Header.Get("x-api-key"); got != "secret" {
			t.Errorf("x-api-key = %q, want secret", got)
		}

		var body struct {
			Query      string `json:"query"`
			Type       string `json:"type"`
			NumResults int    `json:"numResults"`
			Contents   struct {
				Highlights bool `json:"highlights"`
			} `json:"contents"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if body.Query != "current Go release" || body.Type != "fast" || body.NumResults != 3 || !body.Contents.Highlights {
			t.Errorf("unexpected request: %#v", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"requestId":"request-1","results":[{"title":"Go","url":"https://go.dev","highlights":["Current release"]}]}`))
	}))
	defer server.Close()

	sink := &captureObservationSink{}
	tool, err := exa.NewSearchTool("secret", exa.WithEndpoint(server.URL), exa.WithSearchType("fast"), exa.WithNumResults(3), exa.WithObservationSink(sink))
	if err != nil {
		t.Fatalf("NewSearchTool: %v", err)
	}
	if tool.Name() != "web_search" {
		t.Fatalf("Name = %q, want web_search", tool.Name())
	}
	if tool.Description() == "" {
		t.Fatal("expected default prompt metadata")
	}
	if _, err := tool.Params().JSONSchema(); err != nil {
		t.Fatalf("default params invalid: %v", err)
	}

	response, callErr := tool.Function(context.Background(), ai.ToolCall{
		ID: "call-1", Type: "function", Name: tool.Name(), Args: json.RawMessage(`{"query":" current Go release "}`),
	})
	if err := callErr; err != nil {
		t.Fatalf("Function: %v", err)
	}
	if !json.Valid([]byte(response)) {
		t.Fatalf("tool response is not JSON: %s", response)
	}
	events := sink.Events()
	if len(events) != 2 || events[0].Name != "exa_search_started" || events[1].Name != "exa_search_finished" {
		t.Fatalf("unexpected debug events: %#v", events)
	}
	if _, exists := events[0].Fields["query"]; exists {
		t.Fatal("query leaked through a non-sensitive debug sink")
	}
	if got := events[1].Fields["result_count"]; got != 1 {
		t.Fatalf("result_count = %#v, want 1", got)
	}
}

func TestSearchToolAllowsCustomPromptMetadata(t *testing.T) {
	t.Parallel()

	tool, err := exa.NewSearchTool("secret",
		exa.WithPromptDescription("Use this only for live web lookups."),
		exa.WithPromptParams(ai.ToolParameters{
			Strict: true,
			Properties: []ai.ToolParameter{
				{Name: "query", Type: ai.ToolParameterString, Required: true},
				{Name: "reason", Type: ai.ToolParameterString, Required: true},
			},
		}),
	)
	if err != nil {
		t.Fatalf("NewSearchTool: %v", err)
	}
	if got := tool.Description(); got != "Use this only for live web lookups." {
		t.Fatalf("Description = %q", got)
	}
	params, err := tool.Params().JSONSchema()
	if err != nil {
		t.Fatalf("Params JSONSchema: %v", err)
	}
	if got := string(params); got != `{"type":"object","required":["query","reason"],"properties":{"query":{"type":"string"},"reason":{"type":"string"}},"additionalProperties":false}` {
		t.Fatalf("Params = %q", got)
	}
}

func TestSearchToolReturnsAPIError(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("x-request-id", "request-429")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"rate limited"}`))
	}))
	defer server.Close()

	sink := &captureObservationSink{}
	tool, err := exa.NewSearchTool("secret", exa.WithEndpoint(server.URL), exa.WithObservationSink(sink))
	if err != nil {
		t.Fatalf("NewSearchTool: %v", err)
	}
	_, callErr := tool.Function(context.Background(), ai.ToolCall{
		ID: "call-1", Type: "function", Name: tool.Name(), Args: json.RawMessage(`{"query":"news"}`),
	})
	if callErr == nil {
		t.Fatal("expected API error")
	}
	if !errors.Is(callErr, exa.ErrAPIRequest) {
		t.Fatalf("error = %v, want ErrAPIRequest", callErr)
	}
	var apiErr *exa.APIError
	if !errors.As(callErr, &apiErr) {
		t.Fatalf("error = %T, want *exa.APIError", callErr)
	}
	if apiErr.StatusCode != http.StatusTooManyRequests || apiErr.RequestID != "request-429" || apiErr.Message != "rate limited" {
		t.Fatalf("unexpected API error: %#v", apiErr)
	}
	events := sink.Events()
	if len(events) != 2 || events[1].Name != "exa_search_failed" || events[1].Err != nil || events[1].Fields["outcome"] != "error" {
		t.Fatalf("unexpected failure events: %#v", events)
	}
	if got := events[1].Fields["stage"]; got != "api_response" {
		t.Fatalf("failure stage = %#v, want api_response", got)
	}
}

func TestSearchToolPolicyCaptureIncludesQuery(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"requestId":"request-1","results":[]}`))
	}))
	defer server.Close()

	sink := &captureObservationSink{}
	tool, err := exa.NewSearchTool("secret", exa.WithEndpoint(server.URL), exa.WithObservationSink(sink))
	if err != nil {
		t.Fatalf("NewSearchTool: %v", err)
	}
	ctx := gai.WithContentCapturePolicy(context.Background(), gai.ContentCapturePolicy{ToolInput: gai.CaptureEnabled})
	_, callErr := tool.Function(ctx, ai.ToolCall{
		ID: "call-1", Type: "function", Name: tool.Name(), Args: json.RawMessage(`{"query":"private query"}`),
	})
	if err := callErr; err != nil {
		t.Fatalf("Function: %v", err)
	}
	if got := sink.Events()[0].Fields["query"]; got != "private query" {
		t.Fatalf("query = %#v, want private query", got)
	}
}

func TestSearchToolTracing(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	previousProvider := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		otel.SetTracerProvider(previousProvider)
		_ = provider.Shutdown(context.Background())
	})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"rate limited","requestId":"request-trace"}`))
	}))
	defer server.Close()

	tool, err := exa.NewSearchTool("secret", exa.WithEndpoint(server.URL))
	if err != nil {
		t.Fatalf("NewSearchTool: %v", err)
	}
	_, callErr := tool.Function(context.Background(), ai.ToolCall{
		ID: "call-1", Type: "function", Name: tool.Name(), Args: json.RawMessage(`{"query":"news"}`),
	})
	if callErr == nil {
		t.Fatal("expected API error")
	}

	var searchSpan sdktrace.ReadOnlySpan
	for _, span := range recorder.Ended() {
		if span.Name() == "tool.exa.search" {
			searchSpan = span
			break
		}
	}
	if searchSpan == nil {
		t.Fatal("tool.exa.search span was not recorded")
	}
	if searchSpan.Status().Code != codes.Error {
		t.Fatalf("span status = %v, want error", searchSpan.Status().Code)
	}
	if strings.Contains(searchSpan.Status().Description, "rate limited") {
		t.Fatalf("span status leaked provider response content: %q", searchSpan.Status().Description)
	}
	for _, event := range searchSpan.Events() {
		for _, attr := range event.Attributes {
			if strings.Contains(attr.Value.String(), "rate limited") {
				t.Fatalf("span event leaked provider response content: %#v", event)
			}
		}
	}
	attrs := attributeMap(searchSpan.Attributes())
	if attrs["tool.name"].AsString() != "web_search" || attrs["http.response.status_code"].AsInt64() != http.StatusTooManyRequests {
		t.Fatalf("unexpected span attributes: %#v", attrs)
	}
	if attrs["exa.request_id"].AsString() != "request-trace" {
		t.Fatalf("request ID attribute = %q, want request-trace", attrs["exa.request_id"].AsString())
	}
}

func TestNewSearchToolValidatesConfiguration(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		apiKey  string
		options []exa.Option
		wantErr error
	}{
		{name: "missing API key", wantErr: exa.ErrAPIKeyMissing},
		{name: "unsupported type", apiKey: "secret", options: []exa.Option{exa.WithSearchType("neural")}, wantErr: exa.ErrInvalidOption},
		{name: "too many results", apiKey: "secret", options: []exa.Option{exa.WithNumResults(101)}, wantErr: exa.ErrInvalidOption},
		{name: "empty prompt description", apiKey: "secret", options: []exa.Option{exa.WithPromptDescription(" ")}, wantErr: exa.ErrInvalidOption},
		{name: "invalid prompt params", apiKey: "secret", options: []exa.Option{exa.WithPromptParams(ai.ToolParameters{
			Properties: []ai.ToolParameter{{Name: "value", Type: ai.ToolParameterType("unsupported")}},
		})}, wantErr: exa.ErrInvalidOption},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			_, err := exa.NewSearchTool(test.apiKey, test.options...)
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("error = %v, want %v", err, test.wantErr)
			}
		})
	}
}

type captureObservationSink struct {
	mu     sync.Mutex
	events []gai.Observation
}

func (s *captureObservationSink) Emit(_ context.Context, event gai.Observation) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, event)
}

func (s *captureObservationSink) Events() []gai.Observation {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]gai.Observation(nil), s.events...)
}

func attributeMap(attrs []attribute.KeyValue) map[string]attribute.Value {
	result := make(map[string]attribute.Value, len(attrs))
	for _, attr := range attrs {
		result[string(attr.Key)] = attr.Value
	}
	return result
}
