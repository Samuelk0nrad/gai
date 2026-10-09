package ollama

import (
	"context"
	"encoding/json"

	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
)

func TestGenerateStreamUsesOfficialSDKClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/prefix/api/chat" {
			t.Errorf("path = %q, want /prefix/api/chat", r.URL.Path)
		}
		if got := r.Header.Get("User-Agent"); !strings.HasPrefix(got, "ollama/") {
			t.Errorf("User-Agent = %q, want official Ollama SDK user agent", got)
		}
		fmt.Fprintln(w, `{"model":"test","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}`)
	}))
	defer server.Close()

	model, err := New(nil, WithBaseURL(server.URL+"/prefix")).TypedModel("test")
	if err != nil {
		t.Fatal(err)
	}
	events := collectEvents(t, model.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}}))
	if len(events) != 2 || events[0].Text() != "ok" || events[1].Completion == nil {
		t.Fatalf("events = %#v", events)
	}
}

func TestGenerateStreamPreservesCanonicalToolSchemaThroughSDK(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"unit":{"type":"string","default":"celsius"},"location":{"$ref":"#/$defs/location"}},"$defs":{"location":{"type":"string"}},"additionalProperties":false}`)
	var wantSchema any
	if err := json.Unmarshal(schema, &wantSchema); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Tools []struct {
				Function struct {
					Parameters json.RawMessage `json:"parameters"`
				} `json:"function"`
			} `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if len(payload.Tools) != 1 {
			t.Errorf("tools = %#v", payload.Tools)
			return
		}
		var gotSchema any
		if err := json.Unmarshal(payload.Tools[0].Function.Parameters, &gotSchema); err != nil {
			t.Errorf("decode schema: %v", err)
			return
		}
		if !reflect.DeepEqual(gotSchema, wantSchema) {
			t.Errorf("schema = %s, want %s", payload.Tools[0].Function.Parameters, schema)
		}
		fmt.Fprintln(w, `{"model":"test","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}`)
	}))
	defer server.Close()

	tool, err := ai.NewToolDefinition("weather", "Get weather", schema)
	if err != nil {
		t.Fatal(err)
	}
	model, err := New(nil, WithBaseURL(server.URL)).TypedModel("test", WithToolSupport(ai.FeatureSupportSupported))
	if err != nil {
		t.Fatal(err)
	}
	events := collectEvents(t, model.GenerateStream(context.Background(), ai.AIRequest{
		Messages:   []ai.Message{ai.TextMessage(ai.RoleUser, "weather")},
		Tools:      []ai.ToolDefinition{tool},
		ToolChoice: ai.ToolChoice{Mode: ai.ToolChoiceAuto},
	}))
	if len(events) != 2 || events[1].Completion == nil {
		t.Fatalf("events = %#v", events)
	}
}

func TestGenerateStreamEmitsIncrementalTextToolCallsAndCompletion(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" || r.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("request = %s %s auth=%q", r.Method, r.URL.Path, r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Header().Set("X-Request-Id", "req-success")
		fmt.Fprintln(w, `{"model":"qwen3:8b","message":{"role":"assistant","content":"Hel"},"done":false}`)
		fmt.Fprintln(w, `{"model":"qwen3:8b","message":{"role":"assistant","content":"lo","tool_calls":[{"function":{"name":"weather","arguments":{"city":"Tokyo"}}},{"id":"provided","function":{"name":"time","arguments":{}}}]},"done":false}`)
		fmt.Fprint(w, `{"model":"qwen3:8b","message":{"role":"assistant","content":"!"},"done":true,"done_reason":"stop","prompt_eval_count":12,"eval_count":4}`)
	}))
	defer server.Close()

	model, err := New(nil, WithBaseURL(server.URL), WithBearerToken("secret")).TypedModel("qwen3:8b")
	if err != nil {
		t.Fatal(err)
	}
	events := collectEvents(t, model.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}))
	if len(events) != 6 {
		t.Fatalf("events = %#v", events)
	}
	if events[0].Text() != "Hel" || events[1].Text() != "lo" || events[2].Text() != "!" {
		t.Fatalf("text events = %#v", events)
	}
	first := events[3].ToolCall()
	second := events[4].ToolCall()
	if first == nil || !strings.HasPrefix(first.ID, "call_weather_") || first.Name != "weather" || string(first.Args) != `{"city":"Tokyo"}` {
		t.Fatalf("generated call = %#v", first)
	}
	if second == nil || second.ID != "provided" || second.Name != "time" || string(second.Args) != `{}` {
		t.Fatalf("provided call = %#v", second)
	}
	completion := events[5].Completion
	if completion == nil || completion.Provider != "ollama" || completion.RequestID != "req-success" || completion.Model != "qwen3:8b" || completion.FinishReason != "stop" || !completion.UsageReported || completion.Usage.InputTokens != 12 || completion.Usage.OutputTokens != 4 {
		t.Fatalf("completion = %#v", completion)
	}
	for i, event := range events {
		if err := event.Validate(); err != nil {
			t.Fatalf("event %d invalid: %v", i, err)
		}
	}
}

func TestGenerateStreamRejectsMalformedAndTruncatedStreams(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"malformed", "not-json\n"},
		{"truncated", `{"message":{"role":"assistant","content":"partial"},"done":false}` + "\n"},
		{"in-band error", `{"error":"model unavailable"}` + "\n"},
		{"invalid tool arguments", `{"message":{"role":"assistant","tool_calls":[{"function":{"name":"x","arguments":[]}}]},"done":true}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := fixtureModel(t, http.StatusOK, tt.body, http.Header{"X-Request-Id": []string{"req-stream-error"}})
			events := collectEvents(t, model.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}}))
			if len(events) == 0 || events[len(events)-1].Err == nil {
				t.Fatalf("events = %#v", events)
			}
			if events[len(events)-1].Completion != nil {
				t.Fatal("malformed stream emitted completion")
			}
			if tt.name == "in-band error" {
				var providerErr *ai.ProviderError
				if !errors.As(events[len(events)-1].Err, &providerErr) || providerErr.RequestID != "req-stream-error" {
					t.Fatalf("in-band provider error = %#v", providerErr)
				}
			}
		})
	}
}

func TestGenerateStreamPreservesTransportErrorsWithoutHTTPResponse(t *testing.T) {
	transportErr := errors.New("dial failed")
	client := &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		return nil, transportErr
	})}
	model, err := New(nil, WithHTTPClient(client)).TypedModel("test")
	if err != nil {
		t.Fatal(err)
	}

	events := collectEvents(t, model.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}}))
	if len(events) != 1 || !errors.Is(events[0].Err, transportErr) {
		t.Fatalf("events = %#v", events)
	}
}

func TestGenerateStreamDoesNotExposeMalformedResponseContent(t *testing.T) {
	const secret = "private model output"
	model := fixtureModel(t, http.StatusOK, `{"message":{"content":"`+secret, nil)
	events := collectEvents(t, model.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}}))
	if len(events) != 1 || events[0].Err == nil {
		t.Fatalf("events = %#v", events)
	}
	if strings.Contains(events[0].Err.Error(), secret) {
		t.Fatalf("error exposed response content: %v", events[0].Err)
	}
}

func TestGenerateStreamClassifiesHTTPErrorsAndCancellation(t *testing.T) {
	model := fixtureModel(t, http.StatusUnauthorized, `{"error":"unauthorized"}`, http.Header{"X-Request-Id": []string{"req-1"}})
	events := collectEvents(t, model.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}}))
	var providerErr *ai.ProviderError
	if len(events) != 1 || !errors.As(events[0].Err, &providerErr) || providerErr.Kind != ai.ProviderErrorAuthentication || providerErr.StatusCode != http.StatusUnauthorized || providerErr.RequestID != "req-1" {
		t.Fatalf("events = %#v, provider error = %#v", events, providerErr)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		<-r.Context().Done()
	}))
	defer server.Close()
	cancelModel, err := New(nil, WithBaseURL(server.URL)).TypedModel("test")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	canceled := collectEvents(t, cancelModel.GenerateStream(ctx, ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}}))
	if len(canceled) > 1 || len(canceled) == 1 && !errors.Is(canceled[0].Err, context.DeadlineExceeded) {
		t.Fatalf("canceled events = %#v", canceled)
	}
}

func TestGenerateStreamCompletesBeforeServerEOFAndReportsZeroUsage(t *testing.T) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprintln(w, `{"model":"test","message":{"role":"assistant","content":"done"},"done":true,"done_reason":"stop","prompt_eval_count":0,"eval_count":0}`)
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer func() {
		close(release)
		server.Close()
	}()
	model, err := New(nil, WithBaseURL(server.URL)).TypedModel("test")
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan []ai.Token, 1)
	go func() {
		done <- collectEvents(t, model.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}}))
	}()
	select {
	case events := <-done:
		if len(events) != 2 || events[1].Completion == nil || !events[1].Completion.UsageReported {
			t.Fatalf("events = %#v", events)
		}
	case <-time.After(time.Second):
		t.Fatal("stream waited for connection EOF after done record")
	}
}

func TestGenerateStreamSharesModelAcrossConcurrentRuns(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"model":"test","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}`)
	}))
	defer server.Close()
	model, err := New(nil, WithBaseURL(server.URL)).TypedModel("test")
	if err != nil {
		t.Fatal(err)
	}
	var wait sync.WaitGroup
	errs := make(chan error, 24)
	for range 24 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			events := collectEvents(t, model.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}}))
			if len(events) != 2 || events[0].Text() != "ok" || events[1].Completion == nil {
				errs <- fmt.Errorf("events = %#v", events)
			}
		}()
	}
	wait.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestGenerateStreamDeliversTextBeforeTerminalRecord(t *testing.T) {
	firstFlushed := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseServer := func() { releaseOnce.Do(func() { close(release) }) }
	defer releaseServer()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"model":"test","message":{"role":"assistant","content":"first"},"done":false}`)
		w.(http.Flusher).Flush()
		close(firstFlushed)
		<-release
		fmt.Fprintln(w, `{"model":"test","message":{"role":"assistant","content":"second"},"done":true,"done_reason":"stop"}`)
	}))
	defer server.Close()
	model, err := New(nil, WithBaseURL(server.URL)).TypedModel("test")
	if err != nil {
		t.Fatal(err)
	}
	stream := model.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}})
	<-firstFlushed
	select {
	case event := <-stream:
		if event.Text() != "first" {
			t.Fatalf("first event = %#v", event)
		}
	case <-time.After(time.Second):
		t.Fatal("first text was not delivered incrementally")
	}
	releaseServer()
	events := collectEvents(t, stream)
	if len(events) != 2 || events[0].Text() != "second" || events[1].Completion == nil {
		t.Fatalf("remaining events = %#v", events)
	}
}

func TestGenerateStreamCancellationUnblocksUnreadTokenSend(t *testing.T) {
	bodyClosed := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(bodyClosed)
		for _, text := range []string{"one", "two", "three"} {
			fmt.Fprintf(w, `{"model":"test","message":{"role":"assistant","content":%q},"done":false}`+"\n", text)
			w.(http.Flusher).Flush()
		}
		<-r.Context().Done()
	}))
	defer server.Close()
	model, err := New(nil, WithBaseURL(server.URL)).TypedModel("test")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	stream := model.GenerateStream(ctx, ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}})
	// The channel has capacity one. Leaving it unread blocks the next send.
	time.Sleep(20 * time.Millisecond)
	cancel()
	for range stream {
	}
	select {
	case <-bodyClosed:
	case <-time.After(time.Second):
		t.Fatal("cancellation did not close the response body")
	}
}

func TestGenerateStreamRejectsCrossOriginRedirectsBeforeSendingSecrets(t *testing.T) {
	var targetCalls atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		targetCalls.Add(1)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer target.Close()
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+"/stolen", http.StatusTemporaryRedirect)
	}))
	defer origin.Close()
	model, err := New(nil, WithBaseURL(origin.URL), WithBearerToken("secret")).TypedModel("test")
	if err != nil {
		t.Fatal(err)
	}
	events := collectEvents(t, model.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "private prompt")}}))
	if len(events) != 1 || events[0].Err == nil {
		t.Fatalf("events = %#v", events)
	}
	if targetCalls.Load() != 0 {
		t.Fatalf("redirect target received %d requests", targetCalls.Load())
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (fn roundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

func fixtureModel(t *testing.T, status int, body string, headers http.Header) *Model {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for name, values := range headers {
			for _, value := range values {
				w.Header().Add(name, value)
			}
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	model, err := New(nil, WithBaseURL(server.URL)).TypedModel("test")
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func collectEvents(t *testing.T, stream <-chan ai.Token) []ai.Token {
	t.Helper()
	var events []ai.Token
	for event := range stream {
		events = append(events, event)
	}
	return events
}
