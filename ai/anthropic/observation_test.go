package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestGenerateEmitsRequestObservationWithoutContentCapture(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	providerTracer := sdktrace.NewTracerProvider()
	otel.SetTracerProvider(providerTracer)
	t.Cleanup(func() {
		_ = providerTracer.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	})

	var events []gai.Observation
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"answer"}],"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	provider := New("test-key", gai.ObservationSinkFunc(func(_ context.Context, event gai.Observation) {
		events = append(events, event)
	}))
	provider.baseURL = server.URL
	model, err := provider.Model(ClaudeSonnet4_6)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := any(model).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Prompt: "secret prompt"}); err != nil {
		t.Fatal(err)
	}

	var request, finished *gai.Observation
	for index := range events {
		event := &events[index]
		if event.Name == "anthropic_generate_request" {
			request = event
			if _, ok := event.Fields["prompt"]; ok {
				t.Fatalf("request observation exposed prompt without a capture policy: %#v", event.Fields)
			}
		}
		if event.Name == "generation_finished" {
			finished = event
		}
	}
	if request == nil || finished == nil {
		t.Fatalf("request or generation_finished observation not emitted: %#v", events)
	}
	if request.TraceID != finished.TraceID || request.SpanID != finished.SpanID {
		t.Fatalf("request correlation = (%q, %q), generation_finished = (%q, %q)", request.TraceID, request.SpanID, finished.TraceID, finished.SpanID)
	}
}

func TestGenerateEmitsRequestAndContentToOTelWithoutSink(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	providerTracer := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(providerTracer)
	t.Cleanup(func() {
		_ = providerTracer.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"answer"}],"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	provider := New("test-key", nil)
	provider.baseURL = server.URL
	model, err := provider.Model(ClaudeSonnet4_6)
	if err != nil {
		t.Fatal(err)
	}
	ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{Prompt: gai.CaptureEnabled})
	if _, err := any(model).(ai.ModelGenerator).Generate(ctx, ai.AIRequest{Prompt: "allowed prompt"}); err != nil {
		t.Fatal(err)
	}

	for _, span := range recorder.Ended() {
		for _, event := range span.Events() {
			if event.Name != "observation.anthropic_generate_request" {
				continue
			}
			for _, attribute := range event.Attributes {
				if string(attribute.Key) == "observation.prompt" && attribute.Value.AsString() == "allowed prompt" {
					return
				}
			}
			t.Fatalf("request event omitted policy-enabled prompt: %#v", event.Attributes)
		}
	}
	t.Fatalf("OTel-only generation omitted request observation: %#v", recorder.Ended())
}

func TestGenerateOmitsMixedProviderEnvelopesFromCapturedObservations(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"completion text"},{"type":"thinking","thinking":"reasoning secret"},{"type":"tool_use","id":"toolu_1","name":"search","input":{"query":"tool-input secret"}}],"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	var events []gai.Observation
	provider := New("test-key", gai.ObservationSinkFunc(func(_ context.Context, event gai.Observation) {
		events = append(events, event)
	}))
	provider.baseURL = server.URL
	model, err := provider.Model(ClaudeSonnet4_6)
	if err != nil {
		t.Fatal(err)
	}
	ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{
		Prompt:     gai.CaptureEnabled,
		Completion: gai.CaptureEnabled,
	})
	if _, err := any(model).(ai.ModelGenerator).Generate(ctx, ai.AIRequest{
		Prompt: "prompt secret",
		Messages: []ai.RequestMessage{
			{Role: ai.RequestMessageRoleAssistant, ToolCalls: []ai.RequestToolCall{{ID: "call_1", Name: "search", Arguments: []byte(`{"query":"prior tool-input secret"}`)}}},
			{Role: ai.RequestMessageRoleTool, ToolResult: &ai.RequestToolResult{ToolCallID: "call_1", Name: "search", Content: "prior tool-output secret"}},
		},
	}); err != nil {
		t.Fatal(err)
	}

	for _, event := range events {
		switch event.Name {
		case "anthropic_generate_request":
			if _, ok := event.Fields["payload"]; ok {
				t.Fatalf("request observation captured mixed payload: %#v", event.Fields)
			}
			if event.Fields["prompt"] != "prompt secret" {
				t.Fatalf("request observation prompt = %#v", event.Fields)
			}
		case "anthropic_generate_success":
			if _, ok := event.Fields["response"]; ok {
				t.Fatalf("success observation captured mixed response: %#v", event.Fields)
			}
			if event.Fields["response_text"] != "completion text" {
				t.Fatalf("success observation completion = %#v", event.Fields)
			}
			if _, ok := event.Fields["reasoning"]; ok {
				t.Fatalf("success observation captured disabled reasoning: %#v", event.Fields)
			}
		}
	}
}
