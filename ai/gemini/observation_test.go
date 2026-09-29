package gemini

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

func TestGenerateEmitsRequestObservationWithoutContentCapture(t *testing.T) {
	assertGeminiRequestObservationWithoutContentCapture(t, false)
}

func TestGenerateStreamEmitsRequestObservationWithoutContentCapture(t *testing.T) {
	assertGeminiRequestObservationWithoutContentCapture(t, true)
}

func assertGeminiRequestObservationWithoutContentCapture(t *testing.T, streaming bool) {
	t.Helper()
	previousProvider := otel.GetTracerProvider()
	providerTracer := sdktrace.NewTracerProvider()
	otel.SetTracerProvider(providerTracer)
	t.Cleanup(func() {
		_ = providerTracer.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	})

	var events []gai.Observation
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":{"code":429,"message":"generation failed"}}`, http.StatusTooManyRequests)
	}))
	defer server.Close()

	provider := New("test-key", gai.ObservationSinkFunc(func(_ context.Context, event gai.Observation) {
		events = append(events, event)
	}))
	provider.baseURL = server.URL
	provider.httpClient = server.Client()
	model, err := provider.Model("gemini-test")
	if err != nil {
		t.Fatal(err)
	}
	if streaming {
		for range model.GenerateStream(t.Context(), ai.AIRequest{Prompt: "secret prompt"}) {
		}
	} else if _, err := any(model).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Prompt: "secret prompt"}); err == nil {
		t.Fatal("Generate error = nil, want API error")
	}

	want := "gemini_generate_request"
	if streaming {
		want = "gemini_stream_request"
	}
	var request, finished *gai.Observation
	for index := range events {
		event := &events[index]
		if event.Name == want {
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
		t.Fatalf("%s or generation_finished observation not emitted: %#v", want, events)
	}
	if request.TraceID != finished.TraceID || request.SpanID != finished.SpanID {
		t.Fatalf("request correlation = (%q, %q), generation_finished = (%q, %q)", request.TraceID, request.SpanID, finished.TraceID, finished.SpanID)
	}
}
