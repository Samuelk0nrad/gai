package mistral

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

func TestRequestObservationsEmitWithoutContentCapture(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		t.Run(map[bool]string{false: "generate", true: "stream"}[streaming], func(t *testing.T) {
			previousProvider := otel.GetTracerProvider()
			providerTracer := sdktrace.NewTracerProvider()
			otel.SetTracerProvider(providerTracer)
			t.Cleanup(func() {
				_ = providerTracer.Shutdown(context.Background())
				otel.SetTracerProvider(previousProvider)
			})

			var events []gai.Observation
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if streaming {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = w.Write([]byte("data: [DONE]\n\n"))
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
			}))
			defer server.Close()

			provider := New("test-key", gai.ObservationSinkFunc(func(_ context.Context, event gai.Observation) {
				events = append(events, event)
			}))
			provider.baseURL = server.URL
			model, err := provider.Model(MistralSmallLatest)
			if err != nil {
				t.Fatal(err)
			}
			if streaming {
				for range model.GenerateStream(t.Context(), ai.AIRequest{Prompt: "secret prompt"}) {
				}
			} else if _, err := any(model).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Prompt: "secret prompt"}); err != nil {
				t.Fatal(err)
			}

			want := "mistral_generate_request"
			if streaming {
				want = "mistral_stream_request"
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
		})
	}
}
