package observe_test

import (
	"context"
	"errors"
	"testing"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/internal/observe"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestOperationPreservesSpanAndObservationSemantics(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	})

	var events []gai.Observation
	sink := gai.ObservationSinkFunc(func(_ context.Context, event gai.Observation) {
		events = append(events, event)
	})
	ctx, operation := observe.Start(t.Context(), sink, "test-tracer", "test", "test.operation", "run", "test:observer", attribute.String("test.initial", "value"))
	operation.Set(attribute.Int("test.count", 2))
	emitErr := errors.New("emit failed")
	operation.Emit(ctx, "completed", map[string]any{"result": "ok"}, emitErr)
	operation.Finish(errors.New("failed"))

	if len(events) != 1 {
		t.Fatalf("events = %d, want 1", len(events))
	}
	event := events[0]
	if event.Name != "completed" || event.Source != "test:observer" || event.Fields["result"] != "ok" || event.Err != nil || event.Fields["outcome"] != "error" || event.Fields["error_type"] != "*errors.errorString" {
		t.Fatalf("event = %#v", event)
	}
	traceID, spanID, err := gai.SpanContextIDs(ctx)
	if err != nil {
		t.Fatalf("SpanContextIDs() error = %v", err)
	}
	if event.TraceID != traceID || event.SpanID != spanID {
		t.Fatalf("event correlation = trace_id:%q span_id:%q, want trace_id:%q span_id:%q", event.TraceID, event.SpanID, traceID, spanID)
	}

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("spans = %d, want 1", len(spans))
	}
	span := spans[0]
	if span.Name() != "test.run" || span.Status().Code.String() != "Error" {
		t.Fatalf("span = %q status=%s", span.Name(), span.Status().Code)
	}
	attrs := map[string]string{}
	for _, attr := range span.Attributes() {
		attrs[string(attr.Key)] = attr.Value.String()
	}
	if attrs["test.operation"] != "run" || attrs["test.initial"] != "value" || attrs["test.count"] != "2" {
		t.Fatalf("attrs = %#v", attrs)
	}
}

func TestOperationRecordsObservationWithNilSink(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	})

	ctx, operation := observe.Start(t.Context(), nil, "test-tracer", "test", "test.operation", "run", "test:observer")
	operation.Emit(ctx, "completed", map[string]any{"result": "ok"}, nil)
	operation.Finish(nil)

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("spans = %d, want 1", len(spans))
	}
	events := spans[0].Events()
	if len(events) != 1 {
		t.Fatalf("events = %d, want 1", len(events))
	}
	if events[0].Name != "observation.completed" {
		t.Fatalf("event name = %q, want observation.completed", events[0].Name)
	}
}
