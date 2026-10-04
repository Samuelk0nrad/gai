package history

import (
	"context"
	"testing"

	"github.com/lace-ai/gai"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestHistoryObserverBuildFinishedAcceptsNilPart(t *testing.T) {
	t.Parallel()

	observer := &historyObserver{}
	observer.BuildFinished(context.Background(), nil, 0, 0, 0, 0)

	if observer.contentCount != 0 {
		t.Fatalf("content count = %d, want 0", observer.contentCount)
	}
}

func TestHistoryBuildObserverPreservesTelemetryContract(t *testing.T) {
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
	ctx, observer := newHistoryBuildObserver(t.Context(), sink, "session", 100)
	observer.SetTokenCounterID("test-counter")
	observer.BuildFinished(ctx, nil, 80, 3, 2, 5)
	observer.Finish(nil)

	if len(events) != 1 {
		t.Fatalf("events = %d, want 1", len(events))
	}
	event := events[0]
	if event.Name != "history_source_build_finished" || event.Source != "context:HistorySource" {
		t.Fatalf("event name/source = %q/%q", event.Name, event.Source)
	}
	for key, want := range map[string]any{
		"session_id": "session", "counter_id": "test-counter", "token_budget": 100,
		"total_tokens": 80, "turn_count": 3, "message_count": 5, "content_count": 0,
	} {
		if got := event.Fields[key]; got != want {
			t.Errorf("event field %q = %#v, want %#v", key, got, want)
		}
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
	if span.Name() != "context.history.build" || span.Status().Code != codes.Unset {
		t.Fatalf("span name/status = %q/%s, want context.history.build/Unset", span.Name(), span.Status().Code)
	}
	attrs := map[string]string{}
	for _, attr := range span.Attributes() {
		attrs[string(attr.Key)] = attr.Value.String()
	}
	for key, want := range map[string]string{
		"context.operation": "build", "context.source": "history", "context.session_id": "session",
		"context.token_budget": "100", "context.counter_id": "test-counter",
		"context.history.total_tokens": "80", "context.history.turn_count": "3",
		"context.history.included_turn_count": "2", "context.history.message_count": "5",
	} {
		if got := attrs[key]; got != want {
			t.Errorf("span attribute %q = %q, want %q", key, got, want)
		}
	}
}

func TestSummaryObservationsUseBuildLocalCounts(t *testing.T) {
	t.Parallel()
	var events []gai.Observation
	sink := gai.ObservationSinkFunc(func(_ context.Context, event gai.Observation) { events = append(events, event) })
	ctx, observer := newHistoryBuildObserver(t.Context(), sink, "session", 100)
	defer observer.Finish(nil)
	observer.SetTokenCounterID("counter")
	summary := &Summary{ID: "summary"}
	observer.SummaryIncluded(ctx, summary, 7)
	observer.SummaryGenerated(ctx, summary, 11, 2, 1, false)
	if len(events) != 2 || events[0].Fields["summary_tokens"] != 7 || events[1].Fields["summary_tokens"] != 11 || observer.summaryTokens != 11 {
		t.Fatalf("summary counts lost: %#v", events)
	}
}
