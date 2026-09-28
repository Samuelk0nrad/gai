package ai

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/lace-ai/gai"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestToolCallStreamObservationConsolidatesCompletedOutcome(t *testing.T) {
	recorder, restore := installToolCallStreamSpanRecorder(t)
	defer restore()

	var observations []gai.Observation
	sink := gai.ObservationSinkFunc(func(_ context.Context, observation gai.Observation) {
		observations = append(observations, observation)
	})
	in := make(chan Token, 2)
	in <- Token{Type: TokenTypeText, Data: []byte(`{"type":"function","name":"echo","arguments":{"value":"secret"}}`)}
	in <- Token{Type: TokenTypeText, Data: []byte(`{"kind":"secret"}`)}
	close(in)

	output := drainToolCallStream(DetectToolCallsInStream(t.Context(), in, sink))
	if len(output) != 2 || output[0].Type != TokenTypeToolCall || output[1].Type != TokenTypeText {
		t.Fatalf("output = %#v, want tool call followed by replayed text", output)
	}
	if len(observations) != 1 {
		t.Fatalf("observations = %#v, want one terminal outcome", observations)
	}
	observation := observations[0]
	if observation.Name != "tool_call_stream_finished" || observation.Source != "ai:DetectToolCallsInStream" {
		t.Fatalf("observation identity = %#v", observation)
	}
	for key, want := range map[string]any{
		"outcome":                  "completed",
		"input_token_events":       2,
		"output_token_events":      2,
		"detected_tool_call_count": 1,
		"rejected_candidate_count": 1,
		"eof_pending":              false,
		"last_rejection_reason":    "parse_failed",
		"last_tool_call_name":      "echo",
	} {
		if got := observation.Fields[key]; got != want {
			t.Errorf("field %q = %#v, want %#v", key, got, want)
		}
	}
	for _, key := range []string{"last_tool_call_args", "last_rejected_candidate", "pending_data"} {
		if _, ok := observation.Fields[key]; ok {
			t.Errorf("default policy captured %q: %#v", key, observation.Fields)
		}
	}
	if strings.Contains(fmt.Sprint(observation.Fields), "secret") {
		t.Fatalf("default policy leaked content: %#v", observation.Fields)
	}

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("spans = %d, want 1", len(spans))
	}
	attrs := spanAttributes(spans[0].Attributes())
	assertIntAttribute(t, attrs, "ai.input_token_events", 2)
	assertIntAttribute(t, attrs, "ai.output_token_events", 2)
	assertIntAttribute(t, attrs, "ai.tool_call_count", 1)
	assertIntAttribute(t, attrs, "ai.rejected_tool_call_candidate_count", 1)
	assertStringAttribute(t, attrs, "ai.tool_call_stream.outcome", "completed")
}

func TestToolCallStreamObservationCapturesOnlyPolicyEnabledContent(t *testing.T) {
	var observation gai.Observation
	sink := gai.ObservationSinkFunc(func(_ context.Context, emitted gai.Observation) {
		observation = emitted
	})
	ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{
		Completion: gai.CaptureEnabled,
		ToolInput:  gai.CaptureEnabled,
		Redact: func(_ context.Context, _ gai.ContentKind, value []byte) ([]byte, error) {
			return bytes.ReplaceAll(value, []byte("secret"), []byte("[redacted]")), nil
		},
	})
	in := make(chan Token, 2)
	in <- Token{Type: TokenTypeText, Data: []byte(`{"type":"function","name":"echo","arguments":{"value":"secret"}}`)}
	in <- Token{Type: TokenTypeText, Data: []byte(`{"kind":"secret"}`)}
	close(in)

	drainToolCallStream(DetectToolCallsInStream(ctx, in, sink))

	if got := observation.Fields["last_tool_call_args"]; got != `{"value":"[redacted]"}` {
		t.Fatalf("captured tool args = %#v", got)
	}
	if got := observation.Fields["last_rejected_candidate"]; got != `{"kind":"[redacted]"}` {
		t.Fatalf("captured rejected candidate = %#v", got)
	}
	if observation.Fields["last_tool_call_args_content_kind"] != string(gai.ContentKindToolInput) {
		t.Fatalf("tool args metadata = %#v", observation.Fields)
	}
	if observation.Fields["last_rejected_candidate_content_kind"] != string(gai.ContentKindCompletion) {
		t.Fatalf("candidate metadata = %#v", observation.Fields)
	}
}

func TestToolCallStreamObservationReportsEOFPendingCandidate(t *testing.T) {
	var observation gai.Observation
	sink := gai.ObservationSinkFunc(func(_ context.Context, emitted gai.Observation) {
		observation = emitted
	})
	ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{Completion: gai.CaptureEnabled})
	in := make(chan Token, 1)
	in <- Token{Type: TokenTypeText, Data: []byte(`{"type":"function","name":"echo"`)}
	close(in)

	output := drainToolCallStream(DetectToolCallsInStream(ctx, in, sink))

	if len(output) != 1 || string(output[0].Data) != `{"type":"function","name":"echo"` {
		t.Fatalf("output = %#v, want unresolved candidate replayed", output)
	}
	if observation.Fields["outcome"] != "completed" || observation.Fields["eof_pending"] != true {
		t.Fatalf("terminal outcome = %#v", observation.Fields)
	}
	if observation.Fields["rejected_candidate_count"] != 1 || observation.Fields["last_rejection_reason"] != "end_of_stream" {
		t.Fatalf("rejection summary = %#v", observation.Fields)
	}
	if observation.Fields["pending_data"] != `{"type":"function","name":"echo"` {
		t.Fatalf("captured pending data = %#v", observation.Fields)
	}
}

func TestToolCallStreamObservationReportsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	var observation gai.Observation
	sink := gai.ObservationSinkFunc(func(_ context.Context, emitted gai.Observation) {
		observation = emitted
	})
	in := make(chan Token)
	out := DetectToolCallsInStream(ctx, in, sink)
	cancel()

	drainToolCallStream(out)

	if observation.Name != "tool_call_stream_finished" || observation.Fields["outcome"] != "canceled" {
		t.Fatalf("cancellation observation = %#v", observation)
	}
	if observation.Fields["input_token_events"] != 0 || observation.Fields["output_token_events"] != 0 {
		t.Fatalf("cancellation counts = %#v", observation.Fields)
	}
}

func TestToolCallStreamObservationCancellationWinsOverClosedInput(t *testing.T) {
	recorder, restore := installToolCallStreamSpanRecorder(t)
	defer restore()

	const attempts = 100
	for attempt := 0; attempt < attempts; attempt++ {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		in := make(chan Token)
		close(in)

		var observation gai.Observation
		sink := gai.ObservationSinkFunc(func(_ context.Context, emitted gai.Observation) {
			observation = emitted
		})
		drainToolCallStream(DetectToolCallsInStream(ctx, in, sink))

		if observation.Fields["outcome"] != "canceled" {
			t.Fatalf("attempt %d outcome = %#v, want canceled", attempt, observation.Fields["outcome"])
		}
		if observation.Fields["input_token_events"] != 0 || observation.Fields["output_token_events"] != 0 {
			t.Fatalf("attempt %d counts = %#v", attempt, observation.Fields)
		}
	}

	spans := recorder.Ended()
	if len(spans) != attempts {
		t.Fatalf("spans = %d, want %d", len(spans), attempts)
	}
	for attempt, span := range spans {
		attrs := spanAttributes(span.Attributes())
		if got := attrs["ai.tool_call_stream.outcome"].AsString(); got != "canceled" {
			t.Fatalf("attempt %d span outcome = %q, want canceled", attempt, got)
		}
	}
}

func installToolCallStreamSpanRecorder(t *testing.T) (*tracetest.SpanRecorder, func()) {
	t.Helper()
	previous := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(provider)
	return recorder, func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previous)
	}
}

func drainToolCallStream(stream <-chan Token) []Token {
	var tokens []Token
	for token := range stream {
		tokens = append(tokens, token)
	}
	return tokens
}
