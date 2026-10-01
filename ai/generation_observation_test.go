package ai

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/lace-ai/gai"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

func TestGenerationObservationEmitsFinalizedOutcomeToSink(t *testing.T) {
	var emitted gai.Observation
	sink := gai.ObservationSinkFunc(func(_ context.Context, observation gai.Observation) {
		emitted = observation
	})

	_, observation := StartGenerationObservation(context.Background(), AIRequest{}, GenerationConfig{
		Provider: "openai",
		Model:    "gpt-test",
		Sink:     sink,
	})
	observation.Finish(GenerationResult{
		ResponseModel: "gpt-resolved",
		RequestID:     "req-1",
		FinishReason:  "stop",
		Usage:         &Usage{InputTokens: 3, OutputTokens: 5},
		HTTPStatus:    200,
	})

	if emitted.Name != "generation_finished" || emitted.Source != "ai:GenerationObservation" {
		t.Fatalf("observation identity = %#v", emitted)
	}
	if emitted.Fields["provider"] != "openai" || emitted.Fields["model"] != "gpt-test" || emitted.Fields["response_model"] != "gpt-resolved" {
		t.Fatalf("generation fields = %#v", emitted.Fields)
	}
	if emitted.Fields["input_tokens"] != 3 || emitted.Fields["output_tokens"] != 5 || emitted.Fields["total_tokens"] != 8 {
		t.Fatalf("generation usage = %#v", emitted.Fields)
	}
}

func TestGenerationObservationRecordsSemanticContract(t *testing.T) {
	recorder, restore := installGenerationSpanRecorder(t)
	defer restore()

	ctx, parent := otel.Tracer("test").Start(context.Background(), "parent")
	_, observation := StartGenerationObservation(ctx, AIRequest{
		Prompt: "private prompt", MaxTokens: 42,
		ResponseFormat: ResponseFormat{Type: ResponseFormatJSONSchema},
	}, GenerationConfig{Provider: "test-provider", Model: "requested-model", Streaming: true})
	observation.ObserveToken(Token{Type: TokenTypeText, Text: "private completion"})
	observation.ObserveToken(Token{Type: TokenTypeToolCall, ToolCall: &ToolCall{Name: "search"}})
	observation.ObserveToken(Token{Type: TokenTypeCompletion, Completion: &Completion{
		Model: "resolved-model", RequestID: "request-1", FinishReason: "tool_calls",
		UsageReported: true,
		Usage:         Usage{InputTokens: 11, OutputTokens: 7, ReasoningTokens: 3, CachedTokens: 2},
	}})
	observation.Finish(GenerationResult{HTTPStatus: 200})
	observation.Finish(GenerationResult{Err: errors.New("must be ignored")})
	parent.End()

	span := generationSpan(t, recorder.Ended())
	if span.Name() != "chat requested-model" {
		t.Fatalf("span name = %q, want %q", span.Name(), "chat requested-model")
	}
	attrs := spanAttributes(span.Attributes())
	assertStringAttribute(t, attrs, "langfuse.observation.type", "generation")
	assertStringAttribute(t, attrs, "langfuse.observation.model.name", "resolved-model")
	assertStringAttribute(t, attrs, "gen_ai.provider.name", "test-provider")
	assertStringAttribute(t, attrs, "gen_ai.request.model", "requested-model")
	assertStringAttribute(t, attrs, "gen_ai.response.id", "request-1")
	assertIntAttribute(t, attrs, "gen_ai.usage.input_tokens", 11)
	assertIntAttribute(t, attrs, "gen_ai.usage.output_tokens", 7)
	assertIntAttribute(t, attrs, "gen_ai.usage.total_tokens", 18)
	assertIntAttribute(t, attrs, "gai.gen_ai.response.tool_call_count", 1)
	assertIntAttribute(t, attrs, "http.response.status_code", 200)
	if got := attrs["gen_ai.response.finish_reasons"].AsStringSlice(); len(got) != 1 || got[0] != "tool_calls" {
		t.Fatalf("finish reasons = %v", got)
	}
	if value := attrs["langfuse.observation.completion_start_time"].AsString(); value == "" {
		t.Fatal("completion start time missing")
	} else if _, err := time.Parse(time.RFC3339Nano, value); err != nil {
		t.Fatalf("completion start time %q: %v", value, err)
	}
	var completionEvents int
	for _, event := range span.Events() {
		if event.Name == "gen_ai.completion.start" {
			completionEvents++
		}
	}
	if completionEvents != 1 {
		t.Fatalf("completion events = %#v", span.Events())
	}
	if span.Parent().SpanID() != parent.SpanContext().SpanID() {
		t.Fatalf("parent span ID = %s, want %s", span.Parent().SpanID(), parent.SpanContext().SpanID())
	}
	if span.SpanKind() != trace.SpanKindClient {
		t.Fatalf("span kind = %s, want client", span.SpanKind())
	}
	for key, value := range attrs {
		if strings.Contains(value.String(), "private prompt") || strings.Contains(value.String(), "private completion") {
			t.Fatalf("content leaked through %s", key)
		}
	}
}

func TestGenerationObservationFailureRecordsOneException(t *testing.T) {
	recorder, restore := installGenerationSpanRecorder(t)
	defer restore()

	_, observation := StartGenerationObservation(context.Background(), AIRequest{}, GenerationConfig{Provider: "test", Model: "m"})
	observation.Finish(GenerationResult{Err: errors.New("generation failed")})

	span := generationSpan(t, recorder.Ended())
	var exceptions int
	for _, event := range span.Events() {
		if event.Name == "exception" {
			exceptions++
		}
	}
	if exceptions != 1 {
		t.Fatalf("exception events = %d, want 1; events = %#v", exceptions, span.Events())
	}
}

func TestGenerationObservationTimingBaselineDoesNotPredateSpan(t *testing.T) {
	recorder, restore := installGenerationSpanRecorder(t)
	defer restore()

	_, observation := StartGenerationObservation(context.Background(), AIRequest{}, GenerationConfig{Provider: "test", Model: "m"})
	started := recorder.Started()
	if len(started) != 1 {
		t.Fatalf("started spans = %d, want 1", len(started))
	}
	if observation.startedAt.Before(started[0].StartTime()) {
		t.Fatalf("timing baseline %s predates span start %s", observation.startedAt, started[0].StartTime())
	}
	observation.Finish(GenerationResult{})
}

func TestGenerationObservationDistinguishesMissingAndReportedZeroUsage(t *testing.T) {
	recorder, restore := installGenerationSpanRecorder(t)
	defer restore()

	_, missing := StartGenerationObservation(context.Background(), AIRequest{}, GenerationConfig{Provider: "test", Model: "m"})
	missing.Finish(GenerationResult{})
	zero := Usage{}
	_, reported := StartGenerationObservation(context.Background(), AIRequest{}, GenerationConfig{Provider: "test", Model: "m"})
	reported.Finish(GenerationResult{Usage: &zero})

	var spans []sdktrace.ReadOnlySpan
	for _, span := range recorder.Ended() {
		if isGenerationSpan(span) {
			spans = append(spans, span)
		}
	}
	if len(spans) != 2 {
		t.Fatalf("generation spans = %d, want 2", len(spans))
	}
	missingAttrs := spanAttributes(spans[0].Attributes())
	if _, ok := missingAttrs["gen_ai.usage.input_tokens"]; ok {
		t.Fatal("missing usage emitted input tokens")
	}
	reportedAttrs := spanAttributes(spans[1].Attributes())
	assertIntAttribute(t, reportedAttrs, "gen_ai.usage.input_tokens", 0)
	assertIntAttribute(t, reportedAttrs, "gen_ai.usage.output_tokens", 0)
}

func TestGenerationObservationFinalToolCallCountTakesPrecedence(t *testing.T) {
	recorder, restore := installGenerationSpanRecorder(t)
	defer restore()

	_, observation := StartGenerationObservation(context.Background(), AIRequest{}, GenerationConfig{Provider: "test", Model: "m", Streaming: true})
	observation.ObserveToken(Token{Type: TokenTypeToolCall, ToolCall: &ToolCall{Name: "search"}})
	observation.Finish(GenerationResult{ToolCallCount: 1})

	attrs := spanAttributes(generationSpan(t, recorder.Ended()).Attributes())
	assertIntAttribute(t, attrs, "gai.gen_ai.response.tool_call_count", 1)
}

func TestGenerationObservationCreatesOneParentedSpanPerAttempt(t *testing.T) {
	recorder, restore := installGenerationSpanRecorder(t)
	defer restore()

	ctx, parent := otel.Tracer("test").Start(context.Background(), "loop.attempts")
	for range 2 {
		_, observation := StartGenerationObservation(ctx, AIRequest{}, GenerationConfig{Provider: "test", Model: "m"})
		observation.Finish(GenerationResult{})
	}
	parent.End()

	var generations []sdktrace.ReadOnlySpan
	for _, span := range recorder.Ended() {
		if isGenerationSpan(span) {
			generations = append(generations, span)
		}
	}
	if len(generations) != 2 {
		t.Fatalf("generation spans = %d, want one per attempt", len(generations))
	}
	if generations[0].SpanContext().SpanID() == generations[1].SpanContext().SpanID() {
		t.Fatal("attempt spans reused a span ID")
	}
	for _, span := range generations {
		if span.Parent().SpanID() != parent.SpanContext().SpanID() {
			t.Fatalf("attempt parent = %s, want %s", span.Parent().SpanID(), parent.SpanContext().SpanID())
		}
	}
}

func installGenerationSpanRecorder(t *testing.T) (*tracetest.SpanRecorder, func()) {
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

func generationSpan(t *testing.T, spans []sdktrace.ReadOnlySpan) sdktrace.ReadOnlySpan {
	t.Helper()
	var found sdktrace.ReadOnlySpan
	for _, span := range spans {
		if !isGenerationSpan(span) {
			continue
		}
		if found != nil {
			t.Fatal("duplicate generation span")
		}
		found = span
	}
	if found == nil {
		t.Fatal("generation span missing")
	}
	return found
}

func isGenerationSpan(span sdktrace.ReadOnlySpan) bool {
	for _, attr := range span.Attributes() {
		if string(attr.Key) == "langfuse.observation.type" && attr.Value.AsString() == "generation" {
			return true
		}
	}
	return false
}

func spanAttributes(attrs []attribute.KeyValue) map[string]attribute.Value {
	result := make(map[string]attribute.Value, len(attrs))
	for _, attr := range attrs {
		result[string(attr.Key)] = attr.Value
	}
	return result
}

func assertStringAttribute(t *testing.T, attrs map[string]attribute.Value, key, want string) {
	t.Helper()
	if got := attrs[key].AsString(); got != want {
		t.Fatalf("%s = %q, want %q", key, got, want)
	}
}

func assertIntAttribute(t *testing.T, attrs map[string]attribute.Value, key string, want int64) {
	t.Helper()
	if got := attrs[key].AsInt64(); got != want {
		t.Fatalf("%s = %d, want %d", key, got, want)
	}
}

func TestGenerationObservationUsesCanonicalInputAndToken(t *testing.T) {
	recorder, restore := installGenerationSpanRecorder(t)
	defer restore()
	_, observation := StartGenerationObservation(t.Context(), AIRequest{
		Prompt: "obsolete private prompt",
		Messages: []Message{
			TextMessage(RoleSystem, "system"),
			TextMessage(RoleUser, "question"),
			TextMessage(RoleAssistant, "private answer"),
		},
	}, GenerationConfig{Provider: "test", Model: "m", Streaming: true})
	observation.ObserveToken(Token{Type: TokenTypeToolCall, ToolCall: &ToolCall{Name: "stale"}, Part: &ContentPart{Kind: ContentText, Text: "answer"}})
	observation.ObserveToken(Token{Part: &ContentPart{Kind: ContentToolCall, ToolCall: &ToolCall{Name: "actual"}}})
	observation.Finish(GenerationResult{})
	attrs := spanAttributes(generationSpan(t, recorder.Ended()).Attributes())
	assertIntAttribute(t, attrs, "ai.prompt_length", int64(len("systemquestion")))
	assertIntAttribute(t, attrs, "gai.gen_ai.response.tool_call_count", 1)
}
