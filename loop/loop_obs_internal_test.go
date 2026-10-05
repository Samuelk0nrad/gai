package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/internal/obstest"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

type observedTestTool struct {
	name string
	call func(context.Context, ai.ToolCall) (string, error)
}

func TestLoopRunSpanUsesRetryPolicyLimit(t *testing.T) {
	recorder := obstest.Install(t)
	l := &Loop{RetryPolicy: &RetryPolicy{MaxRetries: 0}}

	_, state := newLoopRunState(t.Context(), l)
	state.finish()

	var runSpan sdktrace.ReadOnlySpan
	for _, span := range recorder.Ended() {
		if span.Name() == "loop.run" {
			runSpan = span
			break
		}
	}
	if runSpan == nil {
		t.Fatal("loop run span was not recorded")
	}
	if got := obstest.Attributes(runSpan)["loop.retry_limit"].AsInt64(); got != 0 {
		t.Fatalf("loop.retry_limit = %d, want 0", got)
	}
}

// TestToolObservationEmitsOutcomeToSink checks that finished observations identify the tool call
// and report its successful outcome.
func TestToolObservationEmitsOutcomeToSink(t *testing.T) {
	var emitted gai.Observation
	sink := gai.ObservationSinkFunc(func(_ context.Context, observation gai.Observation) {
		emitted = observation
	})
	call := ai.ToolCall{ID: "call-1", Type: "function", Name: "test", Args: json.RawMessage(`{}`)}

	callObservedTool(t.Context(), call, []Tool{observedTestTool{name: "test", call: func(context.Context, ai.ToolCall) (string, error) {
		return "ok", nil
	}}}, sink)

	if emitted.Name != "loop_tool_finished" || emitted.Source != "loop:Tool" {
		t.Fatalf("observation identity = %#v", emitted)
	}
	if emitted.Fields["tool_name"] != "test" || emitted.Fields["tool_call_id"] != "call-1" || emitted.Fields["outcome"] != toolOutcomeSuccess {
		t.Fatalf("tool fields = %#v", emitted.Fields)
	}
}

// TestExecuteToolCallsDurationExcludesSinkEmissionLatency keeps handler duration independent of
// a slow observation sink and consistent between events and telemetry.
func TestExecuteToolCallsDurationExcludesSinkEmissionLatency(t *testing.T) {
	sinkEntered := make(chan struct{})
	releaseSink := make(chan struct{})
	var observed gai.Observation
	sink := gai.ObservationSinkFunc(func(_ context.Context, observation gai.Observation) {
		observed = observation
		close(sinkEntered)
		<-releaseSink
	})
	tool := observedTestTool{name: "fast", call: func(context.Context, ai.ToolCall) (string, error) {
		return "ok", nil
	}}
	l := &Loop{Tools: []Tool{tool}, ObservationSink: sink}
	iteration := &Iteration{Parts: make([]IterationPart, 1)}
	calls := []pendingToolCall{{partIndex: 0, call: ai.ToolCall{ID: "call-fast", Type: "function", Name: "fast", Args: json.RawMessage(`{}`)}}}
	events := make(chan Event, 2)
	done := make(chan error, 1)

	go func() { done <- l.executeToolCalls(t.Context(), iteration, calls, l.Tools, events, 1, 1, 0) }()
	<-sinkEntered
	time.Sleep(50 * time.Millisecond)
	close(releaseSink)
	if err := <-done; err != nil {
		t.Fatalf("executeToolCalls error = %v", err)
	}
	close(events)

	var result Event
	for event := range events {
		if event.Type == EventToolResult {
			result = event
		}
	}
	if result.Type != EventToolResult {
		t.Fatal("tool result event was not emitted")
	}
	if result.Duration >= 25*time.Millisecond {
		t.Fatalf("tool duration = %s, includes sink emission latency", result.Duration)
	}
	if got, ok := observed.Fields["duration_ms"].(int64); !ok || got != result.Duration.Milliseconds() {
		t.Fatalf("observation duration_ms = %#v, want %d", observed.Fields["duration_ms"], result.Duration.Milliseconds())
	}
}

func (t observedTestTool) Name() string              { return t.name }
func (t observedTestTool) Description() string       { return "Test tool." }
func (t observedTestTool) Params() ai.ToolParameters { return NewEchoTool().Params() }
// Function delegates to a supplied callback so observation tests control each handler outcome.
func (t observedTestTool) Function(ctx context.Context, call ai.ToolCall) (string, error) {
	return t.call(ctx, call)
}

// TestToolObservationOutcomes checks safe telemetry classifications for success, empty output,
// tool errors, deadlines, and cancellation.
func TestToolObservationOutcomes(t *testing.T) {
	tests := []struct {
		name    string
		ctx     func(*testing.T) context.Context
		call    func(context.Context, ai.ToolCall) (string, error)
		outcome string
	}{
		{name: "success", call: func(context.Context, ai.ToolCall) (string, error) {
			return "ok", nil
		}, outcome: toolOutcomeSuccess},
		{name: "tool error", call: func(context.Context, ai.ToolCall) (string, error) {
			return "", errors.New("tool-error-sentinel-secret")
		}, outcome: toolOutcomeError},
		{name: "deadline", ctx: func(t *testing.T) context.Context {
			ctx, cancel := context.WithDeadline(t.Context(), time.Now().Add(-time.Second))
			t.Cleanup(cancel)
			return ctx
		}, call: func(ctx context.Context, _ ai.ToolCall) (string, error) {
			return "", fmt.Errorf("wrapped: %w", ctx.Err())
		}, outcome: toolOutcomeDeadline},
		{name: "cancellation", ctx: func(t *testing.T) context.Context {
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			return ctx
		}, call: func(ctx context.Context, _ ai.ToolCall) (string, error) {
			return "", fmt.Errorf("wrapped: %w", ctx.Err())
		}, outcome: toolOutcomeCancellation},
		{name: "empty success", call: func(context.Context, ai.ToolCall) (string, error) {
			return "", nil
		}, outcome: toolOutcomeSuccess},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := obstest.Install(t)
			ctx := t.Context()
			if tt.ctx != nil {
				ctx = tt.ctx(t)
			}
			call := ai.ToolCall{ID: "call-1", Type: "function", Name: "test", Args: json.RawMessage(`{}`)}
			response, _ := callObservedTool(ctx, call, []Tool{observedTestTool{name: "test", call: tt.call}})
			if response == nil {
				t.Fatal("observed tool returned nil response")
			}

			span := requireToolSpans(t, recorder, 1)[0]
			attrs := obstest.Attributes(span)
			if got := attrs["gai.tool.outcome"].AsString(); got != tt.outcome {
				t.Fatalf("gai.tool.outcome = %q, want %q", got, tt.outcome)
			}
			wantStatus := "error"
			if tt.outcome == toolOutcomeSuccess {
				wantStatus = "success"
			}
			if got := attrs["tool.status"].AsString(); got != wantStatus {
				t.Fatalf("tool.status = %q, want %q", got, wantStatus)
			}
			if strings.Contains(toolSpanText(span), "sentinel-secret") {
				t.Fatalf("raw tool error reached span: %s", toolSpanText(span))
			}
		})
	}
}

// TestObservedToolPanicFinalizesAndRepanics ensures a panic closes its tool span with safe
// metadata before propagating the original panic value.
func TestObservedToolPanicFinalizesAndRepanics(t *testing.T) {
	recorder := obstest.Install(t)
	panicValue := errors.New("panic-sentinel-secret")
	var emitted gai.Observation
	sink := gai.ObservationSinkFunc(func(_ context.Context, observation gai.Observation) {
		emitted = observation
	})
	call := ai.ToolCall{ID: "call-panic", Type: "function", Name: "panic", Args: json.RawMessage(`{}`)}
	tool := observedTestTool{name: "panic", call: func(context.Context, ai.ToolCall) (string, error) {
		time.Sleep(10 * time.Millisecond)
		panic(panicValue)
	}}

	var recovered any
	func() {
		defer func() { recovered = recover() }()
		callObservedTool(t.Context(), call, []Tool{tool}, sink)
	}()
	if recovered != panicValue {
		t.Fatalf("recovered panic = %#v, want original value", recovered)
	}

	span := requireToolSpans(t, recorder, 1)[0]
	attrs := obstest.Attributes(span)
	if attrs["gai.tool.outcome"].AsString() != toolOutcomePanic || attrs["error.type"].AsString() != "gai.tool.panic" {
		t.Fatalf("panic span attributes = %#v", attrs)
	}
	if strings.Contains(toolSpanText(span), "panic-sentinel-secret") {
		t.Fatalf("panic value reached span: %s", toolSpanText(span))
	}
	if got, ok := emitted.Fields["duration_ms"].(int64); !ok || got <= 0 {
		t.Fatalf("panic observation duration_ms = %#v, want a positive measured duration", emitted.Fields["duration_ms"])
	}
}

// TestToolObservationUsesPolicyGatedContentAliases verifies that opted-in, redacted content is
// consistent across tool, GenAI, and Langfuse attributes.
func TestToolObservationUsesPolicyGatedContentAliases(t *testing.T) {
	recorder := obstest.Install(t)
	ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{
		ToolInput: gai.CaptureEnabled, ToolOutput: gai.CaptureEnabled,
		Redact: func(_ context.Context, _ gai.ContentKind, value []byte) ([]byte, error) {
			return []byte(strings.ReplaceAll(string(value), "secret", "[redacted]")), nil
		},
	})
	call := ai.ToolCall{ID: "call-content", Type: "function", Name: "content", Args: json.RawMessage(`{"text":"secret"}`)}
	tool := observedTestTool{name: "content", call: func(context.Context, ai.ToolCall) (string, error) {
		return `{"result":"secret"}`, nil
	}}
	callObservedTool(ctx, call, []Tool{tool})

	attrs := obstest.Attributes(requireToolSpans(t, recorder, 1)[0])
	wantInput := `{"text":"[redacted]"}`
	for _, key := range []string{"tool.input", "gen_ai.tool.call.arguments", "langfuse.observation.input"} {
		if got := attrs[key].AsString(); got != wantInput {
			t.Fatalf("%s = %q, want %q", key, got, wantInput)
		}
	}
	wantOutput := `{"result":"[redacted]"}`
	for _, key := range []string{"tool.output", "gen_ai.tool.call.result", "langfuse.observation.output"} {
		if got := attrs[key].AsString(); got != wantOutput {
			t.Fatalf("%s = %q, want %q", key, got, wantOutput)
		}
	}
	if !attrs["tool.input.redaction_applied"].AsBool() || !attrs["tool.output.redaction_applied"].AsBool() {
		t.Fatalf("redaction metadata missing: %#v", attrs)
	}
	if attrs["langfuse.observation.type"].AsString() != "tool" || attrs["gen_ai.operation.name"].AsString() != "execute_tool" {
		t.Fatalf("semantic tool attributes missing: %#v", attrs)
	}
}

// TestToolObservationCapturesInvalidJSONAndTruncatesLargeValues retains malformed input for
// opted-in diagnostics while bounding captured content and recording truncation.
func TestToolObservationCapturesInvalidJSONAndTruncatesLargeValues(t *testing.T) {
	t.Run("invalid JSON", func(t *testing.T) {
		recorder := obstest.Install(t)
		ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{ToolInput: gai.CaptureEnabled})
		call := ai.ToolCall{ID: "call-invalid", Type: "function", Name: "content", Args: json.RawMessage(`{"broken"`)}
		tool := observedTestTool{name: "content", call: func(context.Context, ai.ToolCall) (string, error) {
			return "ok", nil
		}}
		callObservedTool(ctx, call, []Tool{tool})

		attrs := obstest.Attributes(requireToolSpans(t, recorder, 1)[0])
		if got := attrs["gen_ai.tool.call.arguments"].AsString(); got != `{"broken"` {
			t.Fatalf("captured invalid JSON = %q", got)
		}
	})

	t.Run("large values", func(t *testing.T) {
		recorder := obstest.Install(t)
		ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{
			ToolInput: gai.CaptureEnabled, ToolOutput: gai.CaptureEnabled, MaxBytes: 24,
		})
		call := ai.ToolCall{ID: "call-large", Type: "function", Name: "content", Args: json.RawMessage(`{"text":"abcdefghijklmnopqrstuvwxyz"}`)}
		tool := observedTestTool{name: "content", call: func(context.Context, ai.ToolCall) (string, error) {
			return strings.Repeat("x", 64), nil
		}}
		callObservedTool(ctx, call, []Tool{tool})

		attrs := obstest.Attributes(requireToolSpans(t, recorder, 1)[0])
		if !attrs["tool.input.truncated"].AsBool() || !attrs["tool.output.truncated"].AsBool() {
			t.Fatalf("truncation metadata missing: %#v", attrs)
		}
		if got := len(attrs["langfuse.observation.input"].AsString()); got > 24 {
			t.Fatalf("captured input bytes = %d, want <= 24", got)
		}
		if got := len(attrs["langfuse.observation.output"].AsString()); got > 24 {
			t.Fatalf("captured output bytes = %d, want <= 24", got)
		}
	})
}

// TestToolObservationOmitsContentByDefault ensures default tracing exports no tool arguments or
// results through any supported content alias.
func TestToolObservationOmitsContentByDefault(t *testing.T) {
	recorder := obstest.Install(t)
	call := ai.ToolCall{ID: "call-private", Type: "function", Name: "content", Args: json.RawMessage(`{"secret":"input"}`)}
	tool := observedTestTool{name: "content", call: func(context.Context, ai.ToolCall) (string, error) {
		return "output-secret", nil
	}}
	callObservedTool(t.Context(), call, []Tool{tool})

	attrs := obstest.Attributes(requireToolSpans(t, recorder, 1)[0])
	for _, key := range []string{
		"tool.input", "gen_ai.tool.call.arguments", "langfuse.observation.input",
		"tool.output", "gen_ai.tool.call.result", "langfuse.observation.output",
	} {
		if _, ok := attrs[key]; ok {
			t.Fatalf("disabled content attribute %q was exported: %#v", key, attrs[key])
		}
	}
}

// TestToolObservationFinishesOnce ensures the first finalization fixes the span outcome despite
// later error or panic completion attempts.
func TestToolObservationFinishesOnce(t *testing.T) {
	recorder := obstest.Install(t)
	_, observation := startToolSpan(t.Context(), ai.ToolCall{ID: "call-once", Name: "once"})
	observation.finish(&ToolResult{Text: "ok"}, 0)
	observation.finish(&ToolResult{Err: errors.New("late error")}, 0)
	observation.finishPanic(0)

	span := requireToolSpans(t, recorder, 1)[0]
	if got := obstest.Attributes(span)["gai.tool.outcome"].AsString(); got != toolOutcomeSuccess {
		t.Fatalf("gai.tool.outcome = %q, want first result %q", got, toolOutcomeSuccess)
	}
}

// TestExecuteToolCallsCreatesConcurrentChildSpans verifies overlapping handlers and sink
// callbacks produce one correctly parented span per invocation.
func TestExecuteToolCallsCreatesConcurrentChildSpans(t *testing.T) {
	recorder := obstest.Install(t)
	entered := make(chan string, 2)
	release := make(chan struct{})
	sinkEntered := make(chan struct{}, 2)
	releaseSink := make(chan struct{})
	var observationsMu sync.Mutex
	var observations []gai.Observation
	tool := observedTestTool{name: "concurrent", call: func(_ context.Context, call ai.ToolCall) (string, error) {
		entered <- call.ID
		<-release
		return call.ID, nil
	}}
	l := &Loop{
		Tools: []Tool{tool},
		ObservationSink: gai.ObservationSinkFunc(func(_ context.Context, observation gai.Observation) {
			observationsMu.Lock()
			observations = append(observations, observation)
			observationsMu.Unlock()
			sinkEntered <- struct{}{}
			<-releaseSink
		}),
	}
	iteration := &Iteration{Parts: make([]IterationPart, 2)}
	calls := []pendingToolCall{
		{partIndex: 0, call: ai.ToolCall{ID: "call-1", Type: "function", Name: "concurrent", Args: json.RawMessage(`{}`)}},
		{partIndex: 1, call: ai.ToolCall{ID: "call-2", Type: "function", Name: "concurrent", Args: json.RawMessage(`{}`)}},
	}
	ctx, parent := otel.Tracer("test").Start(t.Context(), "loop.iteration")
	done := make(chan error, 1)
	go func() { done <- l.executeToolCalls(ctx, iteration, calls, l.Tools, nil, 0, 0, 0) }()

	seen := map[string]bool{}
	for range calls {
		select {
		case id := <-entered:
			seen[id] = true
		case <-time.After(time.Second):
			t.Fatal("tool calls did not overlap")
		}
	}
	close(release)
	for range calls {
		select {
		case <-sinkEntered:
		case <-time.After(time.Second):
			t.Fatal("observation sink calls did not overlap")
		}
	}
	close(releaseSink)
	if err := <-done; err != nil {
		t.Fatalf("executeToolCalls error: %v", err)
	}
	parent.End()
	if !seen["call-1"] || !seen["call-2"] {
		t.Fatalf("executed calls = %#v", seen)
	}
	observationsMu.Lock()
	defer observationsMu.Unlock()
	if len(observations) != 2 {
		t.Fatalf("observations = %#v, want one per concurrent tool call", observations)
	}
	for _, observation := range observations {
		if observation.Name != "loop_tool_finished" {
			t.Fatalf("observation name = %q, want loop_tool_finished", observation.Name)
		}
	}

	spans := requireToolSpans(t, recorder, 2)
	for _, span := range spans {
		if span.Parent().SpanID() != parent.SpanContext().SpanID() {
			t.Fatalf("tool span parent = %s, want %s", span.Parent().SpanID(), parent.SpanContext().SpanID())
		}
		if obstest.Attributes(span)["gai.tool.outcome"].AsString() != toolOutcomeSuccess {
			t.Fatalf("tool span outcome = %#v", obstest.Attributes(span))
		}
	}
}

// TestExecuteToolCallsEmitsToolEvents checks start-before-result ordering and preservation of
// call, iteration, attempt, and retry identifiers.
func TestExecuteToolCallsEmitsToolEvents(t *testing.T) {
	tool := observedTestTool{name: "event", call: func(context.Context, ai.ToolCall) (string, error) {
		return "ok", nil
	}}
	l := &Loop{Tools: []Tool{tool}}
	iteration := &Iteration{Parts: make([]IterationPart, 1)}
	calls := []pendingToolCall{{
		partIndex: 0,
		call:      ai.ToolCall{ID: "call-event", Type: "function", Name: "event", Args: json.RawMessage(`{}`)},
	}}
	events := make(chan Event, 2)

	if err := l.executeToolCalls(t.Context(), iteration, calls, l.Tools, events, 3, 4, 5); err != nil {
		t.Fatalf("executeToolCalls error: %v", err)
	}
	close(events)

	var got []Event
	for event := range events {
		got = append(got, event)
	}
	if len(got) != 2 {
		t.Fatalf("tool events = %#v, want start and result", got)
	}
	if got[0].Type != EventToolStart || got[1].Type != EventToolResult {
		t.Fatalf("tool event types = %v, %v; want %v, %v", got[0].Type, got[1].Type, EventToolStart, EventToolResult)
	}
	for _, event := range got {
		if event.IterationCount != 3 || event.AttemptID != 4 || event.RetryCount != 5 || event.ToolCall == nil || event.ToolCall.ID != "call-event" {
			t.Fatalf("tool event = %#v", event)
		}
	}
}

// TestExecuteToolCallsEmitsToolErrorWhenResponseProcessingFails preserves the processor error
// cause and emits a tool-error event when result processing fails.
func TestExecuteToolCallsEmitsToolErrorWhenResponseProcessingFails(t *testing.T) {
	processorErr := errors.New("reject tool response")
	tool := observedTestTool{name: "event", call: func(context.Context, ai.ToolCall) (string, error) {
		return "ok", nil
	}}
	l := &Loop{
		Tools: []Tool{tool},
		ToolResultProcessor: ToolResultProcessorFunc(func(context.Context, ToolPolicyInput, ToolResult) (ToolResult, error) {
			return ToolResult{}, processorErr
		}),
	}
	iteration := &Iteration{Parts: make([]IterationPart, 1)}
	calls := []pendingToolCall{{
		partIndex: 0,
		call:      ai.ToolCall{ID: "call-event", Type: "function", Name: "event", Args: json.RawMessage(`{}`)},
	}}
	events := make(chan Event, 2)

	err := l.executeToolCalls(t.Context(), iteration, calls, l.Tools, events, 3, 4, 5)
	if !errors.Is(err, ErrToolResultProcess) || !errors.Is(err, processorErr) {
		t.Fatalf("executeToolCalls error = %v, want wrapped processor error", err)
	}
	close(events)

	got := make([]Event, 0, 2)
	for event := range events {
		got = append(got, event)
	}
	if len(got) != 2 {
		t.Fatalf("tool events = %#v, want start and error", got)
	}
	if got[0].Type != EventToolStart || got[1].Type != EventToolError {
		t.Fatalf("tool event types = %v, %v; want %v, %v", got[0].Type, got[1].Type, EventToolStart, EventToolError)
	}
	if got[1].ToolCall == nil || got[1].ToolCall.ID != "call-event" || !errors.Is(got[1].Err, processorErr) {
		t.Fatalf("tool error event = %#v", got[1])
	}
}

// TestExecuteToolCallsReturnsCanceledWhenToolResultEventCannotBeSent ensures cancellation
// releases execution when no consumer can receive the result event.
func TestExecuteToolCallsReturnsCanceledWhenToolResultEventCannotBeSent(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	release := make(chan struct{})
	tool := observedTestTool{name: "event", call: func(context.Context, ai.ToolCall) (string, error) {
		<-release
		return "ok", nil
	}}
	l := &Loop{Tools: []Tool{tool}}
	iteration := &Iteration{Parts: make([]IterationPart, 1)}
	calls := []pendingToolCall{{
		partIndex: 0,
		call:      ai.ToolCall{ID: "call-event", Type: "function", Name: "event", Args: json.RawMessage(`{}`)},
	}}
	events := make(chan Event)
	done := make(chan error, 1)
	go func() { done <- l.executeToolCalls(ctx, iteration, calls, l.Tools, events, 3, 4, 5) }()

	if event := <-events; event.Type != EventToolStart {
		t.Fatalf("first tool event = %v, want %v", event.Type, EventToolStart)
	}
	close(release)
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("executeToolCalls error = %v, want context cancellation", err)
		}
	case <-time.After(time.Second):
		t.Fatal("executeToolCalls did not return after context cancellation")
	}
}

func requireToolSpans(t testing.TB, recorder *tracetest.SpanRecorder, count int) []sdktrace.ReadOnlySpan {
	t.Helper()
	spans := make([]sdktrace.ReadOnlySpan, 0, count)
	for _, span := range recorder.Ended() {
		if span.Name() == "loop.tool" {
			spans = append(spans, span)
		}
	}
	if len(spans) != count {
		t.Fatalf("tool spans = %d, want %d", len(spans), count)
	}
	return spans
}

func toolSpanText(span sdktrace.ReadOnlySpan) string {
	var text strings.Builder
	text.WriteString(span.Status().Description)
	for _, attr := range span.Attributes() {
		text.WriteString(string(attr.Key))
		text.WriteString(attr.Value.String())
	}
	for _, event := range span.Events() {
		text.WriteString(event.Name)
		for _, attr := range event.Attributes {
			text.WriteString(string(attr.Key))
			text.WriteString(attr.Value.String())
		}
	}
	return text.String()
}

// TestProcessorFailureHasDistinctSafeObservation keeps filter failures separate from handler failures.
func TestProcessorFailureHasDistinctSafeObservation(t *testing.T) {
	recorder := obstest.Install(t)
	var observed gai.Observation
	sink := gai.ObservationSinkFunc(func(_ context.Context, event gai.Observation) {
		if event.Name == "loop_tool_finished" {
			observed = event
		}
	})
	tool, err := NewTool("test", "Test", ai.ToolParameters{}, func(context.Context, ai.ToolCall) (string, error) { return "secret-output", nil })
	if err != nil {
		t.Fatal(err)
	}
	processor := ToolResultProcessorFunc(func(context.Context, ToolPolicyInput, ToolResult) (ToolResult, error) {
		return ToolResult{}, errors.New("secret-processor-error")
	})
	ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{ToolOutput: gai.CaptureEnabled})
	result, _, err := processObservedTool(ctx, ToolPolicyInput{Call: ai.ToolCall{ID: "1", Name: "test", Type: "function", Args: []byte(`{}`)}}, []Tool{tool}, processor, sink)
	if result != nil || !errors.Is(err, ErrToolResultProcess) {
		t.Fatalf("result=%v err=%v", result, err)
	}
	span := requireToolSpans(t, recorder, 1)[0]
	if obstest.Attributes(span)["gai.tool.outcome"].AsString() != toolOutcomeProcessing || observed.Fields["tool_outcome"] != toolOutcomeProcessing {
		t.Fatalf("observation=%v", observed)
	}
	if strings.Contains(toolSpanText(span), "secret-") {
		t.Fatal("raw output/error escaped")
	}
}
