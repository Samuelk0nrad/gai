package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/obstest"
)

// TestToolMetricsRecordOnlyInvokedExecutions checks that unknown tools and
// malformed calls, which never run, are not recorded.
func TestToolMetricsRecordOnlyInvokedExecutions(t *testing.T) {
	reader := obstest.InstallMetrics(t)
	registered := observedTestTool{name: "test", call: func(context.Context, ai.ToolCall) (string, error) {
		return "ok", nil
	}}

	callObservedTool(t.Context(), ai.ToolCall{ID: "call-1", Type: "function", Name: "unknown-tool-sentinel", Args: json.RawMessage(`{}`)}, []Tool{registered})
	callObservedTool(t.Context(), ai.ToolCall{ID: "call-2", Type: "function", Name: "test", Args: json.RawMessage(`{`)}, []Tool{registered})
	if points := obstest.HistogramPoints[float64](t, obstest.CollectMetrics(t, reader), metricToolDuration); len(points) != 0 {
		t.Fatalf("tool duration points = %#v, want none for uninvoked calls", points)
	}

	callObservedTool(t.Context(), ai.ToolCall{ID: "call-3", Type: "function", Name: "test", Args: json.RawMessage(`{}`)}, []Tool{registered})
	points := obstest.HistogramPoints[float64](t, obstest.CollectMetrics(t, reader), metricToolDuration)
	if len(points) != 1 || points[0].Count != 1 {
		t.Fatalf("tool duration points = %#v, want one invoked execution", points)
	}
	if got := obstest.AttributeStrings(points[0].Attributes)["gen_ai.tool.name"]; got != "test" {
		t.Fatalf("tool name = %q, want test", got)
	}
}

// TestToolMetricsClassifyPanics checks that a panicking tool is recorded with
// a fixed error.type and without its panic value.
func TestToolMetricsClassifyPanics(t *testing.T) {
	reader := obstest.InstallMetrics(t)
	tool := observedTestTool{name: "test", call: func(context.Context, ai.ToolCall) (string, error) {
		panic("panic-value-sentinel")
	}}

	func() {
		defer func() { _ = recover() }()
		callObservedTool(t.Context(), ai.ToolCall{ID: "call-1", Type: "function", Name: "test", Args: json.RawMessage(`{}`)}, []Tool{tool})
	}()

	metrics := obstest.CollectMetrics(t, reader)
	points := obstest.HistogramPoints[float64](t, metrics, metricToolDuration)
	if len(points) != 1 {
		t.Fatalf("tool duration points = %#v, want one panicked execution", points)
	}
	if got := obstest.AttributeStrings(points[0].Attributes)["error.type"]; got != "gai.tool.panic" {
		t.Fatalf("error.type = %q, want gai.tool.panic", got)
	}
	obstest.RequireMetricAttributesAllowed(t, metrics, []string{"gen_ai.tool.name", "gen_ai.tool.type", "error.type"}, "panic-value-sentinel")
}

// TestRetryMetricUsesModelCallErrorClasses checks that retry reasons use the
// model call error.type classes: custom provider kinds and plain errors are
// _OTHER, and context errors win over the provider kind.
func TestRetryMetricUsesModelCallErrorClasses(t *testing.T) {
	reader := obstest.InstallMetrics(t)

	recordRetryMetric(t.Context(), "", &ai.ProviderError{Kind: "custom-kind-sentinel"})
	recordRetryMetric(t.Context(), "", errors.New("plain-error-sentinel"))
	recordRetryMetric(t.Context(), "", ErrAttemptTimeout)
	recordRetryMetric(t.Context(), "", &ai.ProviderError{Kind: ai.ProviderErrorUnknown, Err: context.DeadlineExceeded})

	got := map[string]int64{}
	for _, point := range obstest.SumPoints[int64](t, obstest.CollectMetrics(t, reader), metricLoopRetries) {
		attrs := obstest.AttributeStrings(point.Attributes)
		if _, ok := attrs["gen_ai.request.model"]; ok {
			t.Fatalf("retry attributes = %v, want no empty model", attrs)
		}
		got[attrs["gai.retry.reason"]] += point.Value
	}
	if len(got) != 3 || got["_OTHER"] != 2 || got[retryReasonAttemptTimeout] != 1 || got["timeout"] != 1 {
		t.Fatalf("retry reasons = %v, want two _OTHER, one attempt_timeout, and one timeout", got)
	}
}

// TestRunErrorType checks the bounded run error classes and their match order.
func TestRunErrorType(t *testing.T) {
	for _, tt := range []struct {
		err  error
		want string
	}{
		{err: fmt.Errorf("%w: limit=1: %w", ErrMaxRetries, ErrAttemptTimeout), want: "max_retries"},
		{err: fmt.Errorf("%w: limit=1: %w", ErrMaxRetries, &ai.ProviderError{Kind: ai.ProviderErrorRateLimited}), want: "max_retries"},
		{err: fmt.Errorf("%w: limit=8", ErrMaxIterations), want: "max_iterations"},
		{err: &ai.ProviderError{Kind: ai.ProviderErrorAuthentication}, want: "authentication"},
		{err: &ai.ProviderError{Kind: "custom-kind"}, want: "_OTHER"},
		{err: fmt.Errorf("%w: resolver: %w", ErrToolApproval, context.Canceled), want: "tool_approval"},
		{err: ErrMixedTerminalToolBatch, want: "terminal_tool_batch"},
		{err: fmt.Errorf("%w: budget", ai.ErrRequestBudgetExceeded), want: "request_budget_exceeded"},
		{err: fmt.Errorf("%w: %q", ErrRequiredToolNotConfigured, "missing"), want: "invalid_configuration"},
		// Shared classes match model call error.type.
		{err: &ai.ProviderError{Kind: ai.ProviderErrorUnknown, Err: context.DeadlineExceeded}, want: "timeout"},
		{err: &ai.TerminalError{Provider: "openai", Reason: "length"}, want: "unsuccessful_generation"},
		{err: context.DeadlineExceeded, want: "timeout"},
		{err: context.Canceled, want: "canceled"},
		{err: errors.New("unclassified-sentinel"), want: "_OTHER"},
	} {
		if got := runErrorType(tt.err); got != tt.want {
			t.Errorf("runErrorType(%v) = %q, want %q", tt.err, got, tt.want)
		}
	}
}

// TestToolMetricsProcessorPanicExcludesProcessingTime checks that a panicking
// result processor does not add its own runtime to the tool execution duration.
func TestToolMetricsProcessorPanicExcludesProcessingTime(t *testing.T) {
	const toolTime, processingTime = 20 * time.Millisecond, 250 * time.Millisecond
	reader := obstest.InstallMetrics(t)
	tool := observedTestTool{name: "test", call: func(context.Context, ai.ToolCall) (string, error) {
		time.Sleep(toolTime)
		return "ok", nil
	}}
	processor := ToolResultProcessorFunc(func(context.Context, ToolPolicyInput, ToolResult) (ToolResult, error) {
		time.Sleep(processingTime)
		panic("processor-panic-sentinel")
	})

	func() {
		defer func() { _ = recover() }()
		_, _, _ = processObservedTool(t.Context(), ToolPolicyInput{Call: ai.ToolCall{ID: "call-1", Type: "function", Name: "test", Args: json.RawMessage(`{}`)}}, []Tool{tool}, processor)
	}()

	points := obstest.HistogramPoints[float64](t, obstest.CollectMetrics(t, reader), metricToolDuration)
	if len(points) != 1 || points[0].Count != 1 {
		t.Fatalf("tool duration points = %#v, want one execution", points)
	}
	if got := points[0].Sum; got < toolTime.Seconds() || got >= processingTime.Seconds() {
		t.Fatalf("tool duration = %gs, want at least %gs and less than the %gs processing delay", got, toolTime.Seconds(), processingTime.Seconds())
	}
}

// TestToolMetricsRecordExecutionDurationInSeconds checks the recorded tool
// duration value with a deterministic lower bound and a seconds-unit ceiling.
func TestToolMetricsRecordExecutionDurationInSeconds(t *testing.T) {
	const toolTime = 20 * time.Millisecond
	reader := obstest.InstallMetrics(t)
	tool := observedTestTool{name: "test", call: func(context.Context, ai.ToolCall) (string, error) {
		time.Sleep(toolTime)
		return "ok", nil
	}}

	callObservedTool(t.Context(), ai.ToolCall{ID: "call-1", Type: "function", Name: "test", Args: json.RawMessage(`{}`)}, []Tool{tool})

	points := obstest.HistogramPoints[float64](t, obstest.CollectMetrics(t, reader), metricToolDuration)
	if len(points) != 1 || points[0].Count != 1 {
		t.Fatalf("tool duration points = %#v, want one execution", points)
	}
	// Sleep guarantees the lower bound; a millisecond unit would exceed 5.
	if got := points[0].Sum; got < toolTime.Seconds() || got >= 5 {
		t.Fatalf("tool duration = %g, want at least %gs recorded in seconds", got, toolTime.Seconds())
	}
}

// TestToolMetricsClassifyToolResultBeforeProcessing checks that the tool
// metric's error.type reflects the tool's own result, not the result a
// ToolResultProcessor rewrote or rejected.
func TestToolMetricsClassifyToolResultBeforeProcessing(t *testing.T) {
	succeed := func(context.Context, ai.ToolCall) (string, error) { return "ok", nil }
	fail := func(context.Context, ai.ToolCall) (string, error) { return "", errors.New("tool-error-sentinel") }
	for _, tt := range []struct {
		name      string
		call      func(context.Context, ai.ToolCall) (string, error)
		processor ToolResultProcessor
		wantError string
	}{
		{
			name: "processor hides tool error",
			call: fail,
			processor: ToolResultProcessorFunc(func(context.Context, ToolPolicyInput, ToolResult) (ToolResult, error) {
				return ToolResult{Text: "recovered"}, nil
			}),
			wantError: "gai.tool.tool_error",
		},
		{
			name: "processor turns success into error",
			call: succeed,
			processor: ToolResultProcessorFunc(func(context.Context, ToolPolicyInput, ToolResult) (ToolResult, error) {
				return ToolResult{Err: errors.New("processed-error-sentinel")}, nil
			}),
		},
		{
			name: "processor rejects success",
			call: succeed,
			processor: ToolResultProcessorFunc(func(context.Context, ToolPolicyInput, ToolResult) (ToolResult, error) {
				return ToolResult{}, errors.New("rejected-sentinel")
			}),
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			reader := obstest.InstallMetrics(t)
			tool := observedTestTool{name: "test", call: tt.call}

			_, _, _ = processObservedTool(t.Context(), ToolPolicyInput{Call: ai.ToolCall{ID: "call-1", Type: "function", Name: "test", Args: json.RawMessage(`{}`)}}, []Tool{tool}, tt.processor)

			points := obstest.HistogramPoints[float64](t, obstest.CollectMetrics(t, reader), metricToolDuration)
			if len(points) != 1 || points[0].Count != 1 {
				t.Fatalf("tool duration points = %#v, want one execution", points)
			}
			if got := obstest.AttributeStrings(points[0].Attributes)["error.type"]; got != tt.wantError {
				t.Fatalf("error.type = %q, want %q", got, tt.wantError)
			}
		})
	}
}
