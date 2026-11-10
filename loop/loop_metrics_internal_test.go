package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/internal/obstest"
)

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

func TestRetryMetricBoundsCustomProviderKinds(t *testing.T) {
	reader := obstest.InstallMetrics(t)

	recordRetryMetric(t.Context(), "", "custom-kind-sentinel")
	recordRetryMetric(t.Context(), "", retryReasonAttemptTimeout)

	got := map[string]int64{}
	for _, point := range obstest.SumPoints[int64](t, obstest.CollectMetrics(t, reader), metricLoopRetries) {
		attrs := obstest.AttributeStrings(point.Attributes)
		if _, ok := attrs["gen_ai.request.model"]; ok {
			t.Fatalf("retry attributes = %v, want no empty model", attrs)
		}
		got[attrs["gai.retry.reason"]] += point.Value
	}
	if len(got) != 2 || got["_OTHER"] != 1 || got[retryReasonAttemptTimeout] != 1 {
		t.Fatalf("retry reasons = %v, want _OTHER and attempt_timeout", got)
	}
}

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
		{err: ErrModelNotConfigured, want: "invalid_configuration"},
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
