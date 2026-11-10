package loop_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/internal/obstest"
	"github.com/lace-ai/gai/loop"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
)

// Metric tests install a global MeterProvider and therefore must not call
// t.Parallel.

const (
	metricToolDuration      = "gen_ai.execute_tool.duration"
	metricLoopRetries       = "gai.loop.retries"
	metricLoopRunDuration   = "gai.loop.run.duration"
	metricLoopRunIterations = "gai.loop.run.iterations"
)

// loopMetricAttributes is the complete set of attribute keys loop metrics may
// emit. Anything else risks cardinality or content leaks.
var loopMetricAttributes = []string{
	"gen_ai.tool.name",
	"gen_ai.tool.type",
	"gen_ai.request.model",
	"gai.retry.reason",
	"gai.loop.outcome",
	"error.type",
}

// toolCallToken returns a streamed native tool-call token.
func toolCallToken(id, name, args string) ai.Token {
	return ai.Token{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{
		ID: id, Type: "function", Name: name, Args: json.RawMessage(args),
	}}}
}

// requireAttributes fails unless got contains exactly the want attribute set.
func requireAttributes(t *testing.T, name string, got, want map[string]string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s attributes = %v, want %v", name, got, want)
	}
	for key, value := range want {
		if got[key] != value {
			t.Fatalf("%s attributes = %v, want %v", name, got, want)
		}
	}
}

// requireRunMetrics checks that exactly one run was recorded with the given
// iteration count and attributes.
func requireRunMetrics(t *testing.T, metrics []metricdata.Metrics, iterations int64, attrs map[string]string) {
	t.Helper()
	durations := obstest.HistogramPoints[float64](t, metrics, metricLoopRunDuration)
	if len(durations) != 1 || durations[0].Count != 1 {
		t.Fatalf("run duration points = %#v, want one run", durations)
	}
	requireAttributes(t, metricLoopRunDuration, obstest.AttributeStrings(durations[0].Attributes), attrs)
	counts := obstest.HistogramPoints[int64](t, metrics, metricLoopRunIterations)
	if len(counts) != 1 || counts[0].Count != 1 || counts[0].Sum != iterations {
		t.Fatalf("run iteration points = %#v, want one run with %d iterations", counts, iterations)
	}
	requireAttributes(t, metricLoopRunIterations, obstest.AttributeStrings(counts[0].Attributes), attrs)
}

// TestLoopMetricsRecordRetryToolAndSuccessfulRun checks retry, tool, and run
// metrics for a run that retries, calls a tool, and succeeds.
func TestLoopMetricsRecordRetryToolAndSuccessfulRun(t *testing.T) {
	reader := obstest.InstallMetrics(t)
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("provider-error-sentinel")}}},
		{toolCallToken("call-id-sentinel", "echo", `{"text":"tool-args-sentinel"}`)},
		{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "done"}}},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}

	metrics := obstest.CollectMetrics(t, reader)
	retries := obstest.SumPoints[int64](t, metrics, metricLoopRetries)
	if len(retries) != 1 || retries[0].Value != 1 {
		t.Fatalf("retry points = %#v, want one retry", retries)
	}
	requireAttributes(t, metricLoopRetries, obstest.AttributeStrings(retries[0].Attributes), map[string]string{
		"gai.retry.reason":     "transient",
		"gen_ai.request.model": "scripted-stream-model",
	})

	tools := obstest.HistogramPoints[float64](t, metrics, metricToolDuration)
	if len(tools) != 1 || tools[0].Count != 1 {
		t.Fatalf("tool duration points = %#v, want one tool execution", tools)
	}
	requireAttributes(t, metricToolDuration, obstest.AttributeStrings(tools[0].Attributes), map[string]string{
		"gen_ai.tool.name": "echo",
		"gen_ai.tool.type": "function",
	})

	requireRunMetrics(t, metrics, 2, map[string]string{
		"gai.loop.outcome":     "success",
		"gen_ai.request.model": "scripted-stream-model",
	})
	obstest.RequireMetricAttributesAllowed(t, metrics, loopMetricAttributes,
		"provider-error-sentinel", "call-id-sentinel", "tool-args-sentinel", "Initial prompt")
}

// TestLoopMetricsRecordFailedToolsWithoutModelSuppliedNames checks failed tool
// classification and that hallucinated tool names are never recorded.
func TestLoopMetricsRecordFailedToolsWithoutModelSuppliedNames(t *testing.T) {
	reader := obstest.InstallMetrics(t)
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{
			toolCallToken("call-failure", "failure", `{"text":"payload"}`),
			toolCallToken("call-unknown", "hallucinated-tool-sentinel", `{}`),
		},
		{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "done"}}},
	}}
	l := loop.New(model, []loop.Tool{sentinelErrorTool{}}, testPromptBuilder(), nil)

	collectLoopEvents(t, l, context.Background())

	metrics := obstest.CollectMetrics(t, reader)
	tools := obstest.HistogramPoints[float64](t, metrics, metricToolDuration)
	if len(tools) != 1 || tools[0].Count != 1 {
		t.Fatalf("tool duration points = %#v, want only the invoked registered tool", tools)
	}
	requireAttributes(t, metricToolDuration, obstest.AttributeStrings(tools[0].Attributes), map[string]string{
		"gen_ai.tool.name": "failure",
		"gen_ai.tool.type": "function",
		"error.type":       "gai.tool.tool_error",
	})
	obstest.RequireMetricAttributesAllowed(t, metrics, loopMetricAttributes,
		"hallucinated-tool-sentinel", "tool-output-sentinel-secret")
}

// TestLoopMetricsRecordFailedAndCanceledRuns checks run outcomes and error
// classes for iteration limits, exhausted retries, and cancellation.
func TestLoopMetricsRecordFailedAndCanceledRuns(t *testing.T) {
	t.Run("max_iterations", func(t *testing.T) {
		reader := obstest.InstallMetrics(t)
		model := &scriptedStreamModel{sequences: [][]ai.Token{
			{toolCallToken("call-1", "echo", `{"text":"payload"}`)},
		}}
		l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
		l.MaxLoopIterations = 1

		if err := loopError(collectLoopEvents(t, l, context.Background())); !errors.Is(err, loop.ErrMaxIterations) {
			t.Fatalf("loop error = %v, want ErrMaxIterations", err)
		}
		requireRunMetrics(t, obstest.CollectMetrics(t, reader), 1, map[string]string{
			"gai.loop.outcome":     "error",
			"gen_ai.request.model": "scripted-stream-model",
			"error.type":           "max_iterations",
		})
	})

	t.Run("retries_exhausted", func(t *testing.T) {
		reader := obstest.InstallMetrics(t)
		failure := ai.Token{Err: &ai.ProviderError{Kind: ai.ProviderErrorRateLimited, Err: errors.New("limited")}}
		model := &scriptedStreamModel{sequences: [][]ai.Token{{failure}, {failure}}}
		l := loop.New(model, nil, testPromptBuilder(), nil)
		l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}

		if err := loopError(collectLoopEvents(t, l, context.Background())); !errors.Is(err, loop.ErrMaxRetries) {
			t.Fatalf("loop error = %v, want ErrMaxRetries", err)
		}
		metrics := obstest.CollectMetrics(t, reader)
		if retries := obstest.SumPoints[int64](t, metrics, metricLoopRetries); len(retries) != 1 || retries[0].Value != 1 {
			t.Fatalf("retry points = %#v, want one accepted retry", retries)
		}
		requireRunMetrics(t, metrics, 1, map[string]string{
			"gai.loop.outcome":     "error",
			"gen_ai.request.model": "scripted-stream-model",
			"error.type":           "max_retries",
		})
	})

	t.Run("canceled", func(t *testing.T) {
		reader := obstest.InstallMetrics(t)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		l := loop.New(&scriptedStreamModel{}, nil, testPromptBuilder(), nil)

		collectLoopEvents(t, l, ctx)
		requireRunMetrics(t, obstest.CollectMetrics(t, reader), 0, map[string]string{
			"gai.loop.outcome":     "canceled",
			"gen_ai.request.model": "scripted-stream-model",
			"error.type":           "canceled",
		})
	})
}

// TestLoopMetricsRecordRunDurationInSeconds checks the recorded run duration
// value with a deterministic lower bound and a seconds-unit ceiling.
func TestLoopMetricsRecordRunDurationInSeconds(t *testing.T) {
	const modelTime = 20 * time.Millisecond
	reader := obstest.InstallMetrics(t)
	model := &scriptedStreamModel{
		sequences: [][]ai.Token{{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "done"}}}},
		delays:    []time.Duration{modelTime},
	}
	l := loop.New(model, nil, testPromptBuilder(), nil)

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}

	durations := obstest.HistogramPoints[float64](t, obstest.CollectMetrics(t, reader), metricLoopRunDuration)
	if len(durations) != 1 || durations[0].Count != 1 {
		t.Fatalf("run duration points = %#v, want one run", durations)
	}
	// The model delay guarantees the lower bound; a millisecond unit would exceed 5.
	if got := durations[0].Sum; got < modelTime.Seconds() || got >= 5 {
		t.Fatalf("run duration = %g, want at least %gs recorded in seconds", got, modelTime.Seconds())
	}
}
