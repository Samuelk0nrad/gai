package loop

import (
	"context"
	"errors"
	"time"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/observe"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
)

// Loop metrics. Tool duration follows the OpenTelemetry GenAI execute_tool
// convention; retry and run metrics are GAI-specific. See docs/metrics.md.
const (
	metricToolDuration      = "gen_ai.execute_tool.duration"
	metricLoopRetries       = "gai.loop.retries"
	metricLoopRunDuration   = "gai.loop.run.duration"
	metricLoopRunIterations = "gai.loop.run.iterations"

	runOutcomeSuccess  = "success"
	runOutcomeError    = "error"
	runOutcomeCanceled = "canceled"
	runOutcomePaused   = "paused"

	// runErrorInvalidConfiguration classifies runs rejected by Loop.Validate
	// and configuration errors found while preparing a run.
	runErrorInvalidConfiguration = "invalid_configuration"
)

var (
	// runDurationBuckets match the GenAI invoke_agent duration advice, which
	// covers multi-step runs lasting from sub-second to hours.
	runDurationBuckets = []float64{0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8, 409.6, 900, 1800, 3600, 7200, 14400, 43200, 86400}
	// runIterationBuckets match the GenAI invoke_agent inference-call advice.
	runIterationBuckets = []float64{1, 2, 4, 8, 16, 32, 64, 128}
)

type loopInstruments struct {
	toolDuration  metric.Float64Histogram
	retries       metric.Int64Counter
	runDuration   metric.Float64Histogram
	runIterations metric.Int64Histogram
}

var loopMetrics = observe.NewInstruments(loopTracerName, func(meter metric.Meter) loopInstruments {
	return loopInstruments{
		toolDuration: observe.Float64Histogram(meter, metricToolDuration, "s",
			"The duration of a single tool execution.", observe.DurationBuckets),
		retries: observe.Int64Counter(meter, metricLoopRetries, "{retry}",
			"The number of model retries the loop started."),
		runDuration: observe.Float64Histogram(meter, metricLoopRunDuration, "s",
			"The end-to-end duration of one loop run.", runDurationBuckets),
		runIterations: observe.Int64Histogram(meter, metricLoopRunIterations, "{iteration}",
			"The number of loop iterations started during one run.", runIterationBuckets),
	}
})

// recordToolMetrics records one invoked tool execution. outcome classifies the
// tool function's own result, before any ToolResultProcessor. Callers must
// only pass names of registered tools so the attribute stays bounded.
func recordToolMetrics(ctx context.Context, toolName, outcome string, failed bool, duration time.Duration) {
	attrs := []attribute.KeyValue{
		attribute.String("gen_ai.tool.name", toolName),
		attribute.String("gen_ai.tool.type", "function"),
	}
	if failed {
		attrs = append(attrs, attribute.String("error.type", "gai.tool."+outcome))
	}
	loopMetrics.Get().toolDuration.Record(ctx, duration.Seconds(), metric.WithAttributes(attrs...))
}

// recordRetryMetric counts one retry whose next attempt is about to start.
// The reason uses the same classes as model call error.type, plus
// attempt_timeout.
func recordRetryMetric(ctx context.Context, modelName string, retryErr error) {
	reason := retryReasonAttemptTimeout
	if !errors.Is(retryErr, ErrAttemptTimeout) {
		reason = ai.TelemetryErrorType(retryErr)
	}
	if reason == "" {
		reason = observe.ErrorTypeOther
	}
	attrs := []attribute.KeyValue{attribute.String("gai.retry.reason", reason)}
	if modelName != "" {
		attrs = append(attrs, attribute.String("gen_ai.request.model", modelName))
	}
	loopMetrics.Get().retries.Add(ctx, 1, metric.WithAttributes(attrs...))
}

// runOutcome returns the outcome and bounded error class of a finished run.
// Cancellation takes precedence over a recorded error, and both take
// precedence over a pause.
func runOutcome(err, cancelErr error, paused bool) (outcome, errorType string) {
	switch {
	case cancelErr != nil:
		return runOutcomeCanceled, runErrorType(cancelErr)
	case err != nil:
		return runOutcomeError, runErrorType(err)
	case paused:
		return runOutcomePaused, ""
	default:
		return runOutcomeSuccess, ""
	}
}

// recordRunMetrics records the final state of one loop run.
func recordRunMetrics(ctx context.Context, modelName, outcome, errorType string, iterations int, duration time.Duration) {
	attrs := []attribute.KeyValue{attribute.String("gai.loop.outcome", outcome)}
	if modelName != "" {
		attrs = append(attrs, attribute.String("gen_ai.request.model", modelName))
	}
	if errorType != "" {
		attrs = append(attrs, attribute.String("error.type", errorType))
	}
	instruments := loopMetrics.Get()
	options := metric.WithAttributes(attrs...)
	instruments.runDuration.Record(ctx, duration.Seconds(), options)
	instruments.runIterations.Record(ctx, int64(iterations), options)
}

// runErrorTypes lists bounded run error classes in match order. Wrapping
// errors such as ErrMaxRetries come before the causes they wrap.
var runErrorTypes = []struct {
	err       error
	errorType string
}{
	{ErrMaxRetries, "max_retries"},
	{ErrMaxIterations, "max_iterations"},
	{ErrAttemptTimeout, retryReasonAttemptTimeout},
	{ErrToolPanic, "tool_panic"},
	{ErrToolResultProcess, "tool_result_processing"},
	{ErrToolPolicy, "tool_policy"},
	{ErrToolApproval, "tool_approval"},
	{ErrMixedTerminalToolBatch, "terminal_tool_batch"},
	{ErrTerminalToolBatch, "terminal_tool_batch"},
	{ErrBuildPrompt, "build_prompt"},
	{ai.ErrRequestBudgetExceeded, "request_budget_exceeded"},
	{ErrRequiredToolNotConfigured, runErrorInvalidConfiguration},
	{ErrToolExecutionConfig, runErrorInvalidConfiguration},
	{ai.ErrInvalidToolDefinition, runErrorInvalidConfiguration},
}

// runErrorType maps a run failure or cancellation to a bounded error class.
// Errors outside the loop's own classes use the model call classification,
// so run, retry, and inference metrics agree on shared classes.
func runErrorType(err error) string {
	for _, candidate := range runErrorTypes {
		if errors.Is(err, candidate.err) {
			return candidate.errorType
		}
	}
	return ai.TelemetryErrorType(err)
}

// recordInvalidRunMetrics records a run that Loop.Validate rejected before it
// started.
func recordInvalidRunMetrics(ctx context.Context, l *Loop, started time.Time) {
	if ctx == nil {
		ctx = context.Background()
	}
	modelName := ""
	if l != nil && !nilImplementation(l.Model) {
		modelName = ai.ModelName(l.Model)
	}
	recordRunMetrics(ctx, modelName, runOutcomeError, runErrorInvalidConfiguration, 0, time.Since(started))
}
