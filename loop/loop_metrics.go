package loop

import (
	"context"
	"errors"
	"time"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/internal/observe"
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
			"The number of model attempts the loop scheduled for retry."),
		runDuration: observe.Float64Histogram(meter, metricLoopRunDuration, "s",
			"The end-to-end duration of one loop run.", runDurationBuckets),
		runIterations: observe.Int64Histogram(meter, metricLoopRunIterations, "{iteration}",
			"The number of loop iterations started during one run.", runIterationBuckets),
	}
})

// recordToolMetrics records one invoked tool execution. Callers must only pass
// names of registered tools so the attribute stays application-bounded.
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

// recordRetryMetric counts one accepted retry with its bounded reason.
func recordRetryMetric(ctx context.Context, modelName, reason string) {
	if reason != retryReasonAttemptTimeout && !knownProviderErrorKind(ai.ProviderErrorKind(reason)) {
		// ProviderError.Kind is an open string type; keep custom kinds bounded.
		reason = observe.ErrorTypeOther
	}
	attrs := []attribute.KeyValue{attribute.String("gai.retry.reason", reason)}
	if modelName != "" {
		attrs = append(attrs, attribute.String("gen_ai.request.model", modelName))
	}
	loopMetrics.Get().retries.Add(ctx, 1, metric.WithAttributes(attrs...))
}

// recordRunMetrics records the final state of one loop run.
func recordRunMetrics(ctx context.Context, modelName string, err, cancelErr error, iterations int, duration time.Duration) {
	outcome := runOutcomeSuccess
	var errorType string
	switch {
	case cancelErr != nil:
		outcome, errorType = runOutcomeCanceled, runErrorType(cancelErr)
	case err != nil:
		outcome, errorType = runOutcomeError, runErrorType(err)
	}
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
	{ai.ErrUnsuccessfulGeneration, "unsuccessful_generation"},
	{ErrModelNotConfigured, "invalid_configuration"},
	{ErrPromptNotConfigured, "invalid_configuration"},
	{ErrRequiredToolNotConfigured, "invalid_configuration"},
	{ErrToolExecutionConfig, "invalid_configuration"},
	{ai.ErrInvalidToolDefinition, "invalid_configuration"},
}

// runErrorType maps a run failure or cancellation to a bounded error class.
func runErrorType(err error) string {
	for _, candidate := range runErrorTypes {
		if errors.Is(err, candidate.err) {
			return candidate.errorType
		}
	}
	var providerErr *ai.ProviderError
	if errors.As(err, &providerErr) && knownProviderErrorKind(providerErr.Kind) {
		return string(providerErr.Kind)
	}
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	case errors.Is(err, context.Canceled):
		return "canceled"
	}
	return observe.ErrorTypeOther
}

// knownProviderErrorKind reports whether kind is one of GAI's documented
// provider error classes.
func knownProviderErrorKind(kind ai.ProviderErrorKind) bool {
	switch kind {
	case ai.ProviderErrorRateLimited, ai.ProviderErrorTransient, ai.ProviderErrorAuthentication,
		ai.ProviderErrorInvalidRequest, ai.ProviderErrorUnsupported, ai.ProviderErrorUnknown:
		return true
	default:
		return false
	}
}
