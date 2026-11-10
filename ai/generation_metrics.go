package ai

import (
	"context"
	"errors"
	"slices"
	"time"

	"github.com/Samuelk0nrad/gai/internal/observe"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
)

// OpenTelemetry GenAI client inference metrics. See docs/metrics.md.
const (
	metricInferenceDuration     = "gen_ai.client.inference.duration"
	metricInferenceInputTokens  = "gen_ai.client.inference.usage.input_tokens"
	metricInferenceOutputTokens = "gen_ai.client.inference.usage.output_tokens"

	// GAI's normalized Usage does not split tokens by modality; the GenAI
	// conventions require "unknown" when modality cannot be determined.
	tokenModalityUnknown = "unknown"
)

// Bounded error.type values for generation metrics, in addition to the
// ProviderErrorKind values.
const (
	generationErrorTimeout      = "timeout"
	generationErrorCanceled     = "canceled"
	generationErrorUnsuccessful = "unsuccessful_generation"
)

type generationInstruments struct {
	duration     metric.Float64Histogram
	inputTokens  metric.Int64Counter
	outputTokens metric.Int64Counter
}

var generationMetrics = observe.NewInstruments(generationTracerName, func(meter metric.Meter) generationInstruments {
	return generationInstruments{
		duration: observe.Float64Histogram(meter, metricInferenceDuration, "s",
			"GenAI client inference operation duration.", observe.DurationBuckets),
		inputTokens: observe.Int64Counter(meter, metricInferenceInputTokens, "{token}",
			"The number of input (prompt) tokens used, including cached tokens."),
		outputTokens: observe.Int64Counter(meter, metricInferenceOutputTokens, "{token}",
			"The number of output (completion) tokens used, including reasoning tokens."),
	}
})

// recordGenerationMetrics projects one finished generation into low-cardinality
// metrics. It never records request IDs, prompts, completions, or raw errors.
func recordGenerationMetrics(ctx context.Context, config GenerationConfig, result GenerationResult, duration time.Duration) {
	instruments := generationMetrics.Get()
	attrs := []attribute.KeyValue{
		attribute.String("gen_ai.operation.name", "chat"),
		attribute.String("gen_ai.provider.name", semanticProviderName(config.Provider)),
	}
	if config.Model != "" {
		attrs = append(attrs, attribute.String("gen_ai.request.model", config.Model))
	}
	if result.ResponseModel != "" {
		attrs = append(attrs, attribute.String("gen_ai.response.model", result.ResponseModel))
	}

	durationAttrs := attrs
	if result.Err != nil {
		durationAttrs = append(slices.Clip(attrs), attribute.String("error.type", generationErrorType(result.Err)))
	}
	instruments.duration.Record(ctx, duration.Seconds(), metric.WithAttributes(durationAttrs...))

	// Usage counters accept only strictly positive contributions; a nil Usage
	// means the provider did not report usage and records nothing.
	if result.Usage == nil {
		return
	}
	tokenAttrs := metric.WithAttributes(append(slices.Clip(attrs), attribute.String("gen_ai.token.modality", tokenModalityUnknown))...)
	if result.Usage.InputTokens > 0 {
		instruments.inputTokens.Add(ctx, int64(result.Usage.InputTokens), tokenAttrs)
	}
	if result.Usage.OutputTokens > 0 {
		instruments.outputTokens.Add(ctx, int64(result.Usage.OutputTokens), tokenAttrs)
	}
}

// generationErrorType maps a generation failure to a bounded error class.
func generationErrorType(err error) string {
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return generationErrorTimeout
	case errors.Is(err, context.Canceled):
		return generationErrorCanceled
	case errors.Is(err, ErrUnsuccessfulGeneration):
		return generationErrorUnsuccessful
	}
	var providerErr *ProviderError
	if errors.As(err, &providerErr) {
		switch providerErr.Kind {
		case ProviderErrorRateLimited, ProviderErrorTransient, ProviderErrorAuthentication,
			ProviderErrorInvalidRequest, ProviderErrorUnsupported, ProviderErrorUnknown:
			return string(providerErr.Kind)
		}
	}
	return observe.ErrorTypeOther
}
