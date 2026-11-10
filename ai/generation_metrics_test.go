package ai

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/lace-ai/gai/internal/obstest"
)

// generationMetricAttributes is the complete set of attribute keys generation
// metrics may emit. Anything else risks cardinality or content leaks.
var generationMetricAttributes = []string{
	"gen_ai.operation.name",
	"gen_ai.provider.name",
	"gen_ai.request.model",
	"gen_ai.response.model",
	"gen_ai.token.modality",
	"error.type",
}

func finishGeneration(config GenerationConfig, result GenerationResult) {
	_, observation := StartGenerationObservation(context.Background(), AIRequest{
		Messages: []Message{TextMessage(RoleUser, "prompt-sentinel-secret")},
	}, config)
	observation.Finish(result)
}

func requireSinglePointAttributes(t *testing.T, name string, got map[string]string, want map[string]string) {
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

func TestGenerationMetricsRecordDurationAndReportedUsage(t *testing.T) {
	reader := obstest.InstallMetrics(t)

	finishGeneration(GenerationConfig{Provider: "gemini", Model: "gemini-test", Streaming: true}, GenerationResult{
		ResponseModel: "gemini-test-001",
		RequestID:     "request-id-sentinel",
		FinishReason:  "STOP",
		Usage:         &Usage{InputTokens: 12, OutputTokens: 5, ReasoningTokens: 2, CachedTokens: 4},
	})

	metrics := obstest.CollectMetrics(t, reader)
	base := map[string]string{
		"gen_ai.operation.name": "chat",
		"gen_ai.provider.name":  "gcp.gemini",
		"gen_ai.request.model":  "gemini-test",
		"gen_ai.response.model": "gemini-test-001",
	}

	durations := obstest.HistogramPoints[float64](t, metrics, metricInferenceDuration)
	if len(durations) != 1 || durations[0].Count != 1 {
		t.Fatalf("duration points = %#v, want one recorded generation", durations)
	}
	requireSinglePointAttributes(t, metricInferenceDuration, obstest.AttributeStrings(durations[0].Attributes), base)
	duration, _ := obstest.FindMetric(metrics, metricInferenceDuration)
	if duration.Unit != "s" {
		t.Fatalf("duration unit = %q, want s", duration.Unit)
	}

	tokenAttrs := map[string]string{"gen_ai.token.modality": "unknown"}
	for key, value := range base {
		tokenAttrs[key] = value
	}
	for name, want := range map[string]int64{metricInferenceInputTokens: 12, metricInferenceOutputTokens: 5} {
		points := obstest.SumPoints[int64](t, metrics, name)
		if len(points) != 1 || points[0].Value != want {
			t.Fatalf("%s points = %#v, want one point with %d", name, points, want)
		}
		requireSinglePointAttributes(t, name, obstest.AttributeStrings(points[0].Attributes), tokenAttrs)
		if metric, _ := obstest.FindMetric(metrics, name); metric.Unit != "{token}" {
			t.Fatalf("%s unit = %q, want {token}", name, metric.Unit)
		}
	}
	obstest.RequireMetricAttributesAllowed(t, metrics, generationMetricAttributes, "request-id-sentinel", "prompt-sentinel-secret")
}

func TestGenerationMetricsSkipMissingAndZeroUsage(t *testing.T) {
	for _, tt := range []struct {
		name  string
		usage *Usage
	}{
		{name: "missing", usage: nil},
		{name: "reported_zero", usage: &Usage{}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			reader := obstest.InstallMetrics(t)
			finishGeneration(GenerationConfig{Provider: "openai", Model: "gpt-test"}, GenerationResult{Usage: tt.usage})

			metrics := obstest.CollectMetrics(t, reader)
			if points := obstest.HistogramPoints[float64](t, metrics, metricInferenceDuration); len(points) != 1 {
				t.Fatalf("duration points = %d, want 1", len(points))
			}
			for _, name := range []string{metricInferenceInputTokens, metricInferenceOutputTokens} {
				if points := obstest.SumPoints[int64](t, metrics, name); len(points) != 0 {
					t.Fatalf("%s points = %#v, want none", name, points)
				}
			}
		})
	}
}

func TestGenerationMetricsRecordBoundedErrorType(t *testing.T) {
	reader := obstest.InstallMetrics(t)

	finishGeneration(GenerationConfig{Provider: "anthropic", Model: "claude-test"}, GenerationResult{
		Err:   fmt.Errorf("stream failed: %w", &ProviderError{Kind: ProviderErrorRateLimited, Err: errors.New("raw-provider-error-sentinel")}),
		Usage: &Usage{InputTokens: 7},
	})

	metrics := obstest.CollectMetrics(t, reader)
	durations := obstest.HistogramPoints[float64](t, metrics, metricInferenceDuration)
	if len(durations) != 1 {
		t.Fatalf("duration points = %d, want 1", len(durations))
	}
	requireSinglePointAttributes(t, metricInferenceDuration, obstest.AttributeStrings(durations[0].Attributes), map[string]string{
		"gen_ai.operation.name": "chat",
		"gen_ai.provider.name":  "anthropic",
		"gen_ai.request.model":  "claude-test",
		"error.type":            "rate_limited",
	})
	// Failed operations still report consumed usage, without error.type.
	inputs := obstest.SumPoints[int64](t, metrics, metricInferenceInputTokens)
	if len(inputs) != 1 || inputs[0].Value != 7 {
		t.Fatalf("input token points = %#v, want 7", inputs)
	}
	if _, ok := obstest.AttributeStrings(inputs[0].Attributes)["error.type"]; ok {
		t.Fatal("token usage must not carry error.type")
	}
	obstest.RequireMetricAttributesAllowed(t, metrics, generationMetricAttributes, "raw-provider-error-sentinel")
}

func TestGenerationErrorType(t *testing.T) {
	for _, tt := range []struct {
		err  error
		want string
	}{
		{err: &ProviderError{Kind: ProviderErrorTransient}, want: "transient"},
		{err: &ProviderError{Kind: ProviderErrorUnknown}, want: "unknown"},
		{err: &ProviderError{Kind: "custom-kind"}, want: "_OTHER"},
		{err: fmt.Errorf("wrapped: %w", context.DeadlineExceeded), want: "timeout"},
		{err: context.Canceled, want: "canceled"},
		{err: &TerminalError{Provider: "openai", Reason: "length"}, want: "unsuccessful_generation"},
		{err: errors.New("unclassified-sentinel"), want: "_OTHER"},
	} {
		if got := generationErrorType(tt.err); got != tt.want {
			t.Errorf("generationErrorType(%v) = %q, want %q", tt.err, got, tt.want)
		}
	}
}

func TestGenerationMetricsRecordOncePerGeneration(t *testing.T) {
	reader := obstest.InstallMetrics(t)

	_, observation := StartGenerationObservation(context.Background(), AIRequest{}, GenerationConfig{Provider: "mistral", Model: "mistral-test", Streaming: true})
	observation.ObserveToken(Token{Completion: &Completion{Model: "mistral-test-2", UsageReported: true, Usage: Usage{InputTokens: 3, OutputTokens: 2}}})
	observation.Finish(GenerationResult{})
	observation.Finish(GenerationResult{Usage: &Usage{InputTokens: 100, OutputTokens: 100}})

	metrics := obstest.CollectMetrics(t, reader)
	durations := obstest.HistogramPoints[float64](t, metrics, metricInferenceDuration)
	if len(durations) != 1 || durations[0].Count != 1 {
		t.Fatalf("duration points = %#v, want one generation", durations)
	}
	if got := obstest.AttributeStrings(durations[0].Attributes)["gen_ai.response.model"]; got != "mistral-test-2" {
		t.Fatalf("response model = %q, want streamed completion model", got)
	}
	for name, want := range map[string]int64{metricInferenceInputTokens: 3, metricInferenceOutputTokens: 2} {
		points := obstest.SumPoints[int64](t, metrics, name)
		if len(points) != 1 || points[0].Value != want {
			t.Fatalf("%s points = %#v, want streamed usage %d recorded once", name, points, want)
		}
	}
}
