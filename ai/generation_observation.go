package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/lace-ai/gai"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

const generationTracerName = "github.com/lace-ai/gai/ai"

// GenerationConfig identifies one provider model invocation.
type GenerationConfig struct {
	Provider  string
	Model     string
	Streaming bool
	Sink      gai.ObservationSink
}

// GenerationResult contains provider metadata known when a generation ends.
// Usage must be nil when the provider did not report usage. A non-nil Usage
// records reported zero values as zero rather than treating them as missing.
type GenerationResult struct {
	ResponseModel string
	RequestID     string
	FinishReason  string
	Usage         *Usage
	ToolCallCount int
	HTTPStatus    int
	Err           error
}

// GenerationObservation owns the one semantic generation span associated with
// one provider invocation. It is safe to finish more than once; only the first
// call ends the span.
type GenerationObservation struct {
	ctx        context.Context
	config     GenerationConfig
	span       trace.Span
	startedAt  time.Time
	firstOnce  sync.Once
	finishOnce sync.Once
	mu         sync.Mutex
	stream     GenerationResult
}

// StartGenerationObservation starts a provider-neutral generation span. Call
// it immediately before invoking the provider, after local validation and
// request mapping have succeeded.
func StartGenerationObservation(ctx context.Context, req AIRequest, config GenerationConfig) (context.Context, *GenerationObservation) {
	attrs := []attribute.KeyValue{
		attribute.String("langfuse.observation.type", "generation"),
		attribute.String("langfuse.observation.model.name", config.Model),
		attribute.String("gen_ai.provider.name", semanticProviderName(config.Provider)),
		attribute.String("gen_ai.request.model", config.Model),
		attribute.Bool("gai.gen_ai.streaming", config.Streaming),
		attribute.String("ai.provider", config.Provider),
		attribute.String("ai.model", config.Model),
		attribute.Int("ai.max_tokens", req.MaxTokens),
		attribute.Int("ai.prompt_length", generationPromptLength(req)),
	}
	if req.MaxTokens > 0 {
		attrs = append(attrs, attribute.Int("gen_ai.request.max_tokens", req.MaxTokens))
	}
	if parameters := generationModelParameters(req); parameters != "" {
		attrs = append(attrs, attribute.String("langfuse.observation.model.parameters", parameters))
	}
	ctx, span := gai.StartClientOperationSpan(ctx, generationTracerName, "chat "+config.Model, "gen_ai.operation.name", "chat", attrs...)
	startedAt := time.Now()
	return ctx, &GenerationObservation{ctx: ctx, config: config, span: span, startedAt: startedAt}
}

// generationPromptLength counts selected system/user text without retaining
// opaque provider state or tool payloads in observations.
func generationPromptLength(req AIRequest) int {
	length := 0
	for _, message := range req.Messages {
		if message.Role == RoleSystem || message.Role == RoleUser {
			length += len(message.Text())
		}
	}
	return length
}

func semanticProviderName(provider string) string {
	switch provider {
	case "gemini":
		return "gcp.gemini"
	case "mistral":
		return "mistral_ai"
	default:
		return provider
	}
}

// FirstOutput records when the first meaningful response text, reasoning, or
// tool call arrived. Usage-only and identity-only chunks must not call it.
func (o *GenerationObservation) FirstOutput() {
	if o == nil {
		return
	}
	o.firstOnce.Do(func() {
		now := time.Now()
		o.span.SetAttributes(
			attribute.String("langfuse.observation.completion_start_time", now.UTC().Format(time.RFC3339Nano)),
			attribute.Float64("gai.gen_ai.time_to_first_output_ms", float64(now.Sub(o.startedAt).Microseconds())/1000),
		)
		o.span.AddEvent("gen_ai.completion.start", trace.WithTimestamp(now))
	})
}

// ObserveToken records normalized streaming metadata and first-output timing.
func (o *GenerationObservation) ObserveToken(token Token) {
	if o == nil {
		return
	}
	switch token.Type() {
	case TokenTypeText, TokenTypeThought:
		if token.Text() != "" {
			o.FirstOutput()
		}
	case TokenTypeToolCall:
		o.FirstOutput()
		o.mu.Lock()
		o.stream.ToolCallCount++
		o.mu.Unlock()
	case TokenTypePart:
		if token.Part != nil {
			o.FirstOutput()
		}
	case TokenTypeCompletion:
		if token.Completion == nil {
			return
		}
		o.mu.Lock()
		completion := token.Completion
		if completion.Model != "" {
			o.stream.ResponseModel = completion.Model
		}
		if completion.RequestID != "" {
			o.stream.RequestID = completion.RequestID
		}
		if completion.FinishReason != "" {
			o.stream.FinishReason = completion.FinishReason
		}
		if completion.UsageReported {
			usage := completion.Usage
			o.stream.Usage = &usage
		}
		o.mu.Unlock()
	}
}

// Finish records final metadata and ends the generation span. It is idempotent.
func (o *GenerationObservation) Finish(result GenerationResult) {
	if o == nil {
		return
	}
	o.finishOnce.Do(func() {
		o.mu.Lock()
		result = mergeGenerationResults(o.stream, result)
		o.mu.Unlock()
		o.span.SetAttributes(generationResultAttributes(result)...)
		gai.EmitObservation(o.ctx, o.config.Sink, gai.Observation{
			Name:   "generation_finished",
			Source: "ai:GenerationObservation",
			Fields: generationResultFields(o.config, result),
			Err:    result.Err,
		})
		gai.EndSpan(o.span, result.Err)
	})
}

func generationResultFields(config GenerationConfig, result GenerationResult) map[string]any {
	fields := map[string]any{
		"provider":        config.Provider,
		"model":           config.Model,
		"streaming":       config.Streaming,
		"tool_call_count": result.ToolCallCount,
	}
	if result.ResponseModel != "" {
		fields["response_model"] = result.ResponseModel
	}
	if result.RequestID != "" {
		fields["request_id"] = result.RequestID
	}
	if result.FinishReason != "" {
		fields["finish_reason"] = result.FinishReason
	}
	if result.HTTPStatus > 0 {
		fields["http_status"] = result.HTTPStatus
	}
	if result.Usage != nil {
		fields["input_tokens"] = result.Usage.InputTokens
		fields["output_tokens"] = result.Usage.OutputTokens
		fields["total_tokens"] = result.Usage.InputTokens + result.Usage.OutputTokens
		if result.Usage.ReasoningTokens > 0 {
			fields["reasoning_tokens"] = result.Usage.ReasoningTokens
		}
		if result.Usage.CachedTokens > 0 {
			fields["cached_tokens"] = result.Usage.CachedTokens
		}
		if result.Usage.CacheCreationTokens > 0 {
			fields["cache_creation_tokens"] = result.Usage.CacheCreationTokens
		}
		if result.Usage.ToolUseTokens > 0 {
			fields["tool_use_tokens"] = result.Usage.ToolUseTokens
		}
	}
	return fields
}

func mergeGenerationResults(stream, final GenerationResult) GenerationResult {
	if final.ResponseModel == "" {
		final.ResponseModel = stream.ResponseModel
	}
	if final.RequestID == "" {
		final.RequestID = stream.RequestID
	}
	if final.FinishReason == "" {
		final.FinishReason = stream.FinishReason
	}
	if final.Usage == nil {
		final.Usage = stream.Usage
	}
	// A caller-provided final count is authoritative. Streaming observations are
	// the fallback so providers may summarize their final result without
	// double-counting tool-call tokens already seen by ObserveToken.
	if final.ToolCallCount == 0 {
		final.ToolCallCount = stream.ToolCallCount
	}
	return final
}

func generationResultAttributes(result GenerationResult) []attribute.KeyValue {
	attrs := []attribute.KeyValue{
		attribute.Int("gai.gen_ai.response.tool_call_count", result.ToolCallCount),
		attribute.Int("ai.tool_call_count", result.ToolCallCount),
	}
	if result.ResponseModel != "" {
		attrs = append(attrs,
			attribute.String("gen_ai.response.model", result.ResponseModel),
			attribute.String("langfuse.observation.model.name", result.ResponseModel),
		)
	}
	if result.RequestID != "" {
		attrs = append(attrs, attribute.String("gen_ai.response.id", result.RequestID))
	}
	if result.FinishReason != "" {
		attrs = append(attrs, attribute.StringSlice("gen_ai.response.finish_reasons", []string{result.FinishReason}))
	}
	if result.HTTPStatus > 0 {
		attrs = append(attrs, attribute.Int("http.response.status_code", result.HTTPStatus))
	}
	if result.Err != nil {
		attrs = append(attrs, attribute.String("error.type", fmt.Sprintf("%T", result.Err)))
	}
	if result.Usage == nil {
		return attrs
	}
	usage := *result.Usage
	attrs = append(attrs,
		attribute.Int("gen_ai.usage.input_tokens", usage.InputTokens),
		attribute.Int("gen_ai.usage.output_tokens", usage.OutputTokens),
		attribute.Int("gen_ai.usage.total_tokens", usage.InputTokens+usage.OutputTokens),
		attribute.Int("ai.input_tokens", usage.InputTokens),
		attribute.Int("ai.output_tokens", usage.OutputTokens),
	)
	details := map[string]int{
		"input": usage.InputTokens, "output": usage.OutputTokens,
		"total": usage.InputTokens + usage.OutputTokens,
	}
	if usage.ReasoningTokens > 0 {
		attrs = append(attrs,
			attribute.Int("gen_ai.usage.reasoning.output_tokens", usage.ReasoningTokens),
			attribute.Int("ai.reasoning_tokens", usage.ReasoningTokens),
		)
		details["reasoning"] = usage.ReasoningTokens
	}
	if usage.CachedTokens > 0 {
		attrs = append(attrs, attribute.Int("gen_ai.usage.cache_read.input_tokens", usage.CachedTokens))
		details["cached"] = usage.CachedTokens
	}
	if usage.CacheCreationTokens > 0 {
		attrs = append(attrs, attribute.Int("gai.gen_ai.usage.cache_creation.input_tokens", usage.CacheCreationTokens))
		details["cache_creation"] = usage.CacheCreationTokens
	}
	if usage.ToolUseTokens > 0 {
		attrs = append(attrs, attribute.Int("gai.gen_ai.usage.tool.input_tokens", usage.ToolUseTokens))
		details["tool_use"] = usage.ToolUseTokens
	}
	if raw, err := json.Marshal(details); err == nil {
		attrs = append(attrs, attribute.String("langfuse.observation.usage_details", string(raw)))
	}
	return attrs
}

func generationModelParameters(req AIRequest) string {
	parameters := make(map[string]any, 2)
	if req.MaxTokens > 0 {
		parameters["max_tokens"] = req.MaxTokens
	}
	if req.ResponseFormat.Type != "" {
		parameters["response_format"] = string(req.ResponseFormat.Type)
	}
	if len(parameters) == 0 {
		return ""
	}
	raw, err := json.Marshal(parameters)
	if err != nil {
		return ""
	}
	return string(raw)
}
