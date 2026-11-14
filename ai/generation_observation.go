package ai

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"time"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/internal/appclock"
	"github.com/Samuelk0nrad/gai/internal/observe"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

const generationTracerName = "github.com/Samuelk0nrad/gai/ai"

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
	// Output is the generated assistant message, when the provider returned
	// one. It is exported on the span only as far as the context's
	// ContentCapturePolicy allows completion capture. Streaming providers
	// leave it nil; the observation then uses the tokens passed to Delivered.
	Output *Message
}

// GenerationObservation owns the one semantic generation span associated with
// one provider invocation. It is safe to finish more than once; only the first
// call ends the span.
type GenerationObservation struct {
	ctx        context.Context
	config     GenerationConfig
	span       trace.Span
	startedAt  time.Time
	clock      *appclock.Clock
	firstOnce  sync.Once
	finishOnce sync.Once
	mu         sync.Mutex
	stream     GenerationResult
	// captureOutput reports whether completion content may be exported. Only
	// then do delivered text and tool-call identities accumulate in
	// deliveredText and deliveredCalls.
	captureOutput  bool
	delivered      bool
	deliveredCount int
	deliveredText  strings.Builder
	deliveredCalls []generationToolCall
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
	// Sinks, redactors, and token sends under the returned context pause the
	// clock, so inference duration excludes application time.
	ctx, clock := appclock.With(ctx)
	// Capture the prompt before timing starts, so encoding and redaction do
	// not count toward inference duration.
	if gai.ObservationContentEnabled(ctx, nil, gai.ContentKindPrompt) {
		if captured, ok := captureGenerationContent(ctx, gai.ContentKindPrompt, generationInputMessages(req.Messages)); ok {
			span.SetAttributes(observe.CapturedContentAttributes("gai.gen_ai.input", captured, "langfuse.observation.input")...)
		}
	}
	captureOutput := gai.ObservationContentEnabled(ctx, nil, gai.ContentKindCompletion)
	startedAt := time.Now()
	return ctx, &GenerationObservation{
		ctx: ctx, config: config, span: span, startedAt: startedAt, clock: clock,
		captureOutput: captureOutput,
	}
}

// generationMessage is the content-capture view of one message: its role and
// text, the identities of the tool calls it makes, and the calls its tool
// results answer. Tool arguments and results are tool content, governed by
// their own capture kinds and recorded on tool spans. Reasoning, media, and
// opaque provider state are never included.
type generationMessage struct {
	Role        Role                   `json:"role"`
	Content     string                 `json:"content,omitempty"`
	ToolCalls   []generationToolCall   `json:"tool_calls,omitempty"`
	ToolResults []generationToolResult `json:"tool_results,omitempty"`
}

type generationToolCall struct {
	ID   string `json:"id,omitempty"`
	Name string `json:"name"`
}

type generationToolResult struct {
	ToolCallID string `json:"tool_call_id,omitempty"`
	Name       string `json:"name,omitempty"`
}

func newGenerationMessage(message Message) generationMessage {
	out := generationMessage{Role: message.Role}
	if message.Role != RoleTool {
		out.Content = message.Text()
	}
	for _, part := range message.Parts {
		switch {
		case part.ToolCall != nil:
			out.ToolCalls = append(out.ToolCalls, generationToolCall{ID: part.ToolCall.ID, Name: part.ToolCall.Name})
		case part.ToolResult != nil:
			out.ToolResults = append(out.ToolResults, generationToolResult{ToolCallID: part.ToolResult.ToolCallID, Name: part.ToolResult.Name})
		}
	}
	return out
}

func generationInputMessages(messages []Message) []generationMessage {
	out := make([]generationMessage, 0, len(messages))
	for _, message := range messages {
		out = append(out, newGenerationMessage(message))
	}
	return out
}

// captureGenerationContent encodes value as JSON and applies the context's
// ContentCapturePolicy for kind. Callers check ObservationContentEnabled
// first, so no content view is built when capture is off.
func captureGenerationContent(ctx context.Context, kind gai.ContentKind, value any) (gai.CapturedContent, bool) {
	raw, err := json.Marshal(value)
	if err != nil {
		return gai.CapturedContent{}, false
	}
	return gai.CaptureContent(ctx, kind, raw)
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

// Delivered records a token the provider delivered to its caller, after any
// text-encoded tool-call detection. Streaming providers call it once
// SendToken succeeds, so the generation's output and tool-call count describe
// what the caller received. Once Delivered was called, its tool-call count
// replaces the one ObserveToken counts.
func (o *GenerationObservation) Delivered(token Token) {
	if o == nil {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	o.delivered = true
	switch token.Type() {
	case TokenTypeText:
		if o.captureOutput {
			o.deliveredText.WriteString(token.Text())
		}
	case TokenTypeToolCall:
		call := token.ToolCall()
		if call == nil {
			return
		}
		o.deliveredCount++
		if o.captureOutput {
			o.deliveredCalls = append(o.deliveredCalls, generationToolCall{ID: call.ID, Name: call.Name})
		}
	}
}

// Finish records final metadata and generation metrics, then ends the
// generation span. It is idempotent.
func (o *GenerationObservation) Finish(result GenerationResult) {
	if o == nil {
		return
	}
	o.finishOnce.Do(func() {
		// Measure before emitting so the final sink call is not attributed to
		// the model; earlier sink, redactor, and token-send time is already
		// excluded by the clock.
		duration := o.clock.Since(o.startedAt)
		o.mu.Lock()
		stream := o.stream
		if o.delivered {
			stream.ToolCallCount = o.deliveredCount
		}
		result = mergeGenerationResults(stream, result)
		var output *generationMessage
		switch {
		case !o.captureOutput:
		case result.Output != nil:
			message := newGenerationMessage(*result.Output)
			output = &message
		case o.delivered:
			output = &generationMessage{Content: o.deliveredText.String(), ToolCalls: o.deliveredCalls}
		}
		o.mu.Unlock()
		o.span.SetAttributes(generationResultAttributes(result)...)
		// Set every attribute before the span ends: Langfuse does not merge
		// attributes from a span exported more than once.
		o.setOutput(output)
		// Record metrics before running the application sink, which may panic.
		recordGenerationMetrics(o.ctx, o.config, result, duration)
		gai.EmitObservation(o.ctx, o.config.Sink, gai.Observation{
			Name:   "generation_finished",
			Source: "ai:GenerationObservation",
			Fields: generationResultFields(o.config, result),
			Err:    result.Err,
		})
		gai.EndSpan(o.span, result.Err)
	})
}

// setOutput exports the completion under the context's ContentCapturePolicy.
// A response with neither text nor tool calls exports nothing.
func (o *GenerationObservation) setOutput(output *generationMessage) {
	if output == nil || (output.Content == "" && len(output.ToolCalls) == 0) {
		return
	}
	output.Role = RoleAssistant
	if captured, ok := captureGenerationContent(o.ctx, gai.ContentKindCompletion, output); ok {
		o.span.SetAttributes(observe.CapturedContentAttributes("gai.gen_ai.output", captured, "langfuse.observation.output")...)
	}
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
		attrs = append(attrs, attribute.String("error.type", TelemetryErrorType(result.Err)))
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
