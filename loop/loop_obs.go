package loop

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// retryReason maps generation errors to stable retry classifications.
func retryReason(err error) string {
	if errors.Is(err, ErrAttemptTimeout) {
		return "attempt_timeout"
	}
	var providerErr *ai.ProviderError
	if errors.As(err, &providerErr) && providerErr.Kind != "" {
		return string(providerErr.Kind)
	}
	return string(ai.ProviderErrorUnknown)
}

const loopTracerName = "github.com/lace-ai/gai/loop"

const (
	toolOutcomeSuccess      = "success"
	toolOutcomeError        = "tool_error"
	toolOutcomePanic        = "panic"
	toolOutcomeDeadline     = "deadline"
	toolOutcomeCancellation = "cancellation"
	toolOutcomeProcessing   = "processing_error"
)

var (
	errObservedToolExecution    = errors.New("tool execution failed")
	errObservedToolPanic        = errors.New("tool execution panicked")
	errObservedToolDeadline     = errors.New("tool execution deadline exceeded")
	errObservedToolCancellation = errors.New("tool execution canceled")
	errObservedToolProcessing   = errors.New("tool result processing failed")
)

type loopRunState struct {
	obs        *loopObserver
	sink       gai.ObservationSink
	err        error
	cancelErr  error
	retryCount int
	stats      loopRunStats
}

type loopRunStats struct {
	IterationCount    int
	TokenCount        int
	ToolCallCount     int
	IncrementalPrompt bool
}

type loopIterationState struct {
	obs   *iterationObserver
	ctx   context.Context
	sink  gai.ObservationSink
	stats loopIterationStats
}

type loopIterationStats struct {
	Iteration      int
	Retrying       bool
	Final          bool
	Canceled       bool
	AttemptID      int
	PartCount      int
	RetryCount     int
	RetryReason    string
	RetryDelay     time.Duration
	ToolCallCount  int
	ToolErrorCount int
}

type loopObserver struct {
	span trace.Span
}

func newLoopRunState(ctx context.Context, l *Loop) (context.Context, *loopRunState) {
	maxIterations := 0
	retryLimit := 0
	maxTokens := 0
	toolCount := 0
	modelName := ""
	var sink gai.ObservationSink
	if l != nil {
		maxIterations = l.MaxLoopIterations
		if l.RetryPolicy != nil {
			retryLimit = l.RetryPolicy.MaxRetries
		}
		maxTokens = l.MaxTokens
		toolCount = len(l.Tools)
		if l.Model != nil {
			modelName = ai.ModelName(l.Model)
		}
		sink = l.ObservationSink
	}

	ctx, span := gai.StartOperationSpan(ctx, loopTracerName, "loop", "loop.operation", "run",
		attribute.Int("loop.max_iterations", maxIterations),
		attribute.Int("loop.retry_limit", retryLimit),
		attribute.Int("loop.max_tokens", maxTokens),
		attribute.Int("loop.tool_count", toolCount),
		attribute.String("ai.model", modelName),
	)
	return ctx, &loopRunState{obs: &loopObserver{span: span}, sink: sink}
}

func (s *loopRunState) startIteration(ctx context.Context, count int, attempt int) (context.Context, *loopIterationState) {
	incrementalPrompt := false
	var sink gai.ObservationSink
	if s != nil {
		s.stats.IterationCount = count
		incrementalPrompt = s.stats.IncrementalPrompt
		sink = s.sink
	}
	ctx, span := gai.StartOperationSpan(ctx, loopTracerName, "loop", "loop.operation", "iteration",
		attribute.Int("loop.iteration", count),
		attribute.Int("loop.attempt", attempt),
		attribute.Bool("loop.incremental_prompt", incrementalPrompt),
	)
	return ctx, &loopIterationState{
		obs:   &iterationObserver{span: span},
		ctx:   ctx,
		sink:  sink,
		stats: loopIterationStats{Iteration: count, AttemptID: attempt},
	}
}

func (s *loopRunState) recordToken(token ai.Token) {
	if s == nil {
		return
	}
	s.stats.TokenCount++
	if token.Type() == ai.TokenTypeToolCall && token.ToolCall() != nil {
		s.stats.ToolCallCount++
	}
}

func (s *loopRunState) retry() {
	if s == nil {
		return
	}
	s.retryCount++
}

func (s *loopRunState) resetRetries() {
	if s == nil {
		return
	}
	s.retryCount = 0
}

func (s *loopRunState) fail(err error) {
	if s != nil {
		s.err = err
	}
}

func (s *loopRunState) cancel(err error) {
	if s != nil {
		s.cancelErr = err
	}
}

func (s *loopRunState) finish() {
	if s == nil || s.obs == nil {
		return
	}
	s.obs.finish(s.err, s.cancelErr, s.stats)
}

func (o *loopObserver) finish(err, cancelErr error, stats loopRunStats) {
	if o == nil || o.span == nil {
		return
	}
	attrs := []attribute.KeyValue{
		attribute.Int("loop.iteration_count", stats.IterationCount),
		attribute.Int("loop.token_count", stats.TokenCount),
		attribute.Int("loop.tool_call_count", stats.ToolCallCount),
		attribute.Bool("loop.incremental_prompt", stats.IncrementalPrompt),
	}
	if cancelErr != nil {
		attrs = append(attrs,
			attribute.Bool("loop.canceled", true),
			attribute.String("loop.cancel_reason", cancelErr.Error()),
		)
	}
	o.span.SetAttributes(attrs...)
	gai.EndSpan(o.span, err)
}

type iterationObserver struct {
	span trace.Span
}

func (s *loopIterationState) recordToken(token ai.Token) {
	if s == nil {
		return
	}
	s.stats.PartCount++
	if token.Type() == ai.TokenTypeToolCall && token.ToolCall() != nil {
		s.stats.ToolCallCount++
	}
}

// recordToolResults counts processed tool failures for iteration telemetry.
func (s *loopIterationState) recordToolResults(iteration Iteration) {
	if s == nil {
		return
	}
	s.stats.PartCount = len(iteration.Parts)
	for _, part := range iteration.Parts {
		if part.ToolResp != nil && part.ToolResp.Err != nil {
			s.stats.ToolErrorCount++
		}
	}
}

func (s *loopIterationState) recordIteration(iteration Iteration) {
	if s != nil {
		s.stats.PartCount = len(iteration.Parts)
	}
}

func (s *loopIterationState) markRetrying(retryCount int, reason string, delay time.Duration) {
	if s == nil {
		return
	}
	s.stats.Retrying = true
	s.stats.RetryCount = retryCount
	s.stats.RetryReason = reason
	s.stats.RetryDelay = delay
	gai.EmitObservation(s.ctx, s.sink, gai.Observation{
		Name:   "loop_retry_scheduled",
		Source: "loop:Iteration",
		Fields: map[string]any{
			"iteration":      s.stats.Iteration,
			"attempt":        s.stats.AttemptID,
			"retry_count":    retryCount,
			"retry_reason":   reason,
			"retry_delay_ms": delay.Milliseconds(),
		},
	})
}

func (s *loopIterationState) markFinal() {
	if s == nil {
		return
	}
	s.stats.Final = true
}

func (s *loopIterationState) markCanceled(err error) {
	if s != nil {
		s.stats.Canceled = true
	}
}

func (s *loopIterationState) attemptID() int {
	if s == nil {
		return 0
	}
	return s.stats.AttemptID
}

func (s *loopIterationState) finish(err error) {
	if s == nil || s.obs == nil {
		return
	}
	s.obs.finish(err, s.stats)
}

func (o *iterationObserver) finish(err error, stats loopIterationStats) {
	if o == nil || o.span == nil {
		return
	}
	attrs := []attribute.KeyValue{
		attribute.Int("loop.attempt", stats.AttemptID),
		attribute.Int("loop.part_count", stats.PartCount),
		attribute.Int("loop.retry_count", stats.RetryCount),
		attribute.Int("loop.tool_call_count", stats.ToolCallCount),
		attribute.Int("loop.tool_error_count", stats.ToolErrorCount),
	}
	if stats.Retrying {
		attrs = append(attrs,
			attribute.Bool("loop.retrying", true),
			attribute.String("loop.retry_reason", stats.RetryReason),
			attribute.Int64("loop.retry_delay_ms", stats.RetryDelay.Milliseconds()),
		)
	}
	if stats.Final {
		attrs = append(attrs, attribute.Bool("loop.final_iteration", true))
	}
	if stats.Canceled {
		attrs = append(attrs, attribute.Bool("loop.canceled", true))
	}
	o.span.SetAttributes(attrs...)
	gai.EndSpan(o.span, err)
}

type toolObservation struct {
	ctx        context.Context
	span       trace.Span
	sink       gai.ObservationSink
	call       ai.ToolCall
	finishOnce sync.Once
}

func startToolSpan(ctx context.Context, call ai.ToolCall, sinks ...gai.ObservationSink) (context.Context, *toolObservation) {
	ctx, span := gai.StartOperationSpan(ctx, loopTracerName, "loop", "loop.operation", "tool",
		attribute.String("tool.name", call.Name),
		attribute.String("tool.call_id", call.ID),
		attribute.String("gen_ai.operation.name", "execute_tool"),
		attribute.String("gen_ai.tool.name", call.Name),
		attribute.String("gen_ai.tool.type", "function"),
		attribute.String("gen_ai.tool.call.id", call.ID),
		attribute.String("langfuse.observation.type", "tool"),
	)
	if captured, ok := gai.CaptureContent(ctx, gai.ContentKindToolInput, call.Args); ok {
		setCapturedToolContent(span, "tool.input", captured,
			"gen_ai.tool.call.arguments",
			"langfuse.observation.input",
		)
	}
	var sink gai.ObservationSink
	if len(sinks) > 0 {
		sink = sinks[0]
	}
	return ctx, &toolObservation{ctx: ctx, span: span, sink: sink, call: call}
}

// callObservedTool instruments a direct invocation without a result processor.
func callObservedTool(ctx context.Context, call ai.ToolCall, tools []Tool, sinks ...gai.ObservationSink) (*ToolResult, time.Duration) {
	result, duration, _ := processObservedTool(ctx, ToolPolicyInput{Call: call}, tools, nil, sinks...)
	return result, duration
}

// processObservedTool filters results before observation or caller publication.
func processObservedTool(ctx context.Context, input ToolPolicyInput, tools []Tool, processor ToolResultProcessor, sinks ...gai.ObservationSink) (response *ToolResult, duration time.Duration, processErr error) {
	toolCtx, observation := startToolSpan(ctx, input.Call, sinks...)
	started := time.Now()
	defer func() {
		if panicValue := recover(); panicValue != nil {
			observation.finishPanic(time.Since(started))
			panic(panicValue)
		}
		if processErr != nil {
			observation.finishProcessingError(duration)
		} else {
			observation.finish(response, duration)
		}
	}()
	result := CallTool(toolCtx, input.Call, tools)
	duration = time.Since(started)
	if processor != nil {
		input.Call = input.Call.Clone()
		processed, err := processor.Process(toolCtx, input, result)
		if err != nil {
			return nil, duration, fmt.Errorf("%w: %w", ErrToolResultProcess, err)
		}
		result = normalizeToolResult(processed.Text, processed.Err)
	}
	return &result, duration, nil
}

// finish captures only processed output and closes the observation once.
func (o *toolObservation) finish(response *ToolResult, duration time.Duration) {
	if o == nil {
		return
	}
	o.finishOnce.Do(func() {
		outcome, output, spanErr := toolResult(response)
		if response != nil {
			if captured, ok := gai.CaptureContent(o.ctx, gai.ContentKindToolOutput, []byte(output)); ok {
				aliases := []string{"langfuse.observation.output"}
				if outcome == toolOutcomeSuccess {
					aliases = append(aliases, "gen_ai.tool.call.result")
				}
				setCapturedToolContent(o.span, "tool.output", captured, aliases...)
			}
		}
		o.setOutcome(outcome, spanErr, duration)
	})
}

// finishPanic records a safe classification without the panic payload.
func (o *toolObservation) finishPanic(duration time.Duration) {
	if o == nil {
		return
	}
	o.finishOnce.Do(func() {
		o.setOutcome(toolOutcomePanic, errObservedToolPanic, duration)
	})
}

// finishProcessingError records a processing failure without capturing raw output.
func (o *toolObservation) finishProcessingError(duration time.Duration) {
	o.finishOnce.Do(func() { o.setOutcome(toolOutcomeProcessing, errObservedToolProcessing, duration) })
}

// setOutcome emits fixed error classifications and closes the tool span.
func (o *toolObservation) setOutcome(outcome string, spanErr error, duration time.Duration) {
	status := "success"
	attrs := []attribute.KeyValue{attribute.String("gai.tool.outcome", outcome)}
	if spanErr != nil {
		status = "error"
		attrs = append(attrs, attribute.String("error.type", "gai.tool."+outcome))
	}
	attrs = append(attrs, attribute.String("tool.status", status))
	o.span.SetAttributes(attrs...)
	gai.EmitObservation(o.ctx, o.sink, gai.Observation{
		Name:   "loop_tool_finished",
		Source: "loop:Tool",
		Fields: map[string]any{
			"tool_name":    o.call.Name,
			"tool_call_id": o.call.ID,
			"tool_type":    o.call.Type,
			"outcome":      outcome,
			"tool_outcome": outcome,
			"status":       status,
			"duration_ms":  duration.Milliseconds(),
		},
		Err: spanErr,
	})
	gai.EndSpan(o.span, spanErr)
}

// toolResult separates model-facing output from safe telemetry errors.
func toolResult(response *ToolResult) (outcome string, output string, spanErr error) {
	if response == nil {
		return toolOutcomeError, "", errObservedToolExecution
	}
	responseErr := response.Err
	if responseErr == nil {
		return toolOutcomeSuccess, response.Text, nil
	}
	output = responseErr.Error()
	switch {
	case errors.Is(responseErr, context.DeadlineExceeded):
		return toolOutcomeDeadline, output, errObservedToolDeadline
	case errors.Is(responseErr, context.Canceled):
		return toolOutcomeCancellation, output, errObservedToolCancellation
	default:
		return toolOutcomeError, output, errObservedToolExecution
	}
}

func setCapturedToolContent(span trace.Span, prefix string, captured gai.CapturedContent, aliases ...string) {
	attrs := []attribute.KeyValue{
		attribute.String(prefix, string(captured.Value)),
		attribute.Int(prefix+".original_bytes", captured.OriginalBytes),
		attribute.Int(prefix+".captured_bytes", captured.CapturedBytes),
		attribute.Bool(prefix+".truncated", captured.Truncated),
		attribute.Bool(prefix+".redaction_applied", captured.RedactionApplied),
	}
	for _, alias := range aliases {
		attrs = append(attrs, attribute.String(alias, string(captured.Value)))
	}
	span.SetAttributes(attrs...)
}
