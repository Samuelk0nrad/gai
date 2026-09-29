package ai

import (
	"context"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/internal/observe"
	"go.opentelemetry.io/otel/attribute"
)

type toolCallStreamResult struct {
	inputTokenEvents       int
	outputTokenEvents      int
	detectedToolCallCount  int
	rejectedCandidateCount int
	eofPending             bool

	lastRejectionReason string
	lastRejectedPayload []byte
	lastToolCallName    string
	lastToolCallArgs    []byte
	pendingPayload      []byte
}

type toolCallStreamObserver struct {
	sink              gai.ObservationSink
	operation         *observe.Operation
	captureToolInput  bool
	captureCompletion bool
}

func newToolCallStreamObserver(ctx context.Context, sink gai.ObservationSink) (context.Context, *toolCallStreamObserver) {
	ctx, operation := observe.Start(
		ctx,
		sink,
		aiTracerName,
		"ai",
		"ai.operation",
		"tool_call.detect_stream",
		"ai:DetectToolCallsInStream",
	)
	policy, _ := gai.ContentCapturePolicyFromContext(ctx)
	return ctx, &toolCallStreamObserver{
		sink:              sink,
		operation:         operation,
		captureToolInput:  policy.ToolInput == gai.CaptureEnabled,
		captureCompletion: policy.Completion == gai.CaptureEnabled,
	}
}

func (o *toolCallStreamObserver) Detected(result *toolCallStreamResult, toolCall *ToolCall) {
	if result == nil {
		return
	}
	result.detectedToolCallCount++
	if toolCall == nil {
		return
	}
	result.lastToolCallName = toolCall.Name
	if o != nil && o.captureToolInput {
		result.lastToolCallArgs = append(result.lastToolCallArgs[:0], toolCall.Args...)
	}
}

func (o *toolCallStreamObserver) CandidateRejected(result *toolCallStreamResult, reason string, payload []byte) {
	if result == nil {
		return
	}
	result.rejectedCandidateCount++
	result.lastRejectionReason = reason
	if o != nil && o.captureCompletion {
		result.lastRejectedPayload = append(result.lastRejectedPayload[:0], payload...)
	}
}

func (o *toolCallStreamObserver) Pending(result *toolCallStreamResult, payload []byte) {
	if result == nil || o == nil || !o.captureCompletion {
		return
	}
	result.pendingPayload = append(result.pendingPayload[:0], payload...)
}

func (o *toolCallStreamObserver) Finished(ctx context.Context, result toolCallStreamResult) {
	o.finish(ctx, "completed", result)
}

func (o *toolCallStreamObserver) Canceled(ctx context.Context, result toolCallStreamResult) {
	o.finish(ctx, "canceled", result)
}

func (o *toolCallStreamObserver) finish(ctx context.Context, outcome string, result toolCallStreamResult) {
	if o == nil || o.operation == nil {
		return
	}

	o.operation.Set(
		attribute.Int("ai.input_token_events", result.inputTokenEvents),
		attribute.Int("ai.output_token_events", result.outputTokenEvents),
		attribute.Int("ai.tool_call_count", result.detectedToolCallCount),
		attribute.Int("ai.rejected_tool_call_candidate_count", result.rejectedCandidateCount),
		attribute.Bool("ai.tool_call_stream.eof_pending", result.eofPending),
		attribute.String("ai.tool_call_stream.outcome", outcome),
	)

	fields := map[string]any{
		"outcome":                  outcome,
		"input_token_events":       result.inputTokenEvents,
		"output_token_events":      result.outputTokenEvents,
		"detected_tool_call_count": result.detectedToolCallCount,
		"rejected_candidate_count": result.rejectedCandidateCount,
		"eof_pending":              result.eofPending,
	}
	if result.lastRejectionReason != "" {
		fields["last_rejection_reason"] = result.lastRejectionReason
	}
	if result.lastToolCallName != "" {
		fields["last_tool_call_name"] = result.lastToolCallName
	}
	if result.detectedToolCallCount > 0 {
		gai.AddObservationContent(ctx, o.sink, fields, "last_tool_call_args", gai.ContentKindToolInput, result.lastToolCallArgs)
	}
	if result.rejectedCandidateCount > 0 {
		gai.AddObservationContent(ctx, o.sink, fields, "last_rejected_candidate", gai.ContentKindCompletion, result.lastRejectedPayload)
	}
	if len(result.pendingPayload) > 0 {
		gai.AddObservationContent(ctx, o.sink, fields, "pending_data", gai.ContentKindCompletion, result.pendingPayload)
	}
	o.operation.Emit(ctx, "tool_call_stream_finished", fields, nil)
	o.operation.Finish(nil)
}
