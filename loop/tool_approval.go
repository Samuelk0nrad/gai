package loop

import (
	"context"
	"crypto/rand"
	"fmt"
)

// ToolApprovalRequest is a snapshot of one pending authorization request. ID is
// a fresh correlation token, not a durable invocation or recovery identity.
type ToolApprovalRequest struct {
	ID       string
	Input    ToolPolicyInput
	Decision ToolDecision
}

// Clone copies the request and its mutable call arguments and extensions.
func (r ToolApprovalRequest) Clone() ToolApprovalRequest {
	r.Input.Call = r.Input.Call.Clone()
	return r
}

// ToolApprovalDecision must echo the exact request ID, for approval and denial.
// Reason must be safe to expose in events and model-visible refusal results.
type ToolApprovalDecision struct {
	RequestID string
	Approved  bool
	Reason    string
}

// ToolApprovalResolver resolves approvals within a live run. Implementations
// own their UI/transport and must respect ctx. Calls are sequential within a
// batch, before any worker or guard is acquired. Shared resolvers must be safe
// for concurrent runs. This interface does not provide durable pause/resume.
type ToolApprovalResolver interface {
	ResolveToolApproval(context.Context, ToolApprovalRequest) (ToolApprovalDecision, error)
}

// ToolApprovalResolverFunc adapts a function to ToolApprovalResolver.
type ToolApprovalResolverFunc func(context.Context, ToolApprovalRequest) (ToolApprovalDecision, error)

// ResolveToolApproval calls the adapter after checking for a nil function.
func (f ToolApprovalResolverFunc) ResolveToolApproval(ctx context.Context, request ToolApprovalRequest) (ToolApprovalDecision, error) {
	if f == nil {
		return ToolApprovalDecision{}, fmt.Errorf("%w: nil resolver", ErrToolApproval)
	}
	return f(ctx, request)
}

func resolveToolApproval(ctx context.Context, resolver ToolApprovalResolver, request ToolApprovalRequest) (decision ToolApprovalDecision, err error) {
	defer func() {
		if recover() != nil {
			decision, err = ToolApprovalDecision{}, ErrToolPanic
		}
	}()
	if err := ctx.Err(); err != nil {
		return ToolApprovalDecision{}, err
	}
	decision, err = resolver.ResolveToolApproval(ctx, request.Clone())
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ToolApprovalDecision{}, ctxErr
	}
	if err != nil {
		return ToolApprovalDecision{}, fmt.Errorf("%w: %w", ErrToolApproval, err)
	}
	if decision.RequestID != request.ID {
		return ToolApprovalDecision{}, fmt.Errorf("%w: response request ID mismatch", ErrToolApproval)
	}
	return decision, nil
}

func (l *Loop) preflightToolApprovals(ctx context.Context, tasks []scheduledTool, events chan<- Event, iteration, attempt, retry int) error {
	for i := range tasks {
		task := &tasks[i]
		if err := ctx.Err(); err != nil {
			return err
		}
		if task.execution.Decision.Action != ToolRequireApproval {
			continue
		}
		request := ToolApprovalRequest{ID: rand.Text(), Input: ToolPolicyInput{Call: task.pending.call.Clone(), Traits: task.options.Traits}, Decision: task.execution.Decision}
		task.execution.ApprovalID = request.ID
		eventFor := func(kind EventType, err error) Event {
			return Event{Type: kind, IterationCount: iteration, AttemptID: attempt, RetryCount: retry, ToolCall: &task.pending.call, ToolApproval: &request, ToolExecution: &task.execution, Err: err}
		}
		emit := func(kind EventType) error {
			if events == nil {
				return nil
			}
			return sendEvent(ctx, events, eventFor(kind, nil))
		}
		failed := func(requested bool) {
			safeErr := ErrToolApproval
			task.execution.Approval = ToolDecision{Action: ToolDeny, Code: "approval_failed", Reason: "tool approval failed"}
			if ctx.Err() != nil {
				safeErr = ctx.Err()
				task.execution.Approval.Code = "approval_canceled"
				task.execution.Approval.Reason = "tool approval canceled"
			}
			if !requested || events == nil {
				return
			}
			event := eventFor(EventToolApprovalResolved, safeErr)
			if ctx.Err() == nil && sendEvent(ctx, events, event) == nil {
				return
			}
			// Cancellation must not wait for a consumer that has stopped reading.
			// The execution snapshot retains this outcome if the buffer is full.
			sendTerminalEvent(ctx, events, cloneToolEventPayload(event))
		}
		if err := emit(EventToolApprovalRequested); err != nil {
			failed(false)
			return err
		}
		if l.ToolApprovalResolver == nil {
			task.execution.Approval = ToolDecision{Action: ToolDeny, Code: "approval_unavailable", Reason: "tool approval resolver is not configured"}
		} else {
			decision, err := resolveToolApproval(ctx, l.ToolApprovalResolver, request)
			if err != nil {
				failed(true)
				return err
			}
			task.execution.Approval = ToolDecision{Action: ToolDeny, Code: "approval_denied", Reason: decision.Reason}
			if decision.Approved {
				task.execution.Approval.Action = ToolAllow
				task.execution.Approval.Code = "approval_granted"
				task.result = nil
			} else {
				task.result = &ToolResult{Err: decisionError(task.execution.Approval, ErrToolDenied)}
			}
		}
		if err := emit(EventToolApprovalResolved); err != nil {
			return err
		}
	}
	return nil
}
