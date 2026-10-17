package loop

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/modelinput"
)

func (r *runExecution) prepareContinuation() error {
	control := r.owner.Continuation
	if control == nil {
		return nil
	}
	if control.Commit == nil || control.RunID == "" || control.ExecutionID == "" || control.StageID == "" {
		return ErrContinuation
	}
	d := &durableLoop{control: control, events: r.events}
	r.owner.durable = d
	if control.Initial != nil {
		d.state = control.Initial.Clone()
		if err := d.state.Validate(); err != nil {
			return err
		}
		if err := d.state.ResolveRecovery(nil); err != nil {
			return err
		}
		builder, err := modelinput.Restore(*d.state.Prompt)
		if err != nil {
			return err
		}
		r.owner.PromptBuilder = builder
		r.owner.Iterations = d.state.Iterations()
		r.state.retryCount = d.state.RetryCount
		if d.state.Phase == PhaseGenerating {
			for _, usage := range d.state.Usage {
				if !usage.Reported {
					d.state.UsageIncomplete = true
				}
			}
			// The interrupted physical attempt consumed retry budget as well.
			if r.owner.RetryPolicy != nil && !r.owner.RetryPolicy.hasRetryBudget(r.state.retryCount) {
				return ErrMaxRetries
			}
			r.state.retryCount++
			d.state.RetryCount = r.state.retryCount
			d.state.Phase = PhaseReady
		}
	} else {
		d.state = Continuation{Version: ContinuationVersion, Phase: PhaseReady, NextIteration: 1, NextAttempt: 1, InputPending: true, RequiredToolSatisfied: r.owner.ToolChoice.Mode != ai.ToolChoiceRequired}
		if r.owner.RetryPolicy != nil && r.owner.RetryPolicy.TotalTimeout > 0 {
			d.state.Deadline = time.Now().Add(r.owner.RetryPolicy.TotalTimeout)
		}
	}
	if !d.state.Deadline.IsZero() {
		var cancel context.CancelFunc
		r.ctx, cancel = context.WithDeadline(r.ctx, d.state.Deadline)
		r.continuationCancel = cancel
	}
	return nil
}

func (r *runExecution) materializeContinuation() error {
	d := r.owner.durable
	if d == nil {
		return nil
	}
	if d.state.Prompt == nil {
		snapshotter, ok := r.owner.PromptBuilder.(interface {
			Snapshot(context.Context) (modelinput.Snapshot, error)
		})
		if !ok {
			return fmt.Errorf("%w: prompt builder does not support snapshots", ErrContinuation)
		}
		snapshot, err := snapshotter.Snapshot(r.ctx)
		if err != nil {
			return err
		}
		d.state.Prompt = &snapshot
		// Execute from the same materialized values that are committed.
		builder, err := modelinput.Restore(snapshot)
		if err != nil {
			return err
		}
		r.owner.PromptBuilder = builder
	}
	r.requiredToolCallSatisfied = d.state.RequiredToolSatisfied
	if !d.state.InputPending {
		r.userMessage = nil
	}
	return d.commit(r.ctx)
}
func (r *runExecution) continuationBoundary() bool {
	d := r.owner.durable
	if d == nil {
		return false
	}
	if d.pausing() {
		r.pauseContinuation()
		return true
	}
	if d.state.Pending == nil && len(d.state.Steering) > 0 {
		g := Generation{ID: newIdentity(), Messages: ai.CloneMessages(d.state.Steering)}
		d.state.Generations = append(d.state.Generations, g)
		d.state.Steering = nil
		r.owner.Iterations = append(r.owner.Iterations, g.iteration())
		if d.state.Phase == PhaseDone {
			d.state.Phase = PhaseReady
		}
		if err := d.commit(r.ctx); err != nil {
			sendLoopError(r.ctx, r.events, r.state, err)
			return true
		}
	}
	return false
}
func (r *runExecution) pauseContinuation() {
	d := r.owner.durable
	if err := d.commit(r.ctx); err != nil {
		sendLoopError(r.ctx, r.events, r.state, err)
		return
	}
	event := Event{Type: EventPaused}
	if d.state.Pending != nil {
		event.IterationCount = d.state.Pending.Iteration
		event.AttemptID = d.state.Pending.Attempt
		iteration := d.state.Pending.iteration()
		event.Iteration = &iteration
	}
	_ = sendEvent(r.ctx, r.events, event)
}
func (r *runExecution) resumePending() iterationOutcome {
	d := r.owner.durable
	g := d.state.Pending
	ctx, state := r.state.startIteration(r.ctx, g.Iteration, g.Attempt)
	ctx, cancel := context.WithCancel(ctx)
	d.generationID = g.ID
	a := &attemptExecution{run: r, ctx: ctx, state: state, iteration: g.iteration(), cancel: cancel, generationID: g.ID}
	// The saved pending message has the accepted assistant, but no emitted tool
	// results. Restore diagnostics in precisely the plan's original order.
	for i, inv := range g.Invocations {
		a.toolCalls = append(a.toolCalls, pendingToolCall{partIndex: i, call: inv.Call.Clone()})
	}
	for _, m := range g.Messages {
		if m.Role == ai.RoleAssistant {
			for _, part := range m.Parts {
				if part.Kind == ai.ContentText || part.Kind == ai.ContentReasoning {
					token := ai.Token{Part: &part}
					if err := sendEvent(r.ctx, r.events, TokenEvent(g.Iteration, g.Attempt, r.state.retryCount, token)); err != nil {
						a.terminateSendFailure(err)
						return iterationTerminal
					}
				}
			}
		}
	}
	return r.postAttempt(a, false)
}
func (r *runExecution) commitModelStart(a *attemptExecution) error {
	d := r.owner.durable
	if d == nil {
		return nil
	}
	d.generationID = newIdentity()
	a.generationID = d.generationID
	d.state.Usage = append(d.state.Usage, GenerationUsage{ID: d.generationID})
	d.state.Phase = PhaseGenerating
	d.state.NextIteration = a.iteration.Count
	d.state.NextAttempt = a.state.attemptID() + 1
	d.state.RetryCount = r.state.retryCount
	return d.commit(r.ctx)
}
func (r *runExecution) commitPlan(a *attemptExecution) error {
	d := r.owner.durable
	if d == nil {
		return nil
	}
	if d.state.Pending != nil {
		return nil
	}
	g := generationFromIteration(d.generationID, a.state.attemptID(), a.iteration)
	for _, call := range a.toolCalls {
		g.Invocations = append(g.Invocations, InvocationRecord{ID: newIdentity(), Call: call.call.Clone(), Disposition: InvocationPlanned, Execution: ToolExecution{State: ToolNotStarted}})
	}
	d.state.Pending = &g
	d.state.Phase = PhaseTools
	d.account(g.ID, a.iteration)
	return d.commit(r.ctx)
}
func (d *durableLoop) account(id string, it Iteration) {
	for index, v := range d.state.Usage {
		if v.ID == id {
			d.state.Usage[index] = GenerationUsage{ID: id, Usage: it.Usage, Reported: it.UsageReported}
			if !it.UsageReported {
				d.state.UsageIncomplete = true
			}
			return
		}
	}
	d.state.Usage = append(d.state.Usage, GenerationUsage{ID: id, Usage: it.Usage, Reported: it.UsageReported})
	if !it.UsageReported {
		d.state.UsageIncomplete = true
	}
}
func (r *runExecution) commitDiscard(a *attemptExecution) error {
	d := r.owner.durable
	if d == nil {
		return nil
	}
	d.account(d.generationID, a.iteration)
	d.state.Phase = PhaseReady
	d.state.NextIteration = a.iteration.Count + 1
	d.state.NextAttempt = 1
	d.state.RetryCount = 0
	return d.commit(r.ctx)
}
func (r *runExecution) commitRetry(a *attemptExecution) error {
	d := r.owner.durable
	if d == nil {
		return nil
	}
	d.account(d.generationID, a.iteration)
	d.state.Phase = PhaseReady
	d.state.RetryCount = r.state.retryCount
	return d.commit(r.ctx)
}
func (r *runExecution) commitIteration(a *attemptExecution, final bool) error {
	d := r.owner.durable
	if d == nil {
		return nil
	}
	g := generationFromIteration(d.generationID, a.state.attemptID(), a.iteration)
	g.Invocations = d.state.Pending.Invocations
	d.state.Generations = append(d.state.Generations, g)
	d.state.Pending = nil
	d.state.InputPending = false
	d.state.NextIteration = a.iteration.Count + 1
	d.state.NextAttempt = 1
	d.state.RetryCount = 0
	d.state.RequiredToolSatisfied = r.requiredToolCallSatisfied || hasPermittedToolCall(a.toolCalls, r.owner.Tools, r.owner.ToolChoice.Names)
	d.state.Phase = PhaseReady
	if final && len(d.state.Steering) == 0 {
		d.state.Phase = PhaseDone
	}
	return d.commit(r.ctx)
}
func (d *durableLoop) invocation(index int) Invocation {
	v := d.identity()
	inv := d.state.Pending.Invocations[index]
	v.InvocationID = inv.ID
	v.Attempt = inv.Attempt
	return v
}
func (d *durableLoop) beforeDispatch(ctx context.Context, index int, execution ToolExecution, synthetic bool) error {
	if d.pausing() {
		return ErrPaused
	}
	inv := &d.state.Pending.Invocations[index]
	inv.Execution = execution
	if !synthetic {
		inv.Disposition = InvocationDispatched
		inv.Attempt++
		inv.Execution.State = ToolRunning
	}
	if err := d.commit(ctx); err != nil {
		return err
	}
	if d.pausing() {
		inv.Disposition = InvocationPlanned
		inv.Execution.State = ToolNotStarted
		if err := d.commit(ctx); err != nil {
			return err
		}
		return ErrPaused
	}
	return nil
}
func (d *durableLoop) completed(ctx context.Context, done toolCompletion) error {
	inv := &d.state.Pending.Invocations[done.index]
	inv.Execution = done.execution
	inv.Disposition = InvocationCompleted
	inv.ProcessingFailed = done.err != nil
	if done.result != nil {
		inv.Result = &ai.ToolResult{ToolCallID: inv.Call.ID, Name: inv.Call.Name, Parts: ai.TextParts(done.result.String()), IsError: done.result.Err != nil}
	}
	return d.commit(ctx)
}
func (d *durableLoop) prepareTasks(tasks []scheduledTool) error {
	if d.state.Pending == nil || len(tasks) != len(d.state.Pending.Invocations) {
		return ErrContinuation
	}
	for i := range tasks {
		inv := d.state.Pending.Invocations[i]
		if inv.Disposition == InvocationCompleted {
			if inv.ProcessingFailed {
				return &RecoveryRequiredError{Invocations: []InvocationRecord{inv}}
			}
			tasks[i].execution = inv.Execution
			tasks[i].result = restoredResult(inv.Result)
			tasks[i].restored = true
		} else if inv.Disposition == InvocationDispatched {
			return &RecoveryRequiredError{Invocations: []InvocationRecord{inv}}
		}
	}
	return nil
}
func (d *durableLoop) deferredApprovals(ctx context.Context, l *Loop, tasks []scheduledTool, events chan<- Event, iteration, attempt, retry int) error {
	pending := false
	for i := range tasks {
		task := &tasks[i]
		inv := &d.state.Pending.Invocations[i]
		if task.restored {
			continue
		}
		// A recorded explicit denial survives policy changes. A recorded approval
		// never overrides a new policy denial.
		if inv.ApprovalDecision != nil && !inv.ApprovalDecision.Approved && task.execution.Decision.Action != ToolDeny {
			task.execution.ApprovalID = inv.Approval.ID
			task.execution.Approval = ToolDecision{Action: ToolDeny, Code: "approval_denied", Reason: inv.ApprovalDecision.Reason}
			task.result = &ToolResult{Err: decisionError(task.execution.Approval, ErrToolDenied)}
		}
		if task.execution.Decision.Action != ToolRequireApproval {
			continue
		}
		if inv.Approval == nil {
			inv.Approval = &ToolApprovalRequest{ID: newIdentity(), Input: ToolPolicyInput{Call: inv.Call.Clone(), Traits: task.options.Traits}, Decision: task.execution.Decision}
		}
		task.execution.ApprovalID = inv.Approval.ID
		if inv.ApprovalDecision == nil {
			pending = true
			continue
		}
		task.execution.Approval = ToolDecision{Action: ToolDeny, Code: "approval_denied", Reason: inv.ApprovalDecision.Reason}
		if inv.ApprovalDecision.Approved {
			task.execution.Approval.Action = ToolAllow
			task.execution.Approval.Code = "approval_granted"
			task.result = nil
		} else {
			task.result = &ToolResult{Err: decisionError(task.execution.Approval, ErrToolDenied)}
		}
	}
	for i := range tasks {
		d.state.Pending.Invocations[i].Execution = tasks[i].execution
	}
	if err := d.commit(ctx); err != nil {
		return err
	}
	for i, task := range tasks {
		inv := d.state.Pending.Invocations[i]
		if inv.Approval == nil || task.restored {
			continue
		}
		kind := EventToolApprovalRequested
		if inv.ApprovalDecision != nil {
			kind = EventToolApprovalResolved
		}
		if err := sendEvent(ctx, events, Event{Type: kind, IterationCount: iteration, AttemptID: attempt, RetryCount: retry, ToolCall: &task.pending.call, ToolApproval: inv.Approval, ToolExecution: &task.execution}); err != nil {
			return err
		}
	}
	if pending {
		return ErrPaused
	}
	return nil
}

// Identity is attached at the loop event boundary by the single coordinator.
type durableEventKey struct{}

func stampLoopEvent(ctx context.Context, event Event) Event {
	d, _ := ctx.Value(durableEventKey{}).(*durableLoop)
	if d == nil {
		return event
	}
	event.GenerationID = d.generationID
	if event.ToolCall != nil && d.state.Pending != nil {
		for _, inv := range d.state.Pending.Invocations {
			if inv.Call.ID == event.ToolCall.ID {
				event.InvocationID = inv.ID
				break
			}
		}
	}
	return event
}

func isPause(err error) bool { return errors.Is(err, ErrPaused) }

func (a *attemptExecution) commitTerminalUsage() error {
	d := a.run.owner.durable
	if d == nil || d.failed != nil || a.generationID == "" {
		return nil
	}
	d.account(a.generationID, a.iteration)
	return d.commit(a.run.ctx)
}
