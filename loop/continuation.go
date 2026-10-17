package loop

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/modelinput"
)

const ContinuationVersion = 1

type ContinuationPhase string

const (
	PhaseReady      ContinuationPhase = "ready"
	PhaseGenerating ContinuationPhase = "generating"
	PhaseTools      ContinuationPhase = "tools"
	PhaseDone       ContinuationPhase = "done"
)

var (
	ErrContinuation     = errors.New("invalid loop continuation")
	ErrCheckpointCommit = errors.New("checkpoint commit failed")
	ErrPaused           = errors.New("loop paused")
	ErrRecoveryRequired = errors.New("tool recovery required")
)

// Invocation identifies an accepted call independently of the provider call ID.
// Attempt counts physical dispatches, including explicitly authorized replays.
type Invocation struct {
	RunID        string `json:"run_id"`
	ExecutionID  string `json:"execution_id"`
	StageID      string `json:"stage_id"`
	GenerationID string `json:"generation_id"`
	InvocationID string `json:"invocation_id"`
	Attempt      int    `json:"attempt"`
}
type invocationKey struct{}

func InvocationFromContext(ctx context.Context) (Invocation, bool) {
	v, ok := ctx.Value(invocationKey{}).(Invocation)
	return v, ok
}

type InvocationDisposition string

const (
	InvocationPlanned    InvocationDisposition = "planned"
	InvocationDispatched InvocationDisposition = "dispatched"
	InvocationCompleted  InvocationDisposition = "completed"
)

// InvocationRecord retains only processed output. Processing failure is distinct
// from handler failure and never makes a dispatched handler safe to repeat.
type InvocationRecord struct {
	ID               string                `json:"id"`
	Call             ai.ToolCall           `json:"call"`
	Disposition      InvocationDisposition `json:"disposition"`
	Attempt          int                   `json:"attempt"`
	Execution        ToolExecution         `json:"execution"`
	Result           *ai.ToolResult        `json:"result,omitempty"`
	ProcessingFailed bool                  `json:"processing_failed,omitempty"`
	Approval         *ToolApprovalRequest  `json:"approval,omitempty"`
	ApprovalDecision *ToolApprovalDecision `json:"approval_decision,omitempty"`
}

// Generation is semantic accepted state, not a serialized runtime Iteration.
type Generation struct {
	ID            string             `json:"id"`
	Iteration     int                `json:"iteration"`
	Attempt       int                `json:"attempt"`
	Messages      []ai.Message       `json:"messages"`
	InputMessages int                `json:"input_messages"`
	Usage         ai.Usage           `json:"usage"`
	UsageReported bool               `json:"usage_reported"`
	Invocations   []InvocationRecord `json:"invocations,omitempty"`
}
type GenerationUsage struct {
	ID       string   `json:"id"`
	Usage    ai.Usage `json:"usage"`
	Reported bool     `json:"reported"`
}

// Continuation contains the next operation, consumed limits, exact materialized
// prompt, accepted conversation and independently committed tool outcomes.
type Continuation struct {
	Version               int                  `json:"version"`
	Phase                 ContinuationPhase    `json:"phase"`
	Prompt                *modelinput.Snapshot `json:"prompt,omitempty"`
	NextIteration         int                  `json:"next_iteration"`
	NextAttempt           int                  `json:"next_attempt"`
	RetryCount            int                  `json:"retry_count"`
	Deadline              time.Time            `json:"deadline,omitempty"`
	RequiredToolSatisfied bool                 `json:"required_tool_satisfied"`
	InputPending          bool                 `json:"input_pending"`
	Generations           []Generation         `json:"generations,omitempty"`
	Pending               *Generation          `json:"pending,omitempty"`
	Usage                 []GenerationUsage    `json:"usage,omitempty"`
	UsageIncomplete       bool                 `json:"usage_incomplete,omitempty"`
	Steering              []ai.Message         `json:"steering,omitempty"`
}

func (c Continuation) Clone() Continuation {
	// All continuation fields are JSON values; no error interfaces or callbacks.
	data, _ := json.Marshal(c)
	var copy Continuation
	_ = json.Unmarshal(data, &copy)
	return copy
}

func (c Continuation) Validate() error {
	if _, err := json.Marshal(c); err != nil {
		return ErrContinuation
	}
	if c.Version != ContinuationVersion || c.NextIteration < 1 || c.NextAttempt < 1 || c.RetryCount < 0 {
		return ErrContinuation
	}
	switch c.Phase {
	case PhaseReady, PhaseGenerating, PhaseTools, PhaseDone:
	default:
		return ErrContinuation
	}
	if (c.Phase == PhaseTools) != (c.Pending != nil) {
		return ErrContinuation
	}
	if c.Prompt == nil {
		return ErrContinuation
	}
	if _, err := modelinput.Restore(*c.Prompt); err != nil {
		return fmt.Errorf("%w: %v", ErrContinuation, err)
	}
	ids := map[string]bool{}
	all := append([]Generation(nil), c.Generations...)
	if c.Pending != nil {
		all = append(all, *c.Pending)
	}
	for _, g := range all {
		if g.ID == "" || ids[g.ID] || g.Iteration < 0 || g.Attempt < 0 || g.InputMessages < 0 || g.InputMessages > len(g.Messages) {
			return ErrContinuation
		}
		ids[g.ID] = true
		for _, m := range g.Messages {
			if err := m.Validate(); err != nil {
				return fmt.Errorf("%w: %v", ErrContinuation, err)
			}
		}
		var calls []ai.ToolCall
		for _, m := range g.Messages {
			calls = append(calls, m.ToolCalls()...)
		}
		if len(calls) != len(g.Invocations) {
			return ErrContinuation
		}
		for i, inv := range g.Invocations {
			if inv.ID == "" || ids[inv.ID] || inv.Attempt < 0 || !sameJSON(calls[i], inv.Call) {
				return ErrContinuation
			}
			ids[inv.ID] = true
			switch inv.Disposition {
			case InvocationPlanned:
				if inv.Result != nil || inv.ProcessingFailed {
					return ErrContinuation
				}
			case InvocationDispatched:
				if inv.Attempt < 1 || inv.Result != nil {
					return ErrContinuation
				}
			case InvocationCompleted:
				if inv.Result == nil && !inv.ProcessingFailed {
					return ErrContinuation
				}
			default:
				return ErrContinuation
			}
			if inv.Result != nil {
				if inv.Result.ToolCallID != inv.Call.ID || inv.Result.Name != inv.Call.Name {
					return ErrContinuation
				}
				if err := (ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: inv.Result}}}).Validate(); err != nil {
					return ErrContinuation
				}
			}
			if inv.Approval != nil && (inv.Approval.ID == "" || !sameJSON(inv.Approval.Input.Call, inv.Call)) {
				return ErrContinuation
			}
			if inv.ApprovalDecision != nil && (inv.Approval == nil || inv.ApprovalDecision.RequestID != inv.Approval.ID) {
				return ErrContinuation
			}
		}
	}
	return nil
}
func sameJSON(a, b any) bool {
	x, e := json.Marshal(a)
	y, f := json.Marshal(b)
	return e == nil && f == nil && string(x) == string(y)
}

// ContinuationControl is the optional loop integration boundary. Commit must
// acknowledge the exact snapshot before returning nil. Applications normally use
// agent.Durability, which adds revision ownership and workflow aggregation.
// The loop invokes Commit serially and supplies detached values.
type ContinuationControl struct {
	Revision                    func() uint64
	RunID, ExecutionID, StageID string
	Initial                     *Continuation
	Commit                      func(context.Context, Continuation) error
	PauseRequested              func() bool
	PauseSignal                 <-chan struct{}
	DeferredApprovals           bool
}

type RecoveryRequiredError struct{ Invocations []InvocationRecord }

func (e *RecoveryRequiredError) Error() string {
	return fmt.Sprintf("%v: %d unresolved invocation(s)", ErrRecoveryRequired, len(e.Invocations))
}
func (e *RecoveryRequiredError) Unwrap() error { return ErrRecoveryRequired }

type ToolRecoveryAction string

const (
	UseVerifiedResult  ToolRecoveryAction = "use_verified_result"
	ReplayInvocation   ToolRecoveryAction = "replay"
	ConfirmNotExecuted ToolRecoveryAction = "not_executed"
)

// ToolRecoveryResolution is an explicit application assertion. Replay requires
// SafeReplay, never merely the descriptive ToolTraits.Idempotent flag. Result is
// already processed/redacted canonical output verified by the application.
type ToolRecoveryResolution struct {
	InvocationID string             `json:"invocation_id"`
	Action       ToolRecoveryAction `json:"action"`
	SafeReplay   bool               `json:"safe_replay,omitempty"`
	Result       *ai.ToolResult     `json:"result,omitempty"`
	Execution    ToolExecution      `json:"execution"`
}

// ResolveRecovery applies explicit resolutions to uncertain invocations only.
func (c *Continuation) ResolveRecovery(resolutions []ToolRecoveryResolution) error {
	seen := map[string]bool{}
	for _, resolution := range resolutions {
		if seen[resolution.InvocationID] {
			return ErrContinuation
		}
		seen[resolution.InvocationID] = true
		index := -1
		if c.Pending != nil {
			index = slices.IndexFunc(c.Pending.Invocations, func(v InvocationRecord) bool { return v.ID == resolution.InvocationID })
		}
		if index < 0 {
			return ErrContinuation
		}
		inv := &c.Pending.Invocations[index]
		if inv.Disposition != InvocationDispatched && !(inv.Disposition == InvocationCompleted && inv.ProcessingFailed) {
			return ErrContinuation
		}
		switch resolution.Action {
		case UseVerifiedResult:
			if resolution.Result == nil || resolution.Result.ToolCallID != inv.Call.ID || resolution.Result.Name != inv.Call.Name || resolution.Execution.State == ToolRunning || resolution.Execution.State == "" {
				return ErrContinuation
			}
			switch resolution.Execution.State {
			case ToolNotStarted, ToolSucceeded, ToolFailed, ToolTimedOut, ToolCanceled:
			default:
				return ErrContinuation
			}
			if err := (ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: resolution.Result}}}).Validate(); err != nil {
				return fmt.Errorf("%w: %w", ErrContinuation, err)
			}
			inv.Result = resolution.Result
			inv.Execution = resolution.Execution
			inv.ProcessingFailed = false
			inv.Disposition = InvocationCompleted
		case ReplayInvocation, ConfirmNotExecuted:
			if resolution.Action == ReplayInvocation && !resolution.SafeReplay {
				return ErrContinuation
			}
			inv.Disposition = InvocationPlanned
			inv.Result = nil
			inv.ProcessingFailed = false
			inv.Execution = ToolExecution{State: ToolNotStarted}
		default:
			return ErrContinuation
		}
	}
	var unknown []InvocationRecord
	if c.Pending != nil {
		for _, inv := range c.Pending.Invocations {
			if inv.Disposition == InvocationDispatched || inv.ProcessingFailed {
				unknown = append(unknown, inv)
			}
		}
	}
	if len(unknown) > 0 {
		return &RecoveryRequiredError{Invocations: unknown}
	}
	return nil
}

func (c *Continuation) ApplyApprovals(decisions []ToolApprovalDecision) error {
	seen := map[string]bool{}
	for _, decision := range decisions {
		if seen[decision.RequestID] {
			return ErrContinuation
		}
		seen[decision.RequestID] = true
		found := false
		if c.Pending != nil {
			for i := range c.Pending.Invocations {
				inv := &c.Pending.Invocations[i]
				if inv.Approval != nil && inv.Approval.ID == decision.RequestID {
					if inv.ApprovalDecision != nil && !sameJSON(*inv.ApprovalDecision, decision) {
						return ErrContinuation
					}
					d := decision
					inv.ApprovalDecision = &d
					found = true
				}
			}
		}
		if !found {
			return ErrContinuation
		}
	}
	return nil
}

// CancelUnstarted resolves only calls whose dispatch was never authorized.
func (c *Continuation) CancelUnstarted() error {
	if c.Pending == nil {
		return nil
	}
	for i := range c.Pending.Invocations {
		inv := &c.Pending.Invocations[i]
		if inv.Disposition == InvocationDispatched || inv.ProcessingFailed {
			return &RecoveryRequiredError{Invocations: []InvocationRecord{*inv}}
		}
		if inv.Disposition != InvocationPlanned {
			continue
		}
		inv.Disposition = InvocationCompleted
		inv.Execution = ToolExecution{State: ToolNotStarted, Output: ToolOutputAccepted, Decision: ToolDecision{Action: ToolDeny, Code: "steering_canceled"}}
		inv.Result = &ai.ToolResult{ToolCallID: inv.Call.ID, Name: inv.Call.Name, Parts: ai.TextParts("Not executed: canceled by steering."), IsError: true}
	}
	return nil
}

func (c Continuation) Iterations() []Iteration {
	result := make([]Iteration, len(c.Generations))
	for i, g := range c.Generations {
		result[i] = g.iteration()
	}
	return result
}
func (g Generation) iteration() Iteration {
	it := Iteration{Count: g.Iteration, Conversation: ai.CloneMessages(g.Messages), inputMessages: g.InputMessages, Usage: g.Usage, UsageReported: g.UsageReported}
	for _, inv := range g.Invocations {
		call := inv.Call.Clone()
		it.Parts = append(it.Parts, IterationPart{Type: IterationTypeToolCall, ToolReq: &call, ToolResp: restoredResult(inv.Result), ToolExecution: cloneToolExecution(&inv.Execution)})
	}
	return it
}
func restoredResult(r *ai.ToolResult) *ToolResult {
	if r == nil {
		return nil
	}
	v := &ToolResult{Text: r.Text()}
	if r.IsError {
		v.Err = errors.New(r.Text())
	}
	return v
}
func generationFromIteration(id string, attempt int, it Iteration) Generation {
	return Generation{ID: id, Iteration: it.Count, Attempt: attempt, Messages: ai.CloneMessages(it.Conversation), InputMessages: it.inputMessages, Usage: it.Usage, UsageReported: it.UsageReported}
}

type durableLoop struct {
	control      *ContinuationControl
	state        Continuation
	generationID string
	failed       error
	events       chan<- Event
}

func (d *durableLoop) commit(ctx context.Context) error {
	if d.failed != nil {
		return d.failed
	}
	if err := d.state.Validate(); err != nil {
		return err
	}
	if err := d.control.Commit(ctx, d.state.Clone()); err != nil {
		d.failed = fmt.Errorf("%w: %w", ErrCheckpointCommit, err)
		return d.failed
	}
	if d.events != nil {
		event := Event{Type: EventCheckpointCommitted}
		if d.control.Revision != nil {
			event.CheckpointRevision = d.control.Revision()
		}
		return sendEvent(ctx, d.events, event)
	}
	return nil
}
func (d *durableLoop) pausing() bool {
	return d != nil && d.control.PauseRequested != nil && d.control.PauseRequested()
}
func (d *durableLoop) identity() Invocation {
	return Invocation{RunID: d.control.RunID, ExecutionID: d.control.ExecutionID, StageID: d.control.StageID, GenerationID: d.generationID}
}
func newIdentity() string { return rand.Text() }
