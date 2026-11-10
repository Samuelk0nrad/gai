package loop

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput"
	"github.com/Samuelk0nrad/gai/testutil/mocks"
)

// continuationForTest has one accepted generation and a pending plan whose
// single call was dispatched without a known outcome.
func continuationForTest() Continuation {
	prompt := modelinput.Snapshot{Messages: []ai.Message{{Role: ai.RoleSystem, Parts: ai.TextParts("system")}, {Role: ai.RoleUser, Parts: ai.TextParts("question")}}}
	call := ai.ToolCall{Type: "function", ID: "call-1", Name: "write", Args: json.RawMessage(`{}`)}
	return Continuation{
		Version:       ContinuationVersion,
		Phase:         PhaseTools,
		Prompt:        &prompt,
		NextIteration: 2,
		NextAttempt:   1,
		Generations:   []Generation{{ID: "g1", Iteration: 1, Attempt: 1, Messages: []ai.Message{{Role: ai.RoleAssistant, Parts: ai.TextParts("thinking")}}}},
		Pending: &Generation{ID: "g2", Iteration: 2, Attempt: 1, Messages: []ai.Message{{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &call}}}}, Invocations: []InvocationRecord{
			{ID: "i1", Call: call.Clone(), Disposition: InvocationDispatched, Attempt: 1, Execution: ToolExecution{State: ToolRunning}},
		}},
		Usage: []GenerationUsage{{ID: "g1", Usage: ai.Usage{InputTokens: 3}, Reported: true}},
	}
}

func TestContinuationValidateRejectsInvalidState(t *testing.T) {
	if err := continuationForTest().Validate(); err != nil {
		t.Fatal(err)
	}
	result := &ai.ToolResult{ToolCallID: "call-1", Name: "write", Parts: ai.TextParts("ok")}
	cases := map[string]func(*Continuation){
		"version":              func(c *Continuation) { c.Version = 0 },
		"next iteration":       func(c *Continuation) { c.NextIteration = 0 },
		"next attempt":         func(c *Continuation) { c.NextAttempt = 0 },
		"retry count":          func(c *Continuation) { c.RetryCount = -1 },
		"elapsed":              func(c *Continuation) { c.Elapsed = -time.Second },
		"phase":                func(c *Continuation) { c.Phase = "unknown" },
		"tools without plan":   func(c *Continuation) { c.Pending = nil },
		"plan outside tools":   func(c *Continuation) { c.Phase = PhaseReady },
		"missing prompt":       func(c *Continuation) { c.Prompt = nil },
		"invalid prompt":       func(c *Continuation) { c.Prompt = &modelinput.Snapshot{} },
		"invalid steering":     func(c *Continuation) { c.Steering = []ai.Message{{}} },
		"duplicate generation": func(c *Continuation) { c.Pending.ID = "g1" },
		"empty generation id":  func(c *Continuation) { c.Generations[0].ID = "" },
		"input messages":       func(c *Continuation) { c.Generations[0].InputMessages = 2 },
		"invalid message":      func(c *Continuation) { c.Generations[0].Messages = []ai.Message{{}} },
		"invocation count":     func(c *Continuation) { c.Pending.Invocations = nil },
		"invocation id":        func(c *Continuation) { c.Pending.Invocations[0].ID = "g1" },
		"call mismatch":        func(c *Continuation) { c.Pending.Invocations[0].Call.Name = "other" },
		"disposition":          func(c *Continuation) { c.Pending.Invocations[0].Disposition = "unknown" },
		"dispatched result":    func(c *Continuation) { c.Pending.Invocations[0].Result = result },
		"dispatched attempt":   func(c *Continuation) { c.Pending.Invocations[0].Attempt = 0 },
		"planned result": func(c *Continuation) {
			c.Pending.Invocations[0].Disposition = InvocationPlanned
			c.Pending.Invocations[0].Result = result
		},
		"completed without result": func(c *Continuation) { c.Pending.Invocations[0].Disposition = InvocationCompleted },
		"result pairing": func(c *Continuation) {
			c.Pending.Invocations[0].Disposition = InvocationCompleted
			c.Pending.Invocations[0].Result = &ai.ToolResult{ToolCallID: "other", Name: "write", Parts: ai.TextParts("ok")}
		},
		"approval call": func(c *Continuation) {
			c.Pending.Invocations[0].Approval = &ToolApprovalRequest{ID: "a1", Input: ToolPolicyInput{Call: ai.ToolCall{Type: "function", ID: "other", Name: "write"}}}
		},
		"steering flag on open call": func(c *Continuation) { c.Pending.Invocations[0].CanceledBySteering = true },
		"decision without approval": func(c *Continuation) {
			c.Pending.Invocations[0].ApprovalDecision = &ToolApprovalDecision{RequestID: "a1", Approved: true}
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			c := continuationForTest()
			mutate(&c)
			if err := c.Validate(); !errors.Is(err, ErrContinuation) {
				t.Fatal(err)
			}
		})
	}
}

func TestContinuationIncrementalValidationChecksOnlyNewHistory(t *testing.T) {
	c := continuationForTest()
	// Committed generations are trusted, but their identities stay unique.
	c.Generations[0].Messages = []ai.Message{{}}
	if err := c.validate(1, false); err != nil {
		t.Fatal("trusted generation validated again", err)
	}
	c.Pending.ID = "g1"
	if err := c.validate(1, false); !errors.Is(err, ErrContinuation) {
		t.Fatal("duplicate identity accepted", err)
	}
	c = continuationForTest()
	c.Pending.Messages = []ai.Message{{}}
	if err := c.validate(1, false); !errors.Is(err, ErrContinuation) {
		t.Fatal("new pending plan not validated", err)
	}
}

func TestContinuationRecoveryResolutions(t *testing.T) {
	c := continuationForTest()
	err := c.ResolveRecovery(nil)
	var required *RecoveryRequiredError
	if !errors.As(err, &required) || !errors.Is(err, ErrRecoveryRequired) || len(required.Invocations) != 1 || required.Error() == "" {
		t.Fatal(err)
	}
	invalid := map[string][]ToolRecoveryResolution{
		"unknown invocation": {{InvocationID: "missing", Action: ConfirmNotExecuted}},
		"duplicate":          {{InvocationID: "i1", Action: ConfirmNotExecuted}, {InvocationID: "i1", Action: ConfirmNotExecuted}},
		"unsafe replay":      {{InvocationID: "i1", Action: ReplayInvocation}},
		"unknown action":     {{InvocationID: "i1", Action: "guess"}},
		"missing result":     {{InvocationID: "i1", Action: UseVerifiedResult, Execution: ToolExecution{State: ToolSucceeded}}},
		"result pairing": {{InvocationID: "i1", Action: UseVerifiedResult, Execution: ToolExecution{State: ToolSucceeded},
			Result: &ai.ToolResult{ToolCallID: "other", Name: "write", Parts: ai.TextParts("ok")}}},
		"running state": {{InvocationID: "i1", Action: UseVerifiedResult, Execution: ToolExecution{State: ToolRunning},
			Result: &ai.ToolResult{ToolCallID: "call-1", Name: "write", Parts: ai.TextParts("ok")}}},
	}
	for name, resolutions := range invalid {
		c := continuationForTest()
		if err := c.ResolveRecovery(resolutions); !errors.Is(err, ErrContinuation) {
			t.Errorf("%s: %v", name, err)
		}
	}

	c = continuationForTest()
	if err := c.ResolveRecovery([]ToolRecoveryResolution{{InvocationID: "i1", Action: ReplayInvocation, SafeReplay: true}}); err != nil {
		t.Fatal(err)
	}
	if inv := c.Pending.Invocations[0]; inv.Disposition != InvocationPlanned || inv.Attempt != 1 || inv.Execution.State != ToolNotStarted {
		t.Fatal("replay must keep the attempt count and reopen admission", inv)
	}

	c = continuationForTest()
	verified := &ai.ToolResult{ToolCallID: "call-1", Name: "write", Parts: ai.TextParts("verified")}
	if err := c.ResolveRecovery([]ToolRecoveryResolution{{InvocationID: "i1", Action: UseVerifiedResult, Result: verified, Execution: ToolExecution{State: ToolSucceeded}}}); err != nil {
		t.Fatal(err)
	}
	if inv := c.Pending.Invocations[0]; inv.Disposition != InvocationCompleted || inv.Result.Text() != "verified" {
		t.Fatal(inv)
	}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	// Only result processing failed, so the handler ran: it cannot be
	// confirmed as not executed, only replayed explicitly or verified.
	processed := func(state ToolExecutionState) Continuation {
		c := continuationForTest()
		inv := &c.Pending.Invocations[0]
		inv.Disposition, inv.ProcessingFailed, inv.Execution.State = InvocationCompleted, true, state
		return c
	}
	ran := processed(ToolSucceeded)
	if err := ran.ResolveRecovery([]ToolRecoveryResolution{{InvocationID: "i1", Action: ConfirmNotExecuted}}); !errors.Is(err, ErrContinuation) {
		t.Fatal("handler that ran confirmed as not executed", err)
	}
	ran = processed(ToolSucceeded)
	if err := ran.ResolveRecovery([]ToolRecoveryResolution{{InvocationID: "i1", Action: ReplayInvocation, SafeReplay: true}}); err != nil {
		t.Fatal(err)
	}
	refused := processed(ToolNotStarted) // a synthetic refusal never runs a handler
	if err := refused.ResolveRecovery([]ToolRecoveryResolution{{InvocationID: "i1", Action: ConfirmNotExecuted}}); err != nil {
		t.Fatal(err)
	}

	// A completed invocation is no longer open to recovery.
	if err := c.ResolveRecovery([]ToolRecoveryResolution{{InvocationID: "i1", Action: ConfirmNotExecuted}}); !errors.Is(err, ErrContinuation) {
		t.Fatal(err)
	}
}

func TestContinuationApprovalsAndCancelUnstarted(t *testing.T) {
	c := continuationForTest()
	if err := c.CancelUnstarted(); !errors.Is(err, ErrRecoveryRequired) {
		t.Fatal("dispatched call canceled", err)
	}
	inv := &c.Pending.Invocations[0]
	inv.Disposition = InvocationPlanned
	inv.Attempt = 0
	inv.Execution = ToolExecution{State: ToolNotStarted}
	inv.Approval = &ToolApprovalRequest{ID: "a1", Input: ToolPolicyInput{Call: inv.Call.Clone()}}
	if err := c.ApplyApprovals([]ToolApprovalDecision{{RequestID: "missing", Approved: true}}); !errors.Is(err, ErrContinuation) {
		t.Fatal(err)
	}
	if err := c.ApplyApprovals([]ToolApprovalDecision{{RequestID: "a1", Approved: true}, {RequestID: "a1", Approved: true}}); !errors.Is(err, ErrContinuation) {
		t.Fatal("duplicate decision accepted", err)
	}
	if err := c.ApplyApprovals([]ToolApprovalDecision{{RequestID: "a1", Approved: true}}); err != nil {
		t.Fatal(err)
	}
	if err := c.ApplyApprovals([]ToolApprovalDecision{{RequestID: "a1", Approved: true}}); err != nil {
		t.Fatal("identical decision rejected", err)
	}
	if err := c.ApplyApprovals([]ToolApprovalDecision{{RequestID: "a1", Approved: false}}); !errors.Is(err, ErrContinuation) {
		t.Fatal("conflicting decision accepted", err)
	}
	if err := c.CancelUnstarted(); err != nil {
		t.Fatal(err)
	}
	if inv.Disposition != InvocationCompleted || inv.Result == nil || !inv.Result.IsError || !inv.CanceledBySteering {
		t.Fatal(*inv)
	}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestContinuationCloneAndDetach(t *testing.T) {
	original := continuationForTest()
	before, _ := json.Marshal(original)
	clone := original.Clone()
	clone.Prompt.Messages[0].Parts[0].Text = "changed"
	clone.Generations[0].Messages[0].Parts[0].Text = "changed"
	clone.Pending.Invocations[0].Call.Args[0] = ' '
	clone.Usage[0].Reported = false
	after, _ := json.Marshal(original)
	if string(before) != string(after) {
		t.Fatal("clone shares state with the original")
	}

	// detach shares only committed history; the loop's mutable state is copied.
	detached := original.detach()
	original.Pending.Invocations[0].Disposition = InvocationCompleted
	original.Usage[0].Usage.InputTokens = 99
	original.Generations = append(original.Generations, Generation{ID: "g3"})
	original.Steering = append(original.Steering, ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("steer")})
	if detached.Pending.Invocations[0].Disposition != InvocationDispatched || detached.Usage[0].Usage.InputTokens != 3 || len(detached.Generations) != 1 || len(detached.Steering) != 0 {
		t.Fatal("detached commit value changed with loop state")
	}
	if &detached.Generations[0].Messages[0] != &original.Generations[0].Messages[0] {
		t.Fatal("committed history should be shared, not copied")
	}
}

func TestContinuationLimitError(t *testing.T) {
	l := &Loop{MaxLoopIterations: 5, RetryPolicy: &RetryPolicy{TotalTimeout: time.Minute, MaxRetries: 1}}
	c := continuationForTest()
	if err := c.LimitError(l); err != nil {
		t.Fatal(err)
	}
	c.Elapsed = time.Minute
	if err := c.LimitError(l); !errors.Is(err, ErrTimeoutExhausted) || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("used budget not reported", err)
	}
	if err := c.LimitError(&Loop{MaxLoopIterations: 5}); err != nil {
		t.Fatal("unlimited policy reported exhaustion", err)
	}

	// Only an attempt interrupted by a lost worker consumes retry budget.
	interrupted := continuationForTest()
	interrupted.Phase, interrupted.Pending, interrupted.RetryCount = PhaseGenerating, nil, 1
	if err := interrupted.LimitError(l); !errors.Is(err, ErrMaxRetries) {
		t.Fatal("exhausted retry budget not reported", err)
	}
	interrupted.RetryCount = 0
	if err := interrupted.LimitError(l); err != nil {
		t.Fatal(err)
	}
	// Without a retry policy no retry is allowed, as in ordinary runs.
	if err := interrupted.LimitError(&Loop{MaxLoopIterations: 5}); !errors.Is(err, ErrMaxRetries) {
		t.Fatal("nil retry policy allowed repeating an interrupted attempt", err)
	}

	// New model work needs an iteration; an accepted plan still runs its tools.
	planned := continuationForTest()
	planned.NextIteration = 6
	if err := planned.LimitError(l); err != nil {
		t.Fatal("accepted plan blocked by iteration limit", err)
	}
	ready := continuationForTest()
	ready.Phase, ready.Pending, ready.NextIteration = PhaseReady, nil, 6
	if err := ready.LimitError(l); !errors.Is(err, ErrMaxIterations) {
		t.Fatal("exhausted iterations not reported", err)
	}

	done := ready
	done.Phase, done.Elapsed = PhaseDone, 2*time.Minute
	if err := done.LimitError(l); err != nil {
		t.Fatal("finished continuation needs no limits", err)
	}
	done.Steering = []ai.Message{{Role: ai.RoleUser, Parts: ai.TextParts("more")}}
	if err := done.LimitError(l); err == nil {
		t.Fatal("steering needs execution time and an iteration")
	}
}

// Review findings 4 and 15: a reused Loop drops the previous durable state and
// gets its prompt builder back, and the durable run counter returns to zero.
func TestReusedLoopDropsDurableState(t *testing.T) {
	answer := func(text string) mocks.MockModelResponse {
		return mocks.MockModelResponse{Res: ai.AIResponse{Message: ai.Message{Role: ai.RoleAssistant, Parts: ai.TextParts(text)}}}
	}
	model := &mocks.MockModel{Responses: []mocks.MockModelResponse{answer("first"), answer("second")}}
	builder := modelinput.New(modelinput.Definition{SystemInstructions: []modelinput.Part{modelinput.NewTextPart("system")}, PromptInput: modelinput.PromptInput{User: ai.TextParts("question")}})
	l := New(model, nil, builder, nil)
	commits := 0
	l.Continuation = &ContinuationControl{RunID: "run", ExecutionID: "execution", StageID: "primary", Commit: func(context.Context, Continuation) error {
		commits++
		return nil
	}}
	last := func() Event {
		var event Event
		for e := range l.Run(t.Context()) {
			event = e
		}
		return event
	}
	if event := last(); event.Type != EventDone || commits == 0 {
		t.Fatal(event.Type, event.Err, commits)
	}
	if l.PromptBuilder != modelinput.PromptBuilder(builder) {
		t.Fatal("durable run kept its frozen prompt builder")
	}
	if n := durableRuns.Load(); n != 0 {
		t.Fatal("durable run counter not released", n)
	}
	l.Continuation, l.Iterations = nil, nil
	if event := last(); event.Type != EventDone || model.Count != 2 || l.durable != nil {
		t.Fatal("reused loop continued the previous durable run", event.Type, model.Count, l.durable != nil)
	}
}
