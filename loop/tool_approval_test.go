package loop

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func requireApprovalPolicy() ToolPolicy {
	return ToolPolicyFunc(func(context.Context, ToolPolicyInput) (ToolDecision, error) {
		return ToolDecision{Action: ToolRequireApproval, Code: "review", Reason: "approval needed"}, nil
	})
}

func TestApprovalPreflightPrecedesAdmissionAndOwnsSnapshots(t *testing.T) {
	var policies, approvals, invoked atomic.Int32
	guard := &ToolGuard{}
	tool := schedulerTool(t, "write", ToolOptions{Guard: guard}, func(_ context.Context, call ai.ToolCall) (string, error) {
		if approvals.Load() != 2 {
			t.Error("handler started before all approvals")
		}
		if string(call.Args) != "{}" || string(call.Extensions[0].Data) != `"original"` {
			t.Error("resolver changed executable input")
		}
		invoked.Add(1)
		return "ok", nil
	})
	l := &Loop{Tools: []Tool{tool}, ToolPolicy: ToolPolicyFunc(func(context.Context, ToolPolicyInput) (ToolDecision, error) {
		policies.Add(1)
		return ToolDecision{Action: ToolRequireApproval}, nil
	}), ToolApprovalResolver: ToolApprovalResolverFunc(func(_ context.Context, request ToolApprovalRequest) (ToolApprovalDecision, error) {
		if policies.Load() != 2 || invoked.Load() != 0 {
			t.Error("incorrect preflight order")
		}
		release, _ := guard.tryAcquire()
		if release == nil {
			t.Fatal("approval held tool guard")
		}
		release()
		request.Input.Call.Args[0] = '!'
		request.Input.Call.Extensions[0].Data[0] = '!'
		request.Input.Call.Name = "changed"
		approvals.Add(1)
		return ToolApprovalDecision{RequestID: request.ID, Approved: true}, nil
	})}
	iteration, calls := schedulerCalls("write", "write")
	for i := range calls {
		calls[i].call.Extensions = []ai.Extension{{Data: json.RawMessage(`"original"`)}}
	}
	events := make(chan Event, 20)
	if err := l.executeToolCalls(t.Context(), iteration, calls, l.Tools, events, 1, 1, 0); err != nil {
		t.Fatal(err)
	}
	close(events)
	ids := map[string]bool{}
	resolved := 0
	for event := range events {
		switch event.Type {
		case EventToolApprovalRequested:
			request := event.ToolApproval
			if request == nil || ids[request.ID] || request.ID == "" {
				t.Fatalf("bad approval request: %#v", request)
			}
			ids[request.ID] = true
			if string(request.Input.Call.Args) != "{}" || string(request.Input.Call.Extensions[0].Data) != `"original"` {
				t.Fatal("request snapshot changed")
			}
			request.Input.Call.Args[0] = 'X'
			request.Input.Call.Extensions[0].Data[0] = 'X'
		case EventToolApprovalResolved:
			resolved++
			if string(event.ToolApproval.Input.Call.Args) != "{}" || string(event.ToolApproval.Input.Call.Extensions[0].Data) != `"original"` {
				t.Fatal("events share request snapshot")
			}
			if event.ToolExecution.Decision.Action != ToolRequireApproval || event.ToolExecution.Approval.Action != ToolAllow {
				t.Fatalf("lost provenance: %#v", event.ToolExecution)
			}
		case EventToolStart:
			if resolved != 2 {
				t.Fatal("start before approval events")
			}
		}
	}
	if invoked.Load() != 2 || len(ids) != 2 {
		t.Fatal("approved calls did not execute")
	}
	for _, part := range iteration.Parts {
		if part.ToolExecution.State != ToolSucceeded || !ids[part.ToolExecution.ApprovalID] {
			t.Fatalf("record: %#v", part.ToolExecution)
		}
	}
}

func TestApprovalDenialIsPerCall(t *testing.T) {
	var invoked atomic.Int32
	tool := schedulerTool(t, "write", ToolOptions{}, func(context.Context, ai.ToolCall) (string, error) { invoked.Add(1); return "ok", nil })
	l := &Loop{Tools: []Tool{tool}, ToolPolicy: requireApprovalPolicy(), ToolApprovalResolver: ToolApprovalResolverFunc(func(_ context.Context, r ToolApprovalRequest) (ToolApprovalDecision, error) {
		return ToolApprovalDecision{RequestID: r.ID, Approved: r.Input.Call.ID == "1", Reason: "reviewed refusal"}, nil
	})}
	iteration, calls := schedulerCalls("write", "write")
	if err := l.executeToolCalls(t.Context(), iteration, calls, l.Tools, nil, 1, 1, 0); err != nil {
		t.Fatal(err)
	}
	if invoked.Load() != 1 || !errors.Is(iteration.Parts[0].ToolResp.Err, ErrToolDenied) || iteration.Parts[0].ToolExecution.State != ToolNotStarted || iteration.Parts[1].ToolExecution.State != ToolSucceeded {
		t.Fatalf("unexpected results: %#v", iteration.Parts)
	}
	if iteration.Conversation[0].Parts[0].ToolResult.Text() != "reviewed refusal" {
		t.Fatal("refusal missing from model transcript")
	}
}

func TestApprovalFailureAbortsWholeBatch(t *testing.T) {
	for _, mode := range []string{"error", "panic", "wrong_approve_id", "wrong_deny_id", "canceled"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			var invoked atomic.Int32
			tool := schedulerTool(t, "write", ToolOptions{}, func(context.Context, ai.ToolCall) (string, error) { invoked.Add(1); return "", nil })
			l := &Loop{Tools: []Tool{tool}, ToolPolicy: requireApprovalPolicy(), ToolApprovalResolver: ToolApprovalResolverFunc(func(_ context.Context, r ToolApprovalRequest) (ToolApprovalDecision, error) {
				if r.Input.Call.ID == "0" {
					return ToolApprovalDecision{RequestID: r.ID, Approved: true}, nil
				}
				switch mode {
				case "error":
					return ToolApprovalDecision{}, errors.New("offline")
				case "panic":
					panic("private panic")
				case "wrong_approve_id":
					return ToolApprovalDecision{RequestID: "stale", Approved: true}, nil
				case "wrong_deny_id":
					return ToolApprovalDecision{RequestID: "stale"}, nil
				case "canceled":
					cancel()
					return ToolApprovalDecision{RequestID: r.ID, Approved: true}, nil
				}
				panic("unreachable")
			})}
			iteration, calls := schedulerCalls("write", "write")
			events := make(chan Event, 16)
			err := l.executeToolCalls(ctx, iteration, calls, l.Tools, events, 1, 1, 0)
			close(events)
			var requested, resolved int
			for event := range events {
				switch event.Type {
				case EventToolApprovalRequested:
					requested++
				case EventToolApprovalResolved:
					resolved++
					if requested != resolved || event.ToolExecution.ApprovalID != event.ToolApproval.ID {
						t.Fatal("approval outcome lost ordering or correlation")
					}
					if resolved == 2 && (event.Err == nil || event.ToolExecution.Approval.Action != ToolDeny) {
						t.Fatalf("missing failure outcome: %#v", event)
					}
				}
			}
			if requested != 2 || resolved != 2 {
				t.Fatalf("requested=%d resolved=%d", requested, resolved)
			}
			wantCode := "approval_failed"
			if mode == "canceled" {
				wantCode = "approval_canceled"
			}
			if iteration.Parts[1].ToolExecution.Approval.Code != wantCode {
				t.Fatal("failure absent from retained execution")
			}
			if err == nil || invoked.Load() != 0 {
				t.Fatalf("err=%v invoked=%d", err, invoked.Load())
			}
			if mode == "canceled" && !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation lost: %v", err)
			}
			for _, part := range iteration.Parts {
				if part.ToolExecution.State != ToolNotStarted {
					t.Fatal("failed preflight shows started call")
				}
			}
		})
	}
}

func TestApprovalIDsCannotBeReusedAcrossRuns(t *testing.T) {
	var firstID string
	var invoked atomic.Int32
	tool := schedulerTool(t, "write", ToolOptions{}, func(context.Context, ai.ToolCall) (string, error) { invoked.Add(1); return "ok", nil })
	l := &Loop{Tools: []Tool{tool}, ToolPolicy: requireApprovalPolicy(), ToolApprovalResolver: ToolApprovalResolverFunc(func(_ context.Context, r ToolApprovalRequest) (ToolApprovalDecision, error) {
		if firstID == "" {
			firstID = r.ID
		}
		return ToolApprovalDecision{RequestID: firstID, Approved: true}, nil
	})}
	for run := range 2 {
		iteration, calls := schedulerCalls("write")
		err := l.executeToolCalls(t.Context(), iteration, calls, l.Tools, nil, 1, 1, 0)
		if (run == 0 && err != nil) || (run == 1 && !errors.Is(err, ErrToolApproval)) {
			t.Fatalf("run%d: %v", run, err)
		}
	}
	if invoked.Load() != 1 {
		t.Fatal("stale approval invoked tool")
	}
}

func TestMissingApprovalResolverProducesResolvedRefusal(t *testing.T) {
	tool := schedulerTool(t, "write", ToolOptions{}, func(context.Context, ai.ToolCall) (string, error) {
		t.Error("unapproved handler called")
		return "", nil
	})
	l := &Loop{Tools: []Tool{tool}, ToolPolicy: requireApprovalPolicy()}
	iteration, calls := schedulerCalls("write")
	events := make(chan Event, 8)
	if err := l.executeToolCalls(t.Context(), iteration, calls, l.Tools, events, 1, 1, 0); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(iteration.Parts[0].ToolResp.Err, ErrToolApprovalRequired) || iteration.Parts[0].ToolExecution.Approval.Code != "approval_unavailable" {
		t.Fatal("missing resolver did not fail closed")
	}
	close(events)
	var resolved bool
	for event := range events {
		if event.Type == EventToolApprovalResolved {
			resolved = true
		}
	}
	if !resolved {
		t.Fatal("approval event left apparently pending")
	}
}

func TestApprovalCancellationDoesNotBlockOnFullEventBuffer(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	l := &Loop{ToolApprovalResolver: ToolApprovalResolverFunc(func(context.Context, ToolApprovalRequest) (ToolApprovalDecision, error) {
		cancel()
		return ToolApprovalDecision{}, context.Canceled
	})}
	tasks := []scheduledTool{{execution: ToolExecution{State: ToolNotStarted, Decision: ToolDecision{Action: ToolRequireApproval}}, pending: pendingToolCall{call: ai.ToolCall{ID: "call", Name: "write", Type: "function", Args: []byte(`{}`)}}}}
	events := make(chan Event, 1)
	if err := l.preflightToolApprovals(ctx, tasks, events, 1, 1, 0); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if len(events) != 1 || tasks[0].execution.Approval.Code != "approval_canceled" || tasks[0].execution.ApprovalID == "" {
		t.Fatal("canceled approval lost its retained outcome")
	}
	request := <-events
	if request.ToolExecution.Approval.Action != "" {
		t.Fatal("request event aliases later failure decision")
	}
}
