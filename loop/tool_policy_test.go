package loop

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/internal/obstest"
)

func TestToolPolicyDefaultsPrecedenceAndSnapshot(t *testing.T) {
	rules := ToolPolicyRules{Allow: []string{"read", "both"}, Deny: []string{"both"}, RequireApproval: []string{"write"}, ApprovalEffects: []ToolEffect{ToolEffectDestructive}}
	policy, err := NewToolPolicy(rules)
	if err != nil {
		t.Fatal(err)
	}
	rules.Allow[0] = "mutated"
	rules.Deny[0] = "mutated"
	for _, tc := range []struct {
		name   string
		effect ToolEffect
		want   ToolAction
	}{
		{"read", ToolEffectUnknown, ToolAllow}, {"both", ToolEffectUnknown, ToolDeny}, {"write", ToolEffectUnknown, ToolRequireApproval}, {"missing", ToolEffectUnknown, ToolDeny}, {"read", ToolEffectDestructive, ToolRequireApproval},
	} {
		decision, err := policy.BeforeTool(t.Context(), ToolPolicyInput{Call: ai.ToolCall{Name: tc.name}, Traits: ToolTraits{Effect: tc.effect}})
		if err != nil || decision.Action != tc.want {
			t.Fatalf("%s: %#v %v", tc.name, decision, err)
		}
	}
}

func TestPolicyChainsUseIndependentCalls(t *testing.T) {
	call := ai.ToolCall{ID: "1", Name: "test", Type: "function", Args: json.RawMessage(`{}`), Extensions: []ai.Extension{{Data: json.RawMessage(`"state"`)}}}
	mutator := ToolPolicyFunc(func(_ context.Context, input ToolPolicyInput) (ToolDecision, error) {
		input.Call.Args[0] = '!'
		input.Call.Extensions[0].Data[0] = '!'
		return ToolDecision{Action: ToolRequireApproval}, nil
	})
	verifier := ToolPolicyFunc(func(_ context.Context, input ToolPolicyInput) (ToolDecision, error) {
		if string(input.Call.Args) != "{}" || string(input.Call.Extensions[0].Data) != `"state"` {
			t.Fatal("policy chain aliases input")
		}
		return ToolDecision{Action: ToolDeny}, nil
	})
	chain, err := ChainToolPolicies(mutator, verifier)
	if err != nil {
		t.Fatal(err)
	}
	decision, err := chain.BeforeTool(t.Context(), ToolPolicyInput{Call: call})
	if err != nil || decision.Action != ToolDeny {
		t.Fatalf("decision=%#v %v", decision, err)
	}
}

func TestPolicyDenialAndApprovalDoNotInvoke(t *testing.T) {
	for _, action := range []ToolAction{ToolDeny, ToolRequireApproval} {
		var invoked atomic.Int32
		tool := schedulerTool(t, "test", ToolOptions{}, func(context.Context, ai.ToolCall) (string, error) { invoked.Add(1); return "private", nil })
		l := &Loop{Tools: []Tool{tool}, ToolPolicy: ToolPolicyFunc(func(context.Context, ToolPolicyInput) (ToolDecision, error) {
			return ToolDecision{Action: action, Reason: "safe refusal"}, nil
		})}
		iteration, calls := schedulerCalls("test")
		events := make(chan Event, 8)
		if err := l.executeToolCalls(t.Context(), iteration, calls, l.Tools, events, 1, 1, 0); err != nil {
			t.Fatal(err)
		}
		if invoked.Load() != 0 {
			t.Fatal("refused tool invoked")
		}
		close(events)
		for event := range events {
			if event.Type == EventToolStart {
				t.Fatal("refused tool emitted start")
			}
		}
		execution := iteration.Parts[0].ToolExecution
		if execution.State != ToolNotStarted || execution.Decision.Action != action || iteration.Parts[0].ToolResp.String() != "safe refusal" {
			t.Fatalf("record=%#v", execution)
		}
	}
}

func TestPolicyFailsClosedForWholeBatch(t *testing.T) {
	for _, policy := range []ToolPolicy{
		ToolPolicyFunc(func(context.Context, ToolPolicyInput) (ToolDecision, error) { return ToolDecision{}, nil }),
		ToolPolicyFunc(func(context.Context, ToolPolicyInput) (ToolDecision, error) { panic("private") }),
		ToolPolicyFunc(func(_ context.Context, input ToolPolicyInput) (ToolDecision, error) {
			if input.Call.ID == "1" {
				return ToolDecision{}, errors.New("policy offline")
			}
			return ToolDecision{Action: ToolAllow}, nil
		}),
	} {
		var invoked atomic.Int32
		tool := schedulerTool(t, "test", ToolOptions{}, func(context.Context, ai.ToolCall) (string, error) { invoked.Add(1); return "", nil })
		l := &Loop{Tools: []Tool{tool}, ToolPolicy: policy}
		iteration, calls := schedulerCalls("test", "test")
		if err := l.executeToolCalls(t.Context(), iteration, calls, l.Tools, nil, 1, 1, 0); err == nil {
			t.Fatal("invalid policy allowed batch")
		}
		if invoked.Load() != 0 {
			t.Fatal("batch started before all policy checks succeeded")
		}
	}
}

func TestExecutionSuccessSurvivesRejectedOutputAndProcessorPanic(t *testing.T) {
	for _, panicProcessor := range []bool{false, true} {
		tool := schedulerTool(t, "write", ToolOptions{}, func(context.Context, ai.ToolCall) (string, error) { return "write completed", nil })
		l := &Loop{Tools: []Tool{tool}, ToolResultProcessor: ToolResultProcessorFunc(func(context.Context, ToolPolicyInput, ToolResult) (ToolResult, error) {
			if panicProcessor {
				panic("private")
			}
			return RejectToolResult("withheld"), nil
		})}
		iteration, calls := schedulerCalls("write")
		err := l.executeToolCalls(t.Context(), iteration, calls, l.Tools, nil, 1, 1, 0)
		if (err != nil) != panicProcessor {
			t.Fatalf("error=%v", err)
		}
		execution := iteration.Parts[0].ToolExecution
		if execution.State != ToolSucceeded || execution.Output != ToolOutputRejected {
			t.Fatalf("execution=%#v", execution)
		}
	}
}

func TestBuiltInFiltersProtectAllOuterResults(t *testing.T) {
	for _, handlerError := range []bool{false, true} {
		recorder := obstest.Install(t)
		secret := errors.New("secret-marker")
		tool := schedulerTool(t, "test", ToolOptions{}, func(context.Context, ai.ToolCall) (string, error) {
			if handlerError {
				return "", secret
			}
			return secret.Error(), nil
		})
		redactor, _ := RedactToolResult(func(_ context.Context, text string) (string, error) {
			return strings.ReplaceAll(text, "secret-marker", "safe"), nil
		})
		limiter, _ := LimitToolResultBytes(4)
		processor, _ := ChainToolResultProcessors(redactor, limiter)
		l := &Loop{Tools: []Tool{tool}, ToolResultProcessor: processor}
		iteration, calls := schedulerCalls("test")
		events := make(chan Event, 8)
		ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{ToolOutput: gai.CaptureEnabled})
		if err := l.executeToolCalls(ctx, iteration, calls, l.Tools, events, 1, 1, 0); err != nil {
			t.Fatal(err)
		}
		close(events)
		for event := range events {
			if event.ToolResult != nil && (strings.Contains(event.ToolResult.String(), "secret-marker") || errors.Is(event.ToolResult.Err, secret)) {
				t.Fatal("raw error/output escaped")
			}
		}
		if got := iteration.Conversation[0].Parts[0].ToolResult.Text(); got != "safe" {
			t.Fatalf("conversation=%q", got)
		}
		execution := iteration.Parts[0].ToolExecution
		if execution.Output != ToolOutputChanged {
			t.Fatalf("output=%s", execution.Output)
		}
		if handlerError && execution.State != ToolFailed {
			t.Fatal("handler failure lost")
		}
		if !handlerError && execution.State != ToolSucceeded {
			t.Fatal("handler success lost")
		}
		for _, span := range requireToolSpans(t, recorder, 1) {
			if strings.Contains(toolSpanText(span), "secret-marker") {
				t.Fatal("secret escaped telemetry")
			}
		}
	}
}

func TestByteLimitRejectsRatherThanTruncatesPayload(t *testing.T) {
	for _, maxBytes := range []int{1, 10, 64} {
		limit, _ := LimitToolResultBytes(maxBytes)
		for _, r := range []ToolResult{{Text: strings.Repeat("x", 100)}, {Err: errors.New(strings.Repeat("x", 100))}} {
			result, err := limit.Process(t.Context(), ToolPolicyInput{}, r)
			if err != nil || !errors.Is(result.Err, ErrToolOutputLimit) || !errors.Is(result.Err, ErrToolResultRejected) || len(result.String()) > maxBytes {
				t.Fatalf("limit %d: %#v %v", maxBytes, result, err)
			}
		}
	}
}

func TestProcessorChainsCloneCalls(t *testing.T) {
	first := ToolResultProcessorFunc(func(_ context.Context, input ToolPolicyInput, r ToolResult) (ToolResult, error) {
		input.Call.Args[0] = '!'
		return r, nil
	})
	second := ToolResultProcessorFunc(func(_ context.Context, input ToolPolicyInput, r ToolResult) (ToolResult, error) {
		if string(input.Call.Args) != "{}" {
			t.Fatal("processor input alias")
		}
		return r, nil
	})
	chain, _ := ChainToolResultProcessors(first, second)
	if _, err := chain.Process(t.Context(), ToolPolicyInput{Call: ai.ToolCall{Args: json.RawMessage(`{}`)}}, ToolResult{Text: "ok"}); err != nil {
		t.Fatal(err)
	}
}

func TestAllowPolicyChainRetainsDecisionProvenance(t *testing.T) {
	original := ToolDecision{Action: ToolAllow, Code: "allowlist", Reason: "registered reader"}
	policy, _ := ChainToolPolicies(ToolPolicyFunc(func(context.Context, ToolPolicyInput) (ToolDecision, error) { return original, nil }))
	decision, err := policy.BeforeTool(t.Context(), ToolPolicyInput{})
	if err != nil || decision != original {
		t.Fatalf("decision=%#v %v", decision, err)
	}
}

func TestDeadlineBeforeInvocationRemainsNotStarted(t *testing.T) {
	for range 50 {
		invoked := false
		tool := schedulerTool(t, "test", ToolOptions{}, func(context.Context, ai.ToolCall) (string, error) { invoked = true; return "ok", nil })
		execution := ToolExecution{State: ToolNotStarted}
		_, _, err := processObservedToolExecution(t.Context(), ToolPolicyInput{Call: ai.ToolCall{ID: "1", Name: "test", Type: "function", Args: json.RawMessage(`{}`)}}, []Tool{tool}, nil, time.Nanosecond, &execution)
		if err != nil {
			t.Fatal(err)
		}
		if !invoked && execution.State != ToolNotStarted {
			t.Fatalf("uninvoked state=%s", execution.State)
		}
	}
}
