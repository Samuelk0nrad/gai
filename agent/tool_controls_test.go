package agent_test

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

func toolControlModel(names ...string) ai.Model {
	calls := make([]ai.Token, len(names))
	for i, name := range names {
		calls[i] = ai.Token{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: name + string(rune('a'+i)), Type: "function", Name: name, Args: []byte(`{}`)}}}
	}
	return nativeToolWorkflowModel{&scriptedWorkflowModel{scripts: [][]ai.Token{calls, {{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "done"}}}}}}
}
func controlTool(t *testing.T, fn loop.ToolFunc) loop.Tool {
	t.Helper()
	tool, err := loop.NewTool("test", "Test tool", ai.ToolParameters{}, fn)
	if err != nil {
		t.Fatal(err)
	}
	return tool
}
func controlDecision(action loop.ToolAction) loop.ToolPolicy {
	return loop.ToolPolicyFunc(func(context.Context, loop.ToolPolicyInput) (loop.ToolDecision, error) {
		return loop.ToolDecision{Action: action}, nil
	})
}
func controlApproval(allow bool) loop.ToolApprovalResolver {
	return loop.ToolApprovalResolverFunc(func(_ context.Context, r loop.ToolApprovalRequest) (loop.ToolApprovalDecision, error) {
		return loop.ToolApprovalDecision{RequestID: r.ID, Approved: allow}, nil
	})
}

func TestAgentToolPolicyAndApprovalOverrides(t *testing.T) {
	for _, tc := range []struct {
		name     string
		policy   agent.Optional[loop.ToolPolicy]
		resolver agent.Optional[loop.ToolApprovalResolver]
		invoke   bool
		approval bool
	}{
		{name: "inherit", invoke: true, approval: true},
		{name: "ignored values", policy: agent.Optional[loop.ToolPolicy]{Value: controlDecision(loop.ToolDeny)}, resolver: agent.Optional[loop.ToolApprovalResolver]{Value: controlApproval(false)}, invoke: true, approval: true},
		{name: "replace policy", policy: agent.Optional[loop.ToolPolicy]{Set: true, Value: controlDecision(loop.ToolDeny)}},
		{name: "clear policy", policy: agent.Optional[loop.ToolPolicy]{Set: true}, invoke: true},
		{name: "replace resolver", resolver: agent.Optional[loop.ToolApprovalResolver]{Set: true, Value: controlApproval(false)}, approval: true},
		{name: "clear resolver", resolver: agent.Optional[loop.ToolApprovalResolver]{Set: true}, approval: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var invoked atomic.Bool
			tool := controlTool(t, func(context.Context, ai.ToolCall) (string, error) { invoked.Store(true); return "ok", nil })
			a := agent.New(agent.Definition{Model: toolControlModel("test"), Prompt: executionPrompt, Tools: []loop.Tool{tool}, ToolPolicy: controlDecision(loop.ToolRequireApproval), ToolApprovalResolver: controlApproval(true)})
			workflow, err := a.NewRun(t.Context(), agent.RunInput{Execution: &agent.ExecutionOverrides{ToolPolicy: tc.policy, ToolApprovalResolver: tc.resolver}})
			if err != nil {
				t.Fatal(err)
			}
			var approvals int
			for event := range workflow.RunEvents(t.Context()) {
				if event.Type == agent.EventToolApprovalResolved {
					approvals++
					if event.ToolApproval == nil || event.ToolExecution.ApprovalID != event.ToolApproval.ID {
						t.Fatal("approval provenance missing")
					}
				}
			}
			result, err := workflow.Wait()
			if err != nil {
				t.Fatal(err)
			}
			if invoked.Load() != tc.invoke || (approvals > 0) != tc.approval {
				t.Fatalf("invoked=%v approvals=%d", invoked.Load(), approvals)
			}
			if tc.name == "clear resolver" {
				var found bool
				for _, part := range result.Primary.Iterations[0].Parts {
					if part.ToolResp != nil {
						found = errors.Is(part.ToolResp.Err, loop.ErrToolApprovalRequired)
					}
				}
				if !found {
					t.Fatal("clearing resolver bypassed approval")
				}
			}
		})
	}
}

func TestAgentToolConfigSnapshotsAtomicReplacementAndReset(t *testing.T) {
	for _, tc := range []struct {
		name         string
		patch        *loop.ToolExecutionConfig
		wantDeadline bool
		wantActive   int32
	}{
		{"inherit", nil, true, 1},
		{"replace", &loop.ToolExecutionConfig{MaxConcurrent: 2}, false, 2},
		{"reset", &loop.ToolExecutionConfig{}, false, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				release := make(chan struct{})
				var active atomic.Int32
				tool := controlTool(t, func(ctx context.Context, _ ai.ToolCall) (string, error) {
					_, deadline := ctx.Deadline()
					if deadline != tc.wantDeadline {
						t.Errorf("deadline=%v", deadline)
					}
					active.Add(1)
					defer active.Add(-1)
					<-release
					return "ok", nil
				})
				a := agent.New(agent.Definition{Model: toolControlModel("test", "test"), Prompt: executionPrompt, Tools: []loop.Tool{tool}, ToolExecution: loop.ToolExecutionConfig{MaxConcurrent: 1, DefaultTimeout: time.Hour}})
				workflow, err := a.NewRun(t.Context(), agent.RunInput{Execution: &agent.ExecutionOverrides{ToolExecution: tc.patch}})
				if err != nil {
					t.Fatal(err)
				}
				if tc.patch != nil {
					*tc.patch = loop.ToolExecutionConfig{MaxConcurrent: 1, DefaultTimeout: time.Second}
				}
				done := make(chan error, 1)
				go func() { _, err := workflow.Run(t.Context()); done <- err }()
				synctest.Wait()
				if got := active.Load(); got != tc.wantActive {
					t.Errorf("active=%d want %d", got, tc.wantActive)
				}
				close(release)
				if err := <-done; err != nil {
					t.Fatal(err)
				}
			})
		})
	}
}

type mutableControlTool struct {
	loop.Tool
	options loop.ToolOptions
}

func (m *mutableControlTool) ToolOptions() loop.ToolOptions { return m.options }

func TestAgentCapturesCustomToolOptionsAtDefinitionAndRun(t *testing.T) {
	for _, override := range []bool{false, true} {
		timeout := time.Hour
		var sawOriginal atomic.Bool
		tool := &mutableControlTool{Tool: controlTool(t, func(ctx context.Context, _ ai.ToolCall) (string, error) {
			deadline, ok := ctx.Deadline()
			if !ok || time.Until(deadline) < 59*time.Minute {
				t.Error("tool timeout not snapshotted")
			}
			return "ok", nil
		}), options: loop.ToolOptions{Traits: loop.ToolTraits{Effect: loop.ToolEffectReadOnly}, Timeout: &timeout}}
		def := agent.Definition{Model: toolControlModel("test"), Prompt: executionPrompt, ToolPolicy: loop.ToolPolicyFunc(func(_ context.Context, input loop.ToolPolicyInput) (loop.ToolDecision, error) {
			sawOriginal.Store(input.Traits.Effect == loop.ToolEffectReadOnly)
			return loop.ToolDecision{Action: loop.ToolAllow}, nil
		})}
		if !override {
			def.Tools = []loop.Tool{tool}
		}
		a := agent.New(def)
		if !override {
			tool.options.Traits.Effect = loop.ToolEffectDestructive
			timeout = time.Nanosecond
		}
		input := agent.RunInput{}
		if override {
			input.Execution = &agent.ExecutionOverrides{Tools: []loop.Tool{tool}}
		}
		workflow, err := a.NewRun(t.Context(), input)
		if err != nil {
			t.Fatal(err)
		}
		if override {
			tool.options.Traits.Effect = loop.ToolEffectDestructive
			timeout = time.Nanosecond
		}
		// Returned input snapshots must not expose the retained registration's timeout.
		if override {
			snapshot := workflow.Result().Input.Execution.Tools[0].(loop.ToolOptionsProvider).ToolOptions()
			*snapshot.Timeout = time.Nanosecond
		}
		if _, err := workflow.Run(t.Context()); err != nil {
			t.Fatal(err)
		}
		if !sawOriginal.Load() {
			t.Fatal("tool traits not snapshotted")
		}
	}
}

func TestAgentValidatesToolControlsAfterOverridesBeforePrompt(t *testing.T) {
	var nilPolicy loop.ToolPolicyFunc
	var nilResolver loop.ToolApprovalResolverFunc
	var nilTool *mutableControlTool
	for _, patch := range []*agent.ExecutionOverrides{
		{ToolExecution: &loop.ToolExecutionConfig{MaxConcurrent: -1}},
		{ToolExecution: &loop.ToolExecutionConfig{DefaultTimeout: -1}},
		{ToolPolicy: agent.Optional[loop.ToolPolicy]{Set: true, Value: nilPolicy}},
		{ToolApprovalResolver: agent.Optional[loop.ToolApprovalResolver]{Set: true, Value: nilResolver}},
	} {
		called := false
		a := agent.New(agent.Definition{Model: toolControlModel(), Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			called = true
			return &testPromptBuilder{}, nil
		}})
		if _, err := a.NewRun(t.Context(), agent.RunInput{Execution: patch}); err == nil || called {
			t.Fatalf("err=%v prompt=%v", err, called)
		}
	}
	a := agent.New(agent.Definition{Model: toolControlModel(), Prompt: executionPrompt, Tools: []loop.Tool{nilTool}, ToolExecution: loop.ToolExecutionConfig{MaxConcurrent: -1}, ToolPolicy: nilPolicy, ToolApprovalResolver: nilResolver})
	if _, err := a.NewRun(t.Context(), agent.RunInput{Execution: &agent.ExecutionOverrides{Tools: []loop.Tool{}, ToolExecution: &loop.ToolExecutionConfig{}, ToolPolicy: agent.Optional[loop.ToolPolicy]{Set: true}, ToolApprovalResolver: agent.Optional[loop.ToolApprovalResolver]{Set: true}}}); err != nil {
		t.Fatalf("valid override could not replace invalid defaults: %v", err)
	}
}

func TestAgentPreservesGuardAcrossConcurrentWorkflows(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		release := make(chan struct{})
		var active, maximum, invoked atomic.Int32
		raw := controlTool(t, func(context.Context, ai.ToolCall) (string, error) {
			n := active.Add(1)
			defer active.Add(-1)
			invoked.Add(1)
			for old := maximum.Load(); n > old; old = maximum.Load() {
				if maximum.CompareAndSwap(old, n) {
					break
				}
			}
			<-release
			return "ok", nil
		})
		shared := &mutableControlTool{Tool: raw, options: loop.ToolOptions{Guard: &loop.ToolGuard{}}}
		a := agent.New(agent.Definition{Prompt: executionPrompt, Tools: []loop.Tool{shared}})
		var workflows []*agent.Workflow
		for range 2 {
			w, err := a.NewRun(t.Context(), agent.RunInput{Execution: &agent.ExecutionOverrides{Model: toolControlModel("test")}})
			if err != nil {
				t.Fatal(err)
			}
			workflows = append(workflows, w)
		}
		var group sync.WaitGroup
		for _, w := range workflows {
			group.Go(func() {
				if _, err := w.Run(t.Context()); err != nil {
					t.Error(err)
				}
			})
		}
		synctest.Wait()
		if active.Load() != 1 {
			t.Errorf("guard allowed %d active handlers", active.Load())
		}
		close(release)
		group.Wait()
		if maximum.Load() != 1 || invoked.Load() != 2 {
			t.Fatalf("max=%d invoked=%d", maximum.Load(), invoked.Load())
		}
	})
}
