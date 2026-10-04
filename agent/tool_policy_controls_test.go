package agent_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

func TestAgentPolicyInheritanceReplacementAndClear(t *testing.T) {
	deny := loop.ToolPolicyFunc(func(context.Context, loop.ToolPolicyInput) (loop.ToolDecision, error) {
		return loop.ToolDecision{Action: loop.ToolDeny}, nil
	})
	allow := loop.ToolPolicyFunc(func(context.Context, loop.ToolPolicyInput) (loop.ToolDecision, error) {
		return loop.ToolDecision{Action: loop.ToolAllow}, nil
	})
	for _, tc := range []struct {
		name   string
		patch  agent.Optional[loop.ToolPolicy]
		invoke bool
	}{
		{"inherit", agent.Optional[loop.ToolPolicy]{}, false},
		{"ignore unset value", agent.Optional[loop.ToolPolicy]{Value: allow}, false},
		{"replace", agent.Optional[loop.ToolPolicy]{Set: true, Value: allow}, true},
		{"clear", agent.Optional[loop.ToolPolicy]{Set: true}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var invoked atomic.Bool
			tool, err := loop.NewTool("test", "test", ai.ToolParameters{}, func(context.Context, ai.ToolCall) (string, error) {
				invoked.Store(true)
				return "ok", nil
			})
			if err != nil {
				t.Fatal(err)
			}
			model := &scriptedWorkflowModel{scripts: [][]ai.Token{
				{{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "call", Type: "function", Name: "test", Args: []byte(`{}`)}}}},
				{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "done"}}},
			}}
			a := agent.New(agent.Definition{Model: nativeToolWorkflowModel{model}, Prompt: executionPrompt, Tools: []loop.Tool{tool}, ToolPolicy: deny})
			workflow, err := a.NewRun(t.Context(), agent.RunInput{Execution: &agent.ExecutionOverrides{ToolPolicy: tc.patch}})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := workflow.Run(t.Context()); err != nil {
				t.Fatal(err)
			}
			if invoked.Load() != tc.invoke {
				t.Fatalf("invoked=%v", invoked.Load())
			}
		})
	}
}

func TestAgentValidatesPolicyBeforePromptAfterOverrides(t *testing.T) {
	var nilPolicy loop.ToolPolicyFunc
	for _, clear := range []bool{false, true} {
		called := false
		a := agent.New(agent.Definition{Model: &scriptedWorkflowModel{}, ToolPolicy: nilPolicy, Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			called = true
			return &testPromptBuilder{}, nil
		}})
		_, err := a.NewRun(t.Context(), agent.RunInput{Execution: &agent.ExecutionOverrides{ToolPolicy: agent.Optional[loop.ToolPolicy]{Set: clear}}})
		if clear {
			if err != nil || !called {
				t.Fatalf("clear failed: %v", err)
			}
		} else if !errors.Is(err, loop.ErrToolPolicy) || called {
			t.Fatalf("invalid policy reached prompt: called=%v err=%v", called, err)
		}
	}
}
