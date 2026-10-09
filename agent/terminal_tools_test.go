package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
)

func agentTerminalTool(t *testing.T) loop.Tool {
	t.Helper()
	tool, err := loop.NewTool("present", "Present the response", ai.ToolParameters{}, func(context.Context, ai.ToolCall) (string, error) {
		return "", nil
	})
	if err != nil {
		t.Fatal(err)
	}
	tool, err = loop.WithToolOptions(tool, loop.ToolOptions{Terminal: true})
	if err != nil {
		t.Fatal(err)
	}
	return tool
}

func TestAgentTerminalToolCanAllowOrdinarySiblingAndContinue(t *testing.T) {
	var invoked atomic.Int32
	handler := func(context.Context, ai.ToolCall) (string, error) {
		invoked.Add(1)
		return "ok", nil
	}
	present, err := loop.NewTool("present", "Present the response", ai.ToolParameters{}, handler)
	if err != nil {
		t.Fatal(err)
	}
	present, err = loop.WithToolOptions(present, loop.ToolOptions{
		Terminal:              true,
		AllowNonTerminalCalls: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	lookup, err := loop.NewTool("lookup", "Lookup information", ai.ToolParameters{}, handler)
	if err != nil {
		t.Fatal(err)
	}
	base := &scriptedWorkflowModel{scripts: [][]ai.Token{
		{
			{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "present-1", Type: "function", Name: "present", Args: json.RawMessage(`{}`)}}},
			{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "lookup-1", Type: "function", Name: "lookup", Args: json.RawMessage(`{}`)}}},
		},
		{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "done"}}},
	}}
	workflow, err := agent.New(agent.Definition{
		Model:  nativeToolWorkflowModel{base},
		Prompt: executionPrompt,
		Tools:  []loop.Tool{present, lookup},
		Limits: agent.Limits{MaxLoopIterations: 2},
	}).NewRun(t.Context(), agent.RunInput{})
	if err != nil {
		t.Fatal(err)
	}
	result, err := workflow.Run(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if got := invoked.Load(); got != 2 {
		t.Fatalf("handler invocations = %d, want 2", got)
	}
	if got := len(base.Requests()); got != 2 {
		t.Fatalf("model requests = %d, want 2", got)
	}
	if !result.Complete || result.Text != "done" || len(result.Primary.Iterations) != 2 {
		t.Fatalf("workflow result = %#v", result)
	}
}

func TestAgentTerminalToolsFinishPrimaryAndRetainMiddlewareLifecycle(t *testing.T) {
	for _, override := range []bool{false, true} {
		t.Run(map[bool]string{false: "definition", true: "run override"}[override], func(t *testing.T) {
			base := &scriptedWorkflowModel{scripts: [][]ai.Token{
				{{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "present-1", Type: "function", Name: "present", Args: json.RawMessage(`{}`)}}}},
				{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "must not be requested"}}},
			}}
			var middlewareFinished atomic.Bool
			middleware := agent.MiddlewareFunc(func(_ context.Context, _ *agent.MiddlewareContext, upstream <-chan agent.Event) <-chan agent.Event {
				out := make(chan agent.Event)
				go func() {
					defer close(out)
					defer middlewareFinished.Store(true)
					for event := range upstream {
						out <- event
					}
				}()
				return out
			})
			definition := agent.Definition{
				Model:      nativeToolWorkflowModel{base},
				Prompt:     executionPrompt,
				Middleware: []agent.Middleware{middleware},
				Limits:     agent.Limits{MaxLoopIterations: 1},
			}
			input := agent.RunInput{}
			if override {
				input.Execution = &agent.ExecutionOverrides{Tools: []loop.Tool{agentTerminalTool(t)}}
			} else {
				definition.Tools = []loop.Tool{agentTerminalTool(t)}
			}
			workflow, err := agent.New(definition).NewRun(t.Context(), input)
			if err != nil {
				t.Fatal(err)
			}
			result, err := workflow.Run(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			if len(base.Requests()) != 1 {
				t.Fatalf("model requests = %d, want 1", len(base.Requests()))
			}
			if !result.Complete || result.Text != "" || result.Primary.Text != "" || len(result.Primary.Iterations) != 1 {
				t.Fatalf("terminal workflow result = %#v", result)
			}
			if !middlewareFinished.Load() {
				t.Fatal("workflow completed before middleware lifecycle finished")
			}
			var results int
			for _, message := range result.Primary.Messages {
				results += len(message.ToolResults())
			}
			if results != 1 {
				t.Fatalf("canonical terminal results = %d, want 1", results)
			}
		})
	}
}

func TestAgentRetainsExcludedRegistrationForMixedTerminalClassification(t *testing.T) {
	var invoked atomic.Int32
	handler := func(context.Context, ai.ToolCall) (string, error) {
		invoked.Add(1)
		return "ok", nil
	}
	present, err := loop.NewTool("present", "Present the response", ai.ToolParameters{}, handler)
	if err != nil {
		t.Fatal(err)
	}
	present, err = loop.WithToolOptions(present, loop.ToolOptions{Terminal: true})
	if err != nil {
		t.Fatal(err)
	}
	lookup, err := loop.NewTool("lookup", "Lookup information", ai.ToolParameters{}, handler)
	if err != nil {
		t.Fatal(err)
	}
	base := &scriptedWorkflowModel{scripts: [][]ai.Token{
		{
			{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "present-1", Type: "function", Name: "present", Args: json.RawMessage(`{}`)}}},
			{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "lookup-1", Type: "function", Name: "lookup", Args: json.RawMessage(`{}`)}}},
		},
		{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "must not be requested"}}},
	}}
	workflow, err := agent.New(agent.Definition{
		Model:      nativeToolWorkflowModel{base},
		Prompt:     executionPrompt,
		Tools:      []loop.Tool{present, lookup},
		ToolChoice: ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"present"}},
	}).NewRun(t.Context(), agent.RunInput{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = workflow.Run(t.Context())
	if !errors.Is(err, loop.ErrMixedTerminalToolBatch) {
		t.Fatalf("error = %v, want ErrMixedTerminalToolBatch", err)
	}
	if got := invoked.Load(); got != 0 {
		t.Fatalf("handler invocations = %d, want 0", got)
	}
	if got := len(base.Requests()); got != 1 {
		t.Fatalf("model requests = %d, want 1", got)
	}
}

func TestTerminalPrimaryPreservesUsageEventsAndAgentMiddleware(t *testing.T) {
	primaryModel := &scriptedWorkflowModel{scripts: [][]ai.Token{{
		{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "present-1", Type: "function", Name: "present", Args: json.RawMessage(`{}`)}}},
		{Completion: &ai.Completion{UsageReported: true, Usage: ai.Usage{InputTokens: 7, OutputTokens: 2}}},
	}}}
	middlewareModel := &scriptedWorkflowModel{scripts: [][]ai.Token{{
		{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "middleware output"}},
	}}}
	middlewareAgent := agent.New(agent.Definition{
		Name:   "post",
		Model:  middlewareModel,
		Prompt: executionPrompt,
	})
	workflow, err := agent.New(agent.Definition{
		Name:   "primary",
		Model:  nativeToolWorkflowModel{primaryModel},
		Prompt: executionPrompt,
		Tools:  []loop.Tool{agentTerminalTool(t)},
		Limits: agent.Limits{MaxLoopIterations: 1},
		Middleware: []agent.Middleware{
			agent.NewAgentMiddleware(middlewareAgent, agent.AgentMiddlewareConfig{Output: agent.AppendOutput}),
		},
	}).NewRun(t.Context(), agent.RunInput{})
	if err != nil {
		t.Fatal(err)
	}
	var primaryIterations, workflowDone int
	for event := range workflow.RunEvents(t.Context()) {
		if event.Type == agent.EventIterationDone && event.Source.Kind == agent.SourcePrimary {
			primaryIterations++
		}
		if event.Type == agent.EventDone && event.Source.Kind == agent.SourceWorkflow {
			workflowDone++
		}
	}
	result, err := workflow.Wait()
	if err != nil {
		t.Fatal(err)
	}
	if len(primaryModel.Requests()) != 1 || len(middlewareModel.Requests()) != 1 {
		t.Fatalf("model requests primary=%d middleware=%d, want 1/1", len(primaryModel.Requests()), len(middlewareModel.Requests()))
	}
	if primaryIterations != 1 || workflowDone != 1 {
		t.Fatalf("terminal outcomes primary iterations=%d workflow done=%d, want 1/1", primaryIterations, workflowDone)
	}
	if result.Primary.Usage.InputTokens != 7 || result.Primary.Usage.OutputTokens != 2 ||
		result.Primary.BilledUsage.InputTokens != 7 || result.Primary.BilledUsage.OutputTokens != 2 {
		t.Fatalf("primary accounting usage=%+v billed=%+v", result.Primary.Usage, result.Primary.BilledUsage)
	}
	if !result.Complete || result.Primary.Text != "" || result.Text != "middleware output" || len(result.Stages) != 1 {
		t.Fatalf("terminal middleware result = %#v", result)
	}
}
