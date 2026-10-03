package agent_test

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

// The required user message costs 1 text token + 4 framing tokens, followed
// by the request's 3-token response prefix. No system or optional context exists.
const agentRequestInputTokens = 8

func agentRequestBudgetPrompt(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
	return gaictx.New(gaictx.Definition{TokenBudget: 50, OutputTokenReserve: 6}), nil
}

func agentRequestBudgetModel() *scriptedWorkflowModel {
	return &scriptedWorkflowModel{scripts: [][]ai.Token{{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "answer"}}}}}
}

func requireAgentRequestBudget(t *testing.T, result agent.WorkflowResult, want *ai.RequestBudgetConfig) {
	t.Helper()
	if !result.Complete || result.Text != "answer" || len(result.Errors) != 0 || len(result.Primary.Iterations) != 1 {
		t.Fatalf("successful budgeted result = %+v", result)
	}
	got := result.Primary.Iterations[0].RequestBudget
	if want == nil {
		if got != nil {
			t.Fatalf("disabled policy produced accounting: %+v", got)
		}
		return
	}
	if got == nil || got.InputTokens != agentRequestInputTokens || got.Limit != want.Limit || got.InputLimit != want.InputLimit || got.OutputReserve != want.OutputReserve || got.SafetyMargin != want.SafetyMargin || got.TotalTokens != agentRequestInputTokens+want.OutputReserve+want.SafetyMargin || got.Method != "local_estimate" {
		t.Fatalf("resolved request budget = %+v, want policy %+v with %d input tokens", got, want, agentRequestInputTokens)
	}
}

func TestAgentRequestBudgetInheritanceAndOptionalOverrides(t *testing.T) {
	definition := &ai.RequestBudgetConfig{Limit: 20, InputLimit: 10, OutputReserve: 4, SafetyMargin: 2}
	builderDefaults := &ai.RequestBudgetConfig{Limit: 50, OutputReserve: 6}
	replacement := &ai.RequestBudgetConfig{Limit: agentRequestInputTokens}
	for _, tc := range []struct {
		name       string
		definition *ai.RequestBudgetConfig
		overrides  *agent.ExecutionOverrides
		want       *ai.RequestBudgetConfig
	}{
		{name: "builder defaults", want: builderDefaults},
		{name: "inherited definition", definition: definition, want: definition},
		{name: "empty override inherits", definition: definition, overrides: &agent.ExecutionOverrides{}, want: definition},
		{name: "unset value ignored", definition: definition, overrides: &agent.ExecutionOverrides{RequestBudget: agent.Optional[*ai.RequestBudgetConfig]{Value: &ai.RequestBudgetConfig{Limit: -1, Mode: 99}}}, want: definition},
		{name: "nil restores builder defaults", definition: definition, overrides: &agent.ExecutionOverrides{RequestBudget: agent.Optional[*ai.RequestBudgetConfig]{Set: true}}, want: builderDefaults},
		{name: "explicit zero disables", definition: definition, overrides: &agent.ExecutionOverrides{RequestBudget: agent.Optional[*ai.RequestBudgetConfig]{Set: true, Value: &ai.RequestBudgetConfig{}}}},
		{
			name:       "atomic replacement clears every inherited field",
			definition: &ai.RequestBudgetConfig{Limit: 100, InputLimit: 1, OutputReserve: 11, SafetyMargin: 2, Mode: ai.RequestCountAccurate},
			overrides:  &agent.ExecutionOverrides{RequestBudget: agent.Optional[*ai.RequestBudgetConfig]{Set: true, Value: replacement}},
			want:       replacement,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := agentRequestBudgetModel()
			a := agent.New(agent.Definition{Model: model, Prompt: agentRequestBudgetPrompt, TokenCounter: &mocks.MockTokenCounter{Count: 1}, RequestBudget: tc.definition})
			input := textRunInput("question")
			input.Execution = tc.overrides
			workflow, err := a.NewRun(t.Context(), input)
			if err != nil || len(model.Requests()) != 0 {
				t.Fatalf("NewRun = %v, requests %d", err, len(model.Requests()))
			}
			consumed := consumeWorkflowContext(t, workflow, t.Context())
			result, err := workflow.Wait()
			if err != nil || len(consumed.errs) != 0 || len(model.Requests()) != 1 {
				t.Fatalf("run error = %v, event errors %v, requests %d", err, consumed.errs, len(model.Requests()))
			}
			requireAgentRequestBudget(t, result, tc.want)
			iterations := 0
			for _, event := range consumed.events {
				if event.Type == agent.EventIterationDone {
					iterations++
					if event.Iteration == nil || !reflect.DeepEqual(event.Iteration.RequestBudget, result.Primary.Iterations[0].RequestBudget) {
						t.Fatalf("event/result budget differs: event %+v, result %+v", event.Iteration, result.Primary.Iterations[0])
					}
				}
			}
			if iterations != 1 {
				t.Fatalf("completed iteration events = %d, want one", iterations)
			}
		})
	}
}

func TestAgentRequestBudgetRejectsInvalidConfigurationBeforePrompt(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name   string
		budget ai.RequestBudgetConfig
	}{
		{"negative window", ai.RequestBudgetConfig{Limit: -1}},
		{"negative input limit", ai.RequestBudgetConfig{InputLimit: -1}},
		{"negative output reserve", ai.RequestBudgetConfig{OutputReserve: -1}},
		{"negative safety margin", ai.RequestBudgetConfig{SafetyMargin: -1}},
		{"unknown mode", ai.RequestBudgetConfig{Mode: 99}},
		{"overflow", ai.RequestBudgetConfig{InputLimit: maxInt, OutputReserve: 1}},
	} {
		for _, override := range []bool{false, true} {
			origin := "definition"
			if override {
				origin = "override"
			}
			t.Run(tc.name+"/"+origin, func(t *testing.T) {
				model := agentRequestBudgetModel()
				promptCalls := 0
				definition := agent.Definition{Model: model, Prompt: func(ctx context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
					promptCalls++
					return agentRequestBudgetPrompt(ctx, input)
				}}
				input := textRunInput("question")
				if override {
					input.Execution = &agent.ExecutionOverrides{RequestBudget: agent.Optional[*ai.RequestBudgetConfig]{Set: true, Value: &tc.budget}}
				} else {
					definition.RequestBudget = &tc.budget
				}
				workflow, err := agent.New(definition).NewRun(t.Context(), input)
				if workflow != nil || !errors.Is(err, agent.ErrInvalidExecutionConfig) || !errors.Is(err, ai.ErrInvalidRequestBudget) || promptCalls != 0 || len(model.Requests()) != 0 {
					t.Fatalf("NewRun = %v, workflow %v, prompt calls %d, model requests %d", err, workflow, promptCalls, len(model.Requests()))
				}
			})
		}
	}
}

func TestAgentRequestBudgetValidOverrideReplacesInvalidDefault(t *testing.T) {
	replacement := &ai.RequestBudgetConfig{Limit: agentRequestInputTokens}
	for _, tc := range []struct {
		name     string
		override *ai.RequestBudgetConfig
		want     *ai.RequestBudgetConfig
	}{
		{"valid replacement", replacement, replacement},
		{"nil restores builder defaults", nil, &ai.RequestBudgetConfig{Limit: 50, OutputReserve: 6}},
		{"explicit zero disables", &ai.RequestBudgetConfig{}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := agentRequestBudgetModel()
			a := agent.New(agent.Definition{
				Model: model, Prompt: agentRequestBudgetPrompt, TokenCounter: &mocks.MockTokenCounter{Count: 1},
				RequestBudget: &ai.RequestBudgetConfig{Limit: -1, SafetyMargin: -1, Mode: 99},
			})
			input := textRunInput("question")
			input.Execution = &agent.ExecutionOverrides{RequestBudget: agent.Optional[*ai.RequestBudgetConfig]{Set: true, Value: tc.override}}
			workflow, err := a.NewRun(t.Context(), input)
			if err != nil {
				t.Fatal(err)
			}
			result, err := workflow.Run(t.Context())
			if err != nil || len(model.Requests()) != 1 {
				t.Fatalf("Run = %v, requests %d", err, len(model.Requests()))
			}
			requireAgentRequestBudget(t, result, tc.want)
		})
	}
}

func TestAgentRequestBudgetSnapshotsDefinitionAndRunOverride(t *testing.T) {
	for _, override := range []bool{false, true} {
		name := "definition"
		if override {
			name = "run override"
		}
		t.Run(name, func(t *testing.T) {
			model := agentRequestBudgetModel()
			policy := &ai.RequestBudgetConfig{Limit: agentRequestInputTokens}
			want := *policy
			definition := agent.Definition{Model: model, Prompt: agentRequestBudgetPrompt, TokenCounter: &mocks.MockTokenCounter{Count: 1}}
			input := textRunInput("question")
			if override {
				definition.RequestBudget = &ai.RequestBudgetConfig{Limit: 1}
				input.Execution = &agent.ExecutionOverrides{RequestBudget: agent.Optional[*ai.RequestBudgetConfig]{Set: true, Value: policy}}
			} else {
				definition.RequestBudget = policy
			}
			a := agent.New(definition)
			if !override {
				// Definition ownership transfers at New, before NewRun snapshots it.
				*policy = ai.RequestBudgetConfig{Limit: -1}
			}
			workflow, err := a.NewRun(t.Context(), input)
			if err != nil {
				t.Fatal(err)
			}
			*policy = ai.RequestBudgetConfig{Limit: 1, Mode: ai.RequestCountAccurate}
			if override {
				input.Execution.RequestBudget = agent.Optional[*ai.RequestBudgetConfig]{Set: true}
				snapshot := workflow.Result()
				if snapshot.Input.Execution == nil || snapshot.Input.Execution.RequestBudget.Value == nil || *snapshot.Input.Execution.RequestBudget.Value != want {
					t.Fatalf("requested override was not snapshotted: %+v", snapshot.Input.Execution)
				}
				snapshot.Input.Execution.RequestBudget.Value.Limit = 1
			}
			result, err := workflow.Run(t.Context())
			if err != nil || len(model.Requests()) != 1 {
				t.Fatalf("Run = %v, requests %d", err, len(model.Requests()))
			}
			requireAgentRequestBudget(t, result, &want)
			if override {
				if result.Input.Execution == nil || result.Input.Execution.RequestBudget.Value == nil || *result.Input.Execution.RequestBudget.Value != want {
					t.Fatalf("retained requested override changed: %+v", result.Input.Execution)
				}
				result.Input.Execution.RequestBudget.Value.Limit = 1
				if retained := workflow.Result().Input.Execution.RequestBudget.Value; retained == nil || *retained != want {
					t.Fatalf("returned result aliases the retained policy: %+v", retained)
				}
			}
		})
	}
}

func TestAgentRequestBudgetFailureUsesWorkflowResultsAndEvents(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		name := "Run"
		if streaming {
			name = "RunEvents"
		}
		t.Run(name, func(t *testing.T) {
			model := agentRequestBudgetModel()
			a := agent.New(agent.Definition{
				Model: model, Prompt: agentRequestBudgetPrompt, TokenCounter: &mocks.MockTokenCounter{Count: 1},
				RequestBudget: &ai.RequestBudgetConfig{InputLimit: agentRequestInputTokens - 1},
			})
			workflow, err := a.NewRun(t.Context(), textRunInput("question"))
			if err != nil {
				t.Fatal(err)
			}
			var result agent.WorkflowResult
			if streaming {
				events := collectAgentEvents(workflow.RunEvents(t.Context()))
				if len(events) == 0 || events[len(events)-1].Type != agent.EventError || !errors.Is(events[len(events)-1].Err, ai.ErrRequestBudgetExceeded) {
					t.Fatalf("missing terminal budget error: %+v", events)
				}
				terminal := 0
				for _, event := range events {
					switch event.Type {
					case agent.EventDone, agent.EventError, agent.EventCanceled:
						terminal++
					case agent.EventOutput, agent.EventToolStart, agent.EventToolResult, agent.EventIterationDone:
						t.Fatalf("rejected request emitted generation output: %+v", event)
					}
				}
				if terminal != 1 {
					t.Fatalf("terminal events = %d, want one", terminal)
				}
				result, err = workflow.Wait()
			} else {
				result, err = workflow.Run(t.Context())
			}
			var exceeded *ai.RequestBudgetExceededError
			if !errors.As(err, &exceeded) || exceeded.Budget.InputTokens != agentRequestInputTokens || exceeded.Budget.InputLimit != agentRequestInputTokens-1 {
				t.Fatalf("workflow error = %v, want typed complete-input budget rejection", err)
			}
			if !result.Complete || result.Canceled || len(result.Errors) != 1 || !errors.Is(result.Errors[0], ai.ErrRequestBudgetExceeded) || len(result.Primary.Errors) != 1 || !errors.Is(result.Primary.Errors[0], ai.ErrRequestBudgetExceeded) || len(result.Primary.Iterations) != 0 || len(result.Output) != 0 || result.Text != "" || len(model.Requests()) != 0 {
				t.Fatalf("rejected workflow result = %+v, model requests %d", result, len(model.Requests()))
			}
		})
	}
}
