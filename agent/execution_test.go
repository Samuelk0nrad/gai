package agent_test

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/tooldefinitions"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/testutil/mocks"
)

func executionPtr[T any](value T) *T { return &value }

func executionPrompt(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
	return &testPromptBuilder{}, nil
}

type executionModel struct {
	*scriptedWorkflowModel
	name    string
	counter ai.TokenCounter
	native  bool
}

func (m *executionModel) Name() string                  { return m.name }
func (m *executionModel) TokenCounter() ai.TokenCounter { return m.counter }
func (m *executionModel) Descriptor() ai.ModelDescriptor {
	if m.native {
		return ai.ModelDescriptor{NativeTools: ai.FeatureSupportSupported}
	}
	return ai.ModelDescriptor{NativeTools: ai.FeatureSupportUnsupported}
}

func TestExecutionInheritsReplacesAndResetsValues(t *testing.T) {
	reasoning := ai.ReasoningConfig{Enabled: true, BudgetTokens: 12, Effort: ai.ReasoningEffortHigh}
	format := ai.ResponseFormat{Type: ai.ResponseFormatJSONObject}
	choice := ai.ToolChoice{Mode: ai.ToolChoiceAuto}
	for _, tc := range []struct {
		name               string
		overrides          *agent.ExecutionOverrides
		tokens, iterations int
		reasoning          ai.ReasoningConfig
		format             ai.ResponseFormat
		choice             ai.ToolChoice
	}{
		{"nil", nil, 17, 3, reasoning, format, choice},
		{"empty", &agent.ExecutionOverrides{}, 17, 3, reasoning, format, choice},
		{"one limit", &agent.ExecutionOverrides{Limits: agent.LimitsOverrides{MaxTokens: executionPtr(7)}}, 7, 3, reasoning, format, choice},
		{"other limit", &agent.ExecutionOverrides{Limits: agent.LimitsOverrides{MaxLoopIterations: executionPtr(2)}}, 17, 2, reasoning, format, choice},
		{"zero limits", &agent.ExecutionOverrides{Limits: agent.LimitsOverrides{MaxTokens: executionPtr(0), MaxLoopIterations: executionPtr(0)}}, 0, loop.New(nil, nil, nil, nil).MaxLoopIterations, reasoning, format, choice},
		{"atomic replacement", &agent.ExecutionOverrides{Reasoning: &ai.ReasoningConfig{Effort: ai.ReasoningEffortLow}, ResponseFormat: &ai.ResponseFormat{Type: ai.ResponseFormatText}, ToolChoice: &ai.ToolChoice{Mode: ai.ToolChoiceNone}}, 17, 3, ai.ReasoningConfig{Effort: ai.ReasoningEffortLow}, ai.ResponseFormat{Type: ai.ResponseFormatText}, ai.ToolChoice{Mode: ai.ToolChoiceNone}},
		{"zero objects", &agent.ExecutionOverrides{Reasoning: &ai.ReasoningConfig{}, ResponseFormat: &ai.ResponseFormat{}, ToolChoice: &ai.ToolChoice{}}, 17, 3, ai.ReasoningConfig{}, ai.ResponseFormat{}, ai.ToolChoice{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &scriptedWorkflowModel{}
			sink := &agentObservationSink{}
			a := agent.New(agent.Definition{
				Model: nativeToolWorkflowModel{model}, Prompt: executionPrompt,
				Limits:    agent.Limits{MaxTokens: 17, MaxLoopIterations: 3},
				Reasoning: reasoning, ResponseFormat: format, ToolChoice: choice,
				Tools: []loop.Tool{loop.NewEchoTool()}, ObservationSink: sink,
			})
			workflow, err := a.NewRun(t.Context(), agent.RunInput{Execution: tc.overrides})
			if err != nil {
				t.Fatal(err)
			}
			if got := consumeWorkflow(t, workflow); len(got.errs) != 0 {
				t.Fatal(got.errs)
			}
			req := model.Requests()[0]
			if req.MaxTokens != tc.tokens || req.Reasoning != tc.reasoning || !reflect.DeepEqual(req.ResponseFormat, tc.format) || !reflect.DeepEqual(req.ToolChoice, tc.choice) {
				t.Fatalf("resolved request = %+v", req)
			}
			event, ok := sink.event("agent_run_created")
			if !ok || event.Fields["max_iterations"] != tc.iterations || event.Fields["max_tokens"] != tc.tokens {
				t.Fatalf("effective limits observation = %+v", event)
			}
		})
	}
}

func TestExecutionModelSelectsTransportAndObservations(t *testing.T) {
	for _, tc := range []struct {
		name                         string
		native, existing, clearTools bool
	}{
		{"native to text", false, false, false},
		{"text to native", true, false, false},
		{"text source to native", true, true, false},
		{"text source to native without tools", true, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base := &executionModel{scriptedWorkflowModel: &scriptedWorkflowModel{}, name: "base", native: !tc.native}
			selected := &executionModel{scriptedWorkflowModel: &scriptedWorkflowModel{}, name: "selected", native: tc.native, counter: &mocks.MockTokenCounter{}}
			sink := &agentObservationSink{}
			var builder *gaictx.Builder
			a := agent.New(agent.Definition{
				Model: base, Tools: []loop.Tool{loop.NewEchoTool()}, ObservationSink: sink,
				Prompt: func(ctx context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
					if input.Execution.Model != selected || input.Execution.Limits.MaxTokens != nil {
						t.Fatal("prompt did not receive requested patch")
					}
					builder = gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}})
					if tc.existing {
						source, err := tooldefinitions.New(nil, []gaictx.ToolSignature{namedTool{name: "stale"}}, nil)
						if err != nil {
							return nil, err
						}
						if err := builder.AppendContextSource(ctx, source); err != nil {
							return nil, err
						}
					}
					return builder, nil
				},
			})
			overrides := &agent.ExecutionOverrides{Model: selected}
			if tc.clearTools {
				overrides.Tools = []loop.Tool{}
			}
			workflow, err := a.NewRun(t.Context(), agent.RunInput{Execution: overrides, Prompt: gaictx.PromptInput{User: ai.TextParts("")}})
			if err != nil {
				t.Fatal(err)
			}
			if builder.HasContextSource("tool_definitions") == tc.native {
				t.Fatal("tool protocol does not match effective model")
			}
			if builder.TokenCounter() != selected.counter {
				t.Fatal("counter does not match effective model")
			}
			if got := consumeWorkflow(t, workflow); len(got.errs) != 0 {
				t.Fatal(got.errs)
			}
			requests := selected.Requests()
			if len(base.Requests()) != 0 || len(requests) != 1 || (len(requests[0].Tools) > 0) != (tc.native && !tc.clearTools) {
				t.Fatalf("model selection or native tool request is wrong: %+v", requests)
			}
			count := 1
			if tc.clearTools {
				count = 0
			}
			event, _ := sink.event("agent_run_created")
			if event.Fields["model"] != "selected" || event.Fields["tool_count"] != count {
				t.Fatalf("effective observation = %+v", event)
			}
		})
	}
}

func TestExecutionTokenCounterSelectionAndExplicitClear(t *testing.T) {
	inherited, custom, automatic := &mocks.MockTokenCounter{}, &mocks.MockTokenCounter{}, &mocks.MockTokenCounter{}
	for _, tc := range []struct {
		name        string
		override    agent.Optional[ai.TokenCounter]
		model, want ai.TokenCounter
	}{
		{"inherit", agent.Optional[ai.TokenCounter]{}, automatic, inherited},
		{"ignored value", agent.Optional[ai.TokenCounter]{Value: custom}, automatic, inherited},
		{"replace", agent.Optional[ai.TokenCounter]{Set: true, Value: custom}, automatic, custom},
		{"clear custom", agent.Optional[ai.TokenCounter]{Set: true}, automatic, automatic},
		{"clear stale builder", agent.Optional[ai.TokenCounter]{Set: true}, nil, ai.TextTokenEstimator{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			builder := &testPromptBuilder{counter: custom}
			a := agent.New(agent.Definition{
				Model: &executionModel{scriptedWorkflowModel: &scriptedWorkflowModel{}, counter: tc.model}, TokenCounter: inherited,
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return builder, nil },
			})
			_, err := a.NewRun(t.Context(), agent.RunInput{Execution: &agent.ExecutionOverrides{TokenCounter: tc.override}})
			if err != nil || builder.counter != tc.want {
				t.Fatalf("counter = %v, want %v; error = %v", builder.counter, tc.want, err)
			}
		})
	}
}

func TestExecutionOpaqueBuilderRequiresExplicitTokenCounterSupport(t *testing.T) {
	for _, tc := range []struct {
		name      string
		counter   ai.TokenCounter
		overrides *agent.ExecutionOverrides
		wantError bool
	}{
		{"ordinary inheritance", nil, nil, false},
		{"definition custom", &mocks.MockTokenCounter{}, nil, true},
		{"explicit clear", nil, &agent.ExecutionOverrides{TokenCounter: agent.Optional[ai.TokenCounter]{Set: true}}, true},
		{"model switch", nil, &agent.ExecutionOverrides{Model: &scriptedWorkflowModel{}}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := agent.New(agent.Definition{Model: &scriptedWorkflowModel{}, TokenCounter: tc.counter,
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
					return &unclonablePromptBuilder{}, nil
				},
			})
			_, err := a.NewRun(t.Context(), agent.RunInput{Execution: tc.overrides})
			if errors.Is(err, agent.ErrTokenCounterNotConfigurable) != tc.wantError || (!tc.wantError && err != nil) {
				t.Fatalf("NewRun error = %v, want counter error %v", err, tc.wantError)
			}
		})
	}
}

type executionProcessor struct{ text string }

func (p *executionProcessor) Process(_ ai.ToolCall, res *loop.ToolResponse) error {
	*res = *loop.NewToolSuccess(p.text)
	return nil
}

func TestExecutionRejectsInvalidResolvedValuesBeforePrompt(t *testing.T) {
	var nilModel *executionModel
	var nilTokenCounter *mocks.MockTokenCounter
	var nilProcessor *executionProcessor
	for _, tc := range []struct {
		name      string
		overrides *agent.ExecutionOverrides
	}{
		{"negative tokens", &agent.ExecutionOverrides{Limits: agent.LimitsOverrides{MaxTokens: executionPtr(-1)}}},
		{"negative iterations", &agent.ExecutionOverrides{Limits: agent.LimitsOverrides{MaxLoopIterations: executionPtr(-1)}}},
		{"typed nil model", &agent.ExecutionOverrides{Model: nilModel}},
		{"typed nil counter", &agent.ExecutionOverrides{TokenCounter: agent.Optional[ai.TokenCounter]{Set: true, Value: nilTokenCounter}}},
		{"typed nil model counter", &agent.ExecutionOverrides{Model: &executionModel{scriptedWorkflowModel: &scriptedWorkflowModel{}, counter: nilTokenCounter}}},
		{"typed nil processor", &agent.ExecutionOverrides{ToolResponseProcessor: agent.Optional[loop.ToolResponseProcessor]{Set: true, Value: nilProcessor}}},
		{"bad retry", &agent.ExecutionOverrides{RetryPolicy: agent.Optional[*loop.RetryPolicy]{Set: true, Value: &loop.RetryPolicy{MaxRetries: -1}}}},
		{"bad format", &agent.ExecutionOverrides{ResponseFormat: &ai.ResponseFormat{Type: ai.ResponseFormatJSONSchema}}},
		{"bad choice", &agent.ExecutionOverrides{ToolChoice: &ai.ToolChoice{Mode: "invalid"}}},
		{"required missing tool", &agent.ExecutionOverrides{ToolChoice: &ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"missing"}}}},
		{"required cleared tools", &agent.ExecutionOverrides{Tools: []loop.Tool{}, ToolChoice: &ai.ToolChoice{Mode: ai.ToolChoiceRequired}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			called := false
			sink := &agentObservationSink{}
			a := agent.New(agent.Definition{
				Model: &scriptedWorkflowModel{}, Tools: []loop.Tool{loop.NewEchoTool()}, ObservationSink: sink,
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
					called = true
					return &testPromptBuilder{}, nil
				},
			})
			if _, err := a.NewRun(t.Context(), agent.RunInput{Execution: tc.overrides}); err == nil || called {
				t.Fatalf("NewRun error = %v; prompt called = %v", err, called)
			}
			if !sink.hasEvent("agent_run_creation_failed") {
				t.Fatal("missing failure observation")
			}
		})
	}
}

func TestExecutionCanReplaceInvalidDefaults(t *testing.T) {
	var invalidModel *executionModel
	a := agent.New(agent.Definition{
		Model: invalidModel, Prompt: executionPrompt, Limits: agent.Limits{MaxTokens: -1},
		RetryPolicy: &loop.RetryPolicy{MaxRetries: -1}, ResponseFormat: ai.ResponseFormat{Type: "invalid"},
	})
	workflow, err := a.NewRun(t.Context(), agent.RunInput{Execution: &agent.ExecutionOverrides{
		Model: &scriptedWorkflowModel{}, Limits: agent.LimitsOverrides{MaxTokens: executionPtr(4)},
		RetryPolicy: agent.Optional[*loop.RetryPolicy]{Set: true}, ResponseFormat: &ai.ResponseFormat{},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if got := consumeWorkflow(t, workflow); len(got.errs) != 0 {
		t.Fatal(got.errs)
	}
}

func TestExecutionConcurrentRunsKeepModelsAndLimitsIndependent(t *testing.T) {
	base := &scriptedWorkflowModel{}
	a := agent.New(agent.Definition{Model: base, Prompt: executionPrompt, Limits: agent.Limits{MaxTokens: 19}})
	var wg sync.WaitGroup
	for index := range 12 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			model := &scriptedWorkflowModel{}
			limit := index + 1
			workflow, err := a.NewRun(t.Context(), agent.RunInput{Execution: &agent.ExecutionOverrides{
				Model: model, Limits: agent.LimitsOverrides{MaxTokens: &limit},
			}})
			if err != nil {
				t.Error(err)
				return
			}
			if got := consumeWorkflow(t, workflow); len(got.errs) != 0 {
				t.Error(got.errs)
				return
			}
			if requests := model.Requests(); len(requests) != 1 || requests[0].MaxTokens != limit {
				t.Errorf("run %d received another run's settings: %+v", index, requests)
			}
		}()
	}
	wg.Wait()
	workflow, err := a.NewRun(t.Context(), agent.RunInput{})
	if err != nil {
		t.Fatal(err)
	}
	if got := consumeWorkflow(t, workflow); len(got.errs) != 0 {
		t.Fatal(got.errs)
	}
	if requests := base.Requests(); len(requests) != 1 || requests[0].MaxTokens != 19 {
		t.Fatalf("overrides changed definition defaults: %+v", requests)
	}
}

func TestExecutionSnapshotsDefinitionRequestedPatchAndCallback(t *testing.T) {
	model := &scriptedWorkflowModel{}
	def := agent.Definition{
		Model: nativeToolWorkflowModel{model}, Tools: []loop.Tool{loop.NewEchoTool()},
		ResponseFormat: ai.ResponseFormat{Type: ai.ResponseFormatJSONSchema, Name: "result", Schema: []byte("{}")},
		RetryPolicy:    &loop.RetryPolicy{MaxRetries: 1}, Limits: agent.Limits{MaxTokens: 9},
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			if input.Execution != nil {
				*input.Execution.Limits.MaxTokens = 999
				input.Execution.ResponseFormat.Schema[0] = '['
				input.Execution.RetryPolicy.Value.MaxRetries = 999
				input.Execution.Tools[0] = namedTool{name: "callback"}
			}
			return &testPromptBuilder{}, nil
		},
	}
	a := agent.New(def)
	def.ResponseFormat.Schema[0] = '['
	def.Tools[0] = namedTool{name: "changed"}
	def.RetryPolicy.MaxRetries = -1
	defaultRun, err := a.NewRun(t.Context(), agent.RunInput{})
	if err != nil {
		t.Fatal(err)
	}
	input := agent.RunInput{Execution: &agent.ExecutionOverrides{
		Limits:         agent.LimitsOverrides{MaxTokens: executionPtr(4)},
		ResponseFormat: &ai.ResponseFormat{Type: ai.ResponseFormatJSONSchema, Name: "run", Schema: []byte("{}")},
		RetryPolicy:    agent.Optional[*loop.RetryPolicy]{Set: true, Value: &loop.RetryPolicy{MaxRetries: 0}},
		Tools:          []loop.Tool{namedTool{name: "run"}},
	}}
	run, err := a.NewRun(t.Context(), input)
	if err != nil {
		t.Fatal(err)
	}
	for _, patch := range []*agent.ExecutionOverrides{input.Execution, run.Result().Input.Execution} {
		*patch.Limits.MaxTokens = 888
		patch.ResponseFormat.Schema[0] = '['
		patch.RetryPolicy.Value.MaxRetries = -1
		patch.Tools[0] = namedTool{name: "caller or result"}
	}
	for _, workflow := range []*agent.Workflow{defaultRun, run} {
		result, err := workflow.Run(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		waited, err := workflow.Wait()
		if err != nil {
			t.Fatal(err)
		}
		for _, patch := range []*agent.ExecutionOverrides{result.Input.Execution, waited.Input.Execution} {
			if patch == nil {
				continue
			}
			*patch.Limits.MaxTokens = 777
			patch.ResponseFormat.Schema[0] = '['
			patch.RetryPolicy.Value.MaxRetries = -1
			patch.Tools[0] = namedTool{name: "completed result"}
		}
	}
	requests := model.Requests()
	if requests[0].MaxTokens != 9 || requests[0].Tools[0].Name != "echo" || string(requests[0].ResponseFormat.Schema) != "{}" {
		t.Fatalf("definition was not snapshotted: %+v", requests[0])
	}
	if requests[1].MaxTokens != 4 || requests[1].Tools[0].Name != "run" || string(requests[1].ResponseFormat.Schema) != "{}" {
		t.Fatalf("effective run was mutated: %+v", requests[1])
	}
	retained := run.Result().Input.Execution
	if *retained.Limits.MaxTokens != 4 || string(retained.ResponseFormat.Schema) != "{}" || retained.RetryPolicy.Value.MaxRetries != 0 || retained.Tools[0].Name() != "run" {
		t.Fatalf("retained patch was mutated: %+v", retained)
	}
	if defaultRun.Result().Input.Execution != nil {
		t.Fatal("nil requested patch was filled with effective settings")
	}
}

func TestExecutionRetryOptionalInheritanceReplacementAndClear(t *testing.T) {
	for _, tc := range []struct {
		name     string
		override agent.Optional[*loop.RetryPolicy]
		attempts int
	}{
		{"inherit", agent.Optional[*loop.RetryPolicy]{}, 2},
		{"ignore unset value", agent.Optional[*loop.RetryPolicy]{Value: &loop.RetryPolicy{MaxRetries: -1}}, 2},
		{"replace", agent.Optional[*loop.RetryPolicy]{Set: true, Value: &loop.RetryPolicy{MaxRetries: 0}}, 1},
		{"clear", agent.Optional[*loop.RetryPolicy]{Set: true}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &scriptedWorkflowModel{scripts: [][]ai.Token{
				{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("temporary")}}},
				{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "done"}}},
			}}
			a := agent.New(agent.Definition{Model: model, Prompt: executionPrompt, RetryPolicy: &loop.RetryPolicy{MaxRetries: 1}})
			workflow, err := a.NewRun(t.Context(), agent.RunInput{Execution: &agent.ExecutionOverrides{RetryPolicy: tc.override}})
			if err != nil {
				t.Fatal(err)
			}
			got := consumeWorkflow(t, workflow)
			if len(model.Requests()) != tc.attempts || (len(got.errs) == 0) != (tc.attempts == 2) {
				t.Fatalf("attempts = %d; errors = %v", len(model.Requests()), got.errs)
			}
		})
	}
}

type deadlineExecutionModel struct {
	*scriptedWorkflowModel
	hasDeadline bool
}

func (m *deadlineExecutionModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	_, m.hasDeadline = ctx.Deadline()
	return m.scriptedWorkflowModel.GenerateStream(ctx, req)
}

func TestExecutionClearingRetryPolicyAlsoClearsTimeouts(t *testing.T) {
	for _, clear := range []bool{false, true} {
		t.Run(map[bool]string{false: "zero retries retains timeout", true: "nil clears timeout"}[clear], func(t *testing.T) {
			model := &deadlineExecutionModel{scriptedWorkflowModel: &scriptedWorkflowModel{}}
			a := agent.New(agent.Definition{Model: model, Prompt: executionPrompt, RetryPolicy: &loop.RetryPolicy{MaxRetries: 0, AttemptTimeout: time.Hour}})
			workflow, err := a.NewRun(context.Background(), agent.RunInput{Execution: &agent.ExecutionOverrides{RetryPolicy: agent.Optional[*loop.RetryPolicy]{Set: clear}}})
			if err != nil {
				t.Fatal(err)
			}
			if got := consumeWorkflow(t, workflow); len(got.errs) != 0 {
				t.Fatal(got.errs)
			}
			if model.hasDeadline == clear {
				t.Fatalf("attempt deadline present = %v, clear = %v", model.hasDeadline, clear)
			}
		})
	}
}

func TestExecutionProcessorCanBeInheritedReplacedOrCleared(t *testing.T) {
	for _, tc := range []struct {
		name     string
		override agent.Optional[loop.ToolResponseProcessor]
		want     string
	}{
		{"inherit", agent.Optional[loop.ToolResponseProcessor]{}, "definition"},
		{"ignore unset value", agent.Optional[loop.ToolResponseProcessor]{Value: &executionProcessor{"unused"}}, "definition"},
		{"replace", agent.Optional[loop.ToolResponseProcessor]{Set: true, Value: &executionProcessor{"run"}}, "run"},
		{"clear", agent.Optional[loop.ToolResponseProcessor]{Set: true}, "original"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &scriptedWorkflowModel{scripts: [][]ai.Token{
				{{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "call", Type: "function", Name: "echo", Args: []byte("{\"text\":\"original\"}")}}}},
				{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "done"}}},
			}}
			a := agent.New(agent.Definition{Model: nativeToolWorkflowModel{model}, Prompt: executionPrompt, Tools: []loop.Tool{loop.NewEchoTool()}, ToolResponseProcessor: &executionProcessor{"definition"}})
			workflow, err := a.NewRun(t.Context(), agent.RunInput{Execution: &agent.ExecutionOverrides{ToolResponseProcessor: tc.override}})
			if err != nil {
				t.Fatal(err)
			}
			if got := consumeWorkflow(t, workflow); len(got.errs) != 0 {
				t.Fatal(got.errs)
			}
			var response string
			for _, part := range workflow.Result().Primary.Iterations[0].Parts {
				if part.ToolResp != nil {
					response = part.ToolResp.TextValue()
				}
			}
			if response != tc.want {
				t.Fatalf("tool output = %q, want %q", response, tc.want)
			}
		})
	}
}

func TestExecutionOverridesStayLocalToPrimaryAgent(t *testing.T) {
	nestedModel := &scriptedWorkflowModel{}
	var nestedInput agent.RunInput
	post := agent.New(agent.Definition{
		Name: "post", Model: nestedModel, Limits: agent.Limits{MaxTokens: 11},
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			nestedInput = input
			return &testPromptBuilder{}, nil
		},
	})
	primaryModel := &scriptedWorkflowModel{}
	primary := agent.New(agent.Definition{
		Model: &scriptedWorkflowModel{}, Prompt: executionPrompt,
		Middleware: []agent.Middleware{agent.NewAgentMiddleware(post, agent.AgentMiddlewareConfig{})},
	})
	workflow, err := primary.NewRun(t.Context(), agent.RunInput{Execution: &agent.ExecutionOverrides{Model: primaryModel, Limits: agent.LimitsOverrides{MaxTokens: executionPtr(4)}}})
	if err != nil {
		t.Fatal(err)
	}
	if got := consumeWorkflow(t, workflow); len(got.errs) != 0 {
		t.Fatal(got.errs)
	}
	if nestedInput.Execution != nil || len(primaryModel.Requests()) != 1 || nestedModel.Requests()[0].MaxTokens != 11 {
		t.Fatal("primary execution overrides leaked into middleware")
	}
}

func TestExecutionToolChoiceOverridesPromptSourceOptions(t *testing.T) {
	for _, existing := range []bool{false, true} {
		t.Run(map[bool]string{false: "automatic source", true: "existing source"}[existing], func(t *testing.T) {
			model := &scriptedWorkflowModel{}
			a := agent.New(agent.Definition{
				Model: model, Tools: []loop.Tool{namedTool{name: "selected"}, namedTool{name: "excluded"}},
				ToolChoice:            ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"selected"}},
				ToolDefinitionOptions: []tooldefinitions.Option{tooldefinitions.WithToolChoice(ai.ToolChoice{Mode: ai.ToolChoiceNone})},
				Prompt: func(ctx context.Context, _ agent.RunInput) (gaictx.PromptBuilder, error) {
					builder := gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}})
					if existing {
						source, err := tooldefinitions.New(nil, []gaictx.ToolSignature{namedTool{name: "stale"}}, nil)
						if err != nil {
							return nil, err
						}
						if err := builder.AppendContextSource(ctx, source); err != nil {
							return nil, err
						}
					}
					return builder, nil
				},
			})
			workflow, err := a.NewRun(t.Context(), agent.RunInput{})
			if err != nil {
				t.Fatal(err)
			}
			// Inspect the request even though this model does not call the required tool.
			consumeWorkflow(t, workflow)
			prompt := requestText(model.Requests()[0])
			if !strings.Contains(prompt, "selected") || strings.Contains(prompt, "excluded") || strings.Contains(prompt, "stale") || !strings.Contains(prompt, "must") {
				t.Fatalf("resolved choice was not applied to prompt: %s", prompt)
			}
		})
	}
}
