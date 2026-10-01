package agent_test

import (
	"context"
	"errors"
	"html"
	"strings"
	"sync"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/tooldefinitions"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/testutil/mocks"
)

type testPromptBuilder struct {
	prompt  string
	input   gaictx.PromptInput
	counter ai.TokenCounter
}

// unclonablePromptBuilder models a fresh third-party builder returned for one
// run. It intentionally does not offer a clone operation.
type unclonablePromptBuilder struct{}

func (*unclonablePromptBuilder) PrependContextSource(context.Context, gaictx.ContextSource) error {
	return nil
}

func (*unclonablePromptBuilder) AppendContextSource(context.Context, gaictx.ContextSource) error {
	return nil
}

func (*unclonablePromptBuilder) AppendContextSources(context.Context, ...gaictx.ContextSource) error {
	return nil
}

func (*unclonablePromptBuilder) AppendSystemInstructions(context.Context, ...gaictx.Part) error {
	return nil
}

func (*unclonablePromptBuilder) BuildContext(context.Context) ([]gaictx.Part, error) { return nil, nil }

func (*unclonablePromptBuilder) BuildPrompt(context.Context, gaictx.Conversation) (string, error) {
	return "", nil
}

func (*unclonablePromptBuilder) Input() gaictx.PromptInput { return gaictx.PromptInput{} }

func (*unclonablePromptBuilder) SetInput(gaictx.PromptInput) {}

// hasToolDefinitionsPromptBuilder models a third-party builder that can report
// its configured tool definitions but cannot replace or remove them.
type hasToolDefinitionsPromptBuilder struct {
	*unclonablePromptBuilder
	prependedSources int
}

func (b *hasToolDefinitionsPromptBuilder) PrependContextSource(context.Context, gaictx.ContextSource) error {
	b.prependedSources++
	return nil
}

func (*hasToolDefinitionsPromptBuilder) HasContextSource(name string) bool {
	return name == "tool_definitions"
}

type testContextSource struct{ name string }

type nativeToolWorkflowModel struct {
	*scriptedWorkflowModel
}

func (nativeToolWorkflowModel) NativeTools() bool { return true }

type disabledNativeToolWorkflowModel struct {
	*scriptedWorkflowModel
}

func (disabledNativeToolWorkflowModel) NativeTools() bool { return false }

type describedToolWorkflowModel struct {
	*scriptedWorkflowModel
	descriptor        ai.ModelDescriptor
	legacyNativeTools bool
}

func (m describedToolWorkflowModel) Descriptor() ai.ModelDescriptor { return m.descriptor }
func (m describedToolWorkflowModel) NativeTools() bool              { return m.legacyNativeTools }
func (m describedToolWorkflowModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	if err := m.descriptor.ValidateRequest(req); err != nil {
		out := make(chan ai.Token, 1)
		out <- ai.Token{Type: ai.TokenTypeErr, Err: err}
		close(out)
		return out
	}
	return m.scriptedWorkflowModel.GenerateStream(ctx, req)
}

func (s testContextSource) Name() string { return s.name }
func (s testContextSource) Function(context.Context, int) (gaictx.Part, error) {
	return gaictx.NewTextPart(s.name), nil
}

func (b *testPromptBuilder) PrependContextSource(ctx context.Context, source gaictx.ContextSource) error {
	return nil
}

func (b *testPromptBuilder) AppendContextSource(ctx context.Context, source gaictx.ContextSource) error {
	return nil
}

func (b *testPromptBuilder) AppendContextSources(ctx context.Context, sources ...gaictx.ContextSource) error {
	return nil
}

func (b *testPromptBuilder) AppendSystemInstructions(ctx context.Context, instructions ...gaictx.Part) error {
	return nil
}

func (b *testPromptBuilder) BuildContext(ctx context.Context) ([]gaictx.Part, error) {
	return nil, nil
}

func (b *testPromptBuilder) BuildPrompt(ctx context.Context, conv gaictx.Conversation) (string, error) {
	return b.prompt, nil
}

func (b *testPromptBuilder) BuildRequest(ctx context.Context, conv gaictx.Conversation) (ai.AIRequest, error) {
	return ai.AIRequest{Prompt: b.prompt, Messages: []ai.Message{ai.TextMessage(ai.RoleUser, b.prompt)}}, nil
}

func (b *testPromptBuilder) Input() gaictx.PromptInput {
	return b.input.Clone()
}

func (b *testPromptBuilder) SetInput(input gaictx.PromptInput) {
	b.input = input.Clone()
	b.prompt = ""
	if input.User != nil {
		b.prompt = (ai.Message{Parts: input.User}).Text()
		return
	}
	if len(input.Context) > 0 && input.Context[0] != nil {
		if node, err := input.Context[0].Render(context.Background()); err == nil {
			b.prompt = node.Value
		}
	}
}

func (b *testPromptBuilder) SetTokenCounter(counter ai.TokenCounter) {
	b.counter = counter
}

func TestAgentNewRunAcceptsFreshUnclonablePromptBuilder(t *testing.T) {
	t.Parallel()

	builder := &unclonablePromptBuilder{}
	assistant := agent.New(agent.Definition{
		Model: &mocks.MockModel{},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	workflow, err := assistant.NewRun(context.Background(), agent.RunInput{})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if workflow == nil {
		t.Fatal("NewRun returned nil workflow")
	}
}

func TestAgentNewRunCreatesLoop(t *testing.T) {
	t.Parallel()

	model := &scriptedWorkflowModel{scripts: [][]ai.Token{{}}}
	tool := loop.NewEchoTool()
	builder := &testPromptBuilder{}
	assistant := agent.New(agent.Definition{
		Name:      "test-agent",
		Model:     nativeToolWorkflowModel{model},
		Tools:     []loop.Tool{tool},
		Prompt:    func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return builder, nil },
		Limits:    agent.Limits{MaxLoopIterations: 2, MaxTokens: 9},
		Reasoning: ai.ReasoningConfig{Enabled: true, IncludeThoughts: true, BudgetTokens: 128, Effort: ai.ReasoningEffortHigh},
	})

	workflow, err := assistant.NewRun(context.Background(), textRunInput("input"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if got := consumeWorkflow(t, workflow); len(got.errs) != 0 {
		t.Fatalf("workflow errors: %v", got.errs)
	}
	requests := model.Requests()
	if len(requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(requests))
	}
	request := requests[0]
	if request.MaxTokens != 9 || request.Reasoning != (ai.ReasoningConfig{Enabled: true, IncludeThoughts: true, BudgetTokens: 128, Effort: ai.ReasoningEffortHigh}) {
		t.Fatalf("request configuration = %+v", request)
	}
	if len(request.Tools) != 1 || request.Tools[0].Name != tool.Name() {
		t.Fatalf("request tools = %+v", request.Tools)
	}
	if builder.counter == nil {
		t.Fatal("expected model counter to be set on prompt builder")
	}
}

func TestAgentNewRunSnapshotsRetryPolicy(t *testing.T) {
	t.Parallel()

	policy := &loop.RetryPolicy{MaxRetries: 1}
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{
		{{Type: ai.TokenTypeErr, Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("retry first")}}},
		{},
		{{Type: ai.TokenTypeErr, Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("retry second")}}},
		{},
	}}
	assistant := agent.New(agent.Definition{
		Model:       model,
		RetryPolicy: policy,
		Prompt:      func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return &testPromptBuilder{}, nil },
	})
	first, err := assistant.NewRun(context.Background(), textRunInput("first"))
	if err != nil {
		t.Fatalf("first NewRun failed: %v", err)
	}
	second, err := assistant.NewRun(context.Background(), textRunInput("second"))
	if err != nil {
		t.Fatalf("second NewRun failed: %v", err)
	}
	policy.MaxRetries = 0
	for name, workflow := range map[string]*agent.Workflow{"first": first, "second": second} {
		if got := consumeWorkflow(t, workflow); len(got.errs) != 0 {
			t.Fatalf("%s workflow errors: %v", name, got.errs)
		}
	}
	if requests := model.Requests(); len(requests) != 4 {
		t.Fatalf("requests = %d, want 4 after two independent retries", len(requests))
	}
}

func TestAgentNewRunPreservesNilAndEmptyExecutionToolOverrides(t *testing.T) {
	t.Parallel()

	definitionTool := loop.NewEchoTool()
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{{}, {}}}
	assistant := agent.New(agent.Definition{
		Model: nativeToolWorkflowModel{scriptedWorkflowModel: model},
		Tools: []loop.Tool{definitionTool},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})

	inherited, err := assistant.NewRun(context.Background(), textRunInput("inherit tools"))
	if err != nil {
		t.Fatalf("NewRun with nil tools failed: %v", err)
	}
	if got := consumeWorkflow(t, inherited); len(got.errs) != 0 {
		t.Fatalf("inherited workflow errors: %v", got.errs)
	}

	disabledInput := textRunInput("disable tools")
	disabledInput.Execution = &agent.ExecutionOverrides{}
	disabledInput.Execution.Tools = []loop.Tool{}
	disabled, err := assistant.NewRun(context.Background(), disabledInput)
	if err != nil {
		t.Fatalf("NewRun with empty tools failed: %v", err)
	}
	if got := consumeWorkflow(t, disabled); len(got.errs) != 0 {
		t.Fatalf("disabled workflow errors: %v", got.errs)
	}
	requests := model.Requests()
	if len(requests) != 2 || len(requests[0].Tools) != 1 || requests[0].Tools[0].Name != definitionTool.Name() || len(requests[1].Tools) != 0 {
		t.Fatalf("tool override requests = %#v", requests)
	}
}

func TestAgentNewRunExecutionOverridesSnapshotToolsAndConfiguration(t *testing.T) {
	t.Parallel()

	definitionTool := loop.NewEchoTool()
	runTool := namedTool{name: "run_tool"}
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "run_tool", Args: []byte(`{}`)}}},
		{},
	}}
	assistant := agent.New(agent.Definition{
		Model:     nativeToolWorkflowModel{scriptedWorkflowModel: model},
		Tools:     []loop.Tool{definitionTool},
		Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortLow},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})
	input := textRunInput("hello")
	input.Execution = &agent.ExecutionOverrides{
		Tools:      []loop.Tool{runTool},
		ToolChoice: &ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"run_tool"}},
		Reasoning:  &ai.ReasoningConfig{Enabled: true, Effort: ai.ReasoningEffortHigh},
	}

	workflow, err := assistant.NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	input.Execution.Tools[0] = definitionTool
	input.Execution.ToolChoice.Names[0] = "changed"
	input.Execution.Reasoning.Effort = ai.ReasoningEffortLow

	if got := consumeWorkflow(t, workflow); len(got.errs) != 0 {
		t.Fatalf("workflow errors: %v", got.errs)
	}
	requests := model.Requests()
	if len(requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(requests))
	}
	if request := requests[0]; len(request.Tools) != 1 || request.Tools[0].Name != "run_tool" || request.ToolChoice.Mode != ai.ToolChoiceRequired || len(request.ToolChoice.Names) != 1 || request.ToolChoice.Names[0] != "run_tool" || request.Reasoning != (ai.ReasoningConfig{Enabled: true, Effort: ai.ReasoningEffortHigh}) {
		t.Fatalf("execution override request = %#v", request)
	}
}

func TestAgentNewRunRejectsDuplicateExecutionToolNamesBeforeRun(t *testing.T) {
	t.Parallel()

	tool := loop.NewEchoTool()
	assistant := agent.New(agent.Definition{
		Model: nativeToolWorkflowModel{scriptedWorkflowModel: &scriptedWorkflowModel{}},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})
	input := textRunInput("hello")
	input.Execution = &agent.ExecutionOverrides{}
	input.Execution.Tools = []loop.Tool{tool, tool}

	_, err := assistant.NewRun(context.Background(), input)
	if err == nil || !strings.Contains(err.Error(), "duplicate tool name") {
		t.Fatalf("NewRun error = %v, want duplicate tool name", err)
	}
}

func TestAgentConcurrentRunsUseIndependentExecutionToolSets(t *testing.T) {
	t.Parallel()

	model := nativeToolWorkflowModel{scriptedWorkflowModel: &scriptedWorkflowModel{
		scripts: [][]ai.Token{{{Type: ai.TokenTypeText, Text: "done"}}, {{Type: ai.TokenTypeText, Text: "done"}}},
	}}
	assistant := agent.New(agent.Definition{
		Model: model,
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})

	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for _, name := range []string{"first", "second"} {
		wg.Add(1)
		go func(name string) {
			defer wg.Done()
			input := textRunInput("hello")
			input.Execution = &agent.ExecutionOverrides{}
			input.Execution.Tools = []loop.Tool{namedTool{name: name}}
			workflow, err := assistant.NewRun(context.Background(), input)
			if err == nil {
				consumeWorkflow(t, workflow)
			}
			errs <- err
		}(name)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("NewRun failed: %v", err)
		}
	}

	requests := model.Requests()
	if len(requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(requests))
	}
	names := map[string]bool{}
	for _, request := range requests {
		if len(request.Tools) != 1 {
			t.Fatalf("request tools = %#v", request.Tools)
		}
		names[request.Tools[0].Name] = true
	}
	if !names["first"] || !names["second"] {
		t.Fatalf("request tool names = %#v", names)
	}
}

type namedTool struct{ name string }

func (t namedTool) Name() string        { return t.name }
func (t namedTool) Description() string { return "test tool" }
func (t namedTool) Params() ai.ToolParameters {
	return ai.ToolParameters{}
}
func (t namedTool) Function(context.Context, *ai.ToolCall) *loop.ToolResponse {
	return loop.NewToolSuccess("ok")
}

type recordingTool struct {
	name  string
	calls int
}

func (t *recordingTool) Name() string        { return t.name }
func (t *recordingTool) Description() string { return "test tool" }
func (t *recordingTool) Params() ai.ToolParameters {
	return ai.ToolParameters{}
}
func (t *recordingTool) Function(context.Context, *ai.ToolCall) *loop.ToolResponse {
	t.calls++
	return loop.NewToolSuccess("called")
}

func toolSignatures(tools []loop.Tool) []gaictx.ToolSignature {
	signatures := make([]gaictx.ToolSignature, len(tools))
	for i, tool := range tools {
		signatures[i] = tool
	}
	return signatures
}

func TestAgentToolsAutomaticallyAddPromptContract(t *testing.T) {
	t.Parallel()

	builder := gaictx.New(gaictx.Definition{
		Renderer:       &gaictx.SimpleRenderer{},
		ContextSources: []gaictx.ContextSource{testContextSource{name: "application_context"}},
	})
	assistant := agent.New(agent.Definition{
		Model: &mocks.MockModel{},
		Tools: []loop.Tool{loop.NewEchoTool()},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	_, err := assistant.NewRun(context.Background(), textRunInput("remember my name"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	runBuilder := builder
	if len(runBuilder.ContextSources) != 2 || runBuilder.ContextSources[0].Name() != "tool_definitions" || runBuilder.ContextSources[1].Name() != "application_context" {
		t.Fatalf("tool definitions were not prepended: %+v", runBuilder.ContextSources)
	}
	if _, err := builder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := builder.BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	for _, expected := range []string{
		"tool: echo",
		`{"type":"function","name":"<tool-name>","arguments":{...}}`,
	} {
		if !strings.Contains(html.UnescapeString(prompt), expected) {
			t.Fatalf("automatic tool prompt missing %q:\n%s", expected, prompt)
		}
	}
}

func TestAgentNativeToolModelOmitsPromptToolProtocol(t *testing.T) {
	t.Parallel()

	builder := gaictx.New(gaictx.Definition{
		Renderer:       &gaictx.SimpleRenderer{},
		ContextSources: []gaictx.ContextSource{testContextSource{name: "application_context"}},
	})
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{{}}}
	assistant := agent.New(agent.Definition{
		Model: nativeToolWorkflowModel{model},
		Tools: []loop.Tool{loop.NewEchoTool()},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	workflow, err := assistant.NewRun(context.Background(), textRunInput("use echo"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if len(builder.ContextSources) != 1 || builder.ContextSources[0].Name() != "application_context" {
		t.Fatalf("native tool model added prompt tool protocol: %+v", builder.ContextSources)
	}

	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 0 {
		t.Fatalf("workflow errors: %v", consumed.errs)
	}
	requests := model.Requests()
	if len(requests) != 1 {
		t.Fatalf("model received %d requests, want 1", len(requests))
	}
	if len(requests[0].Tools) != 1 || requests[0].Tools[0].Name != "echo" {
		t.Fatalf("native tool model request did not include tool definition: %+v", requests[0])
	}
	if strings.Contains(html.UnescapeString(requests[0].Prompt), `{"type":"function","name":"<tool-name>","arguments":{...}}`) {
		t.Fatalf("native tool model request included prompt tool protocol:\n%s", requests[0].Prompt)
	}
}

func TestAgentNativeToolModelWithoutSupportAddsPromptToolProtocol(t *testing.T) {
	t.Parallel()

	builder := gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}})
	assistant := agent.New(agent.Definition{
		Model: disabledNativeToolWorkflowModel{&scriptedWorkflowModel{}},
		Tools: []loop.Tool{loop.NewEchoTool()},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	_, err := assistant.NewRun(context.Background(), textRunInput("use echo"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	runBuilder := builder
	if len(runBuilder.ContextSources) != 1 || runBuilder.ContextSources[0].Name() != "tool_definitions" {
		t.Fatalf("disabled native tool model did not add prompt tool protocol: %+v", runBuilder.ContextSources)
	}
	if _, err := builder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := builder.BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if !strings.Contains(html.UnescapeString(prompt), `{"type":"function","name":"<tool-name>","arguments":{...}}`) {
		t.Fatalf("disabled native tool model prompt missing tool protocol:\n%s", prompt)
	}
}

func TestAgentModelDescriberControlsPromptToolProtocol(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name               string
		nativeTools        ai.FeatureSupport
		legacyNativeTools  bool
		wantPromptProtocol bool
	}{
		{name: "supported overrides legacy false", nativeTools: ai.FeatureSupportSupported},
		{name: "unknown ignores legacy support", nativeTools: ai.FeatureSupportUnknown, legacyNativeTools: true, wantPromptProtocol: true},
		{name: "unsupported ignores legacy support", nativeTools: ai.FeatureSupportUnsupported, legacyNativeTools: true, wantPromptProtocol: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}})
			assistant := agent.New(agent.Definition{
				Model: describedToolWorkflowModel{
					scriptedWorkflowModel: &scriptedWorkflowModel{},
					descriptor:            ai.ModelDescriptor{NativeTools: tt.nativeTools},
					legacyNativeTools:     tt.legacyNativeTools,
				},
				Tools: []loop.Tool{loop.NewEchoTool()},
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
					return builder, nil
				},
			})

			_, err := assistant.NewRun(context.Background(), textRunInput("use echo"))
			if err != nil {
				t.Fatalf("NewRun failed: %v", err)
			}
			runBuilder := builder
			hasPromptProtocol := len(runBuilder.ContextSources) == 1 && runBuilder.ContextSources[0].Name() == "tool_definitions"
			if hasPromptProtocol != tt.wantPromptProtocol {
				t.Fatalf("prompt protocol = %t, want %t; sources: %+v", hasPromptProtocol, tt.wantPromptProtocol, runBuilder.ContextSources)
			}
		})
	}
}

func TestAgentResolvesToolTransportForPromptAndRequest(t *testing.T) {
	tests := []struct {
		name               string
		described          bool
		nativeTools        ai.FeatureSupport
		legacyNativeTools  bool
		wantPromptProtocol bool
		wantRequestTools   bool
	}{
		{name: "descriptor supported", described: true, nativeTools: ai.FeatureSupportSupported, wantRequestTools: true},
		{name: "descriptor unknown", described: true, nativeTools: ai.FeatureSupportUnknown, legacyNativeTools: true, wantPromptProtocol: true},
		{name: "descriptor unsupported", described: true, nativeTools: ai.FeatureSupportUnsupported, legacyNativeTools: true, wantPromptProtocol: true},
		{name: "legacy native model", legacyNativeTools: true, wantRequestTools: true},
		{name: "legacy text model", wantPromptProtocol: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}})
			model := &scriptedWorkflowModel{scripts: [][]ai.Token{{}}}
			var configuredModel ai.Model = model
			if tt.described {
				configuredModel = describedToolWorkflowModel{
					scriptedWorkflowModel: model,
					descriptor:            ai.ModelDescriptor{NativeTools: tt.nativeTools},
					legacyNativeTools:     tt.legacyNativeTools,
				}
			} else if tt.legacyNativeTools {
				configuredModel = nativeToolWorkflowModel{model}
			}

			assistant := agent.New(agent.Definition{
				Model: configuredModel,
				Tools: []loop.Tool{loop.NewEchoTool()},
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
					return builder, nil
				},
			})
			workflow, err := assistant.NewRun(context.Background(), textRunInput("use echo"))
			if err != nil {
				t.Fatalf("NewRun failed: %v", err)
			}
			consumed := consumeWorkflow(t, workflow)
			if len(consumed.errs) != 0 {
				t.Fatalf("workflow errors: %v", consumed.errs)
			}
			runBuilder := builder
			if got := len(runBuilder.ContextSources) == 1 && runBuilder.ContextSources[0].Name() == "tool_definitions"; got != tt.wantPromptProtocol {
				t.Fatalf("prompt protocol = %t, want %t; sources: %+v", got, tt.wantPromptProtocol, runBuilder.ContextSources)
			}
			requests := model.Requests()
			if len(requests) != 1 {
				t.Fatalf("requests = %d, want 1", len(requests))
			}
			if got := len(requests[0].Tools) > 0; got != tt.wantRequestTools {
				t.Fatalf("request tools = %#v, want present=%t", requests[0].Tools, tt.wantRequestTools)
			}
		})
	}
}

func TestAgentToolDefinitionOptionsCustomizeAutomaticPromptContract(t *testing.T) {
	t.Parallel()

	builder := gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}})
	assistant := agent.New(agent.Definition{
		Model:                 &mocks.MockModel{},
		Tools:                 []loop.Tool{loop.NewEchoTool()},
		ToolDefinitionOptions: []tooldefinitions.Option{tooldefinitions.WithUsageProtocol("Use tools only after asking for confirmation.")},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	_, err := assistant.NewRun(context.Background(), textRunInput("remember my name"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if _, err := builder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := builder.BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if !strings.Contains(prompt, "Use tools only after asking for confirmation.") {
		t.Fatalf("custom tool definition protocol missing:\n%s", prompt)
	}
	if strings.Contains(html.UnescapeString(prompt), `{"type":"function","name":"<tool-name>","arguments":{...}}`) {
		t.Fatalf("default tool definition protocol still present:\n%s", prompt)
	}
}

func TestAgentTextTransportRequiresSelectedToolInPrompt(t *testing.T) {
	t.Parallel()

	builder := gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}})
	assistant := agent.New(agent.Definition{
		Model: disabledNativeToolWorkflowModel{&scriptedWorkflowModel{}},
		Tools: []loop.Tool{
			namedTool{name: "search"},
			namedTool{name: "weather"},
		},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	input := textRunInput("find the weather")
	input.Execution = &agent.ExecutionOverrides{}
	input.Execution.ToolChoice = &ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"weather"}}
	_, err := assistant.NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if _, err := builder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := builder.BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if !strings.Contains(prompt, "You must make at least one tool call before producing a normal response.") {
		t.Fatalf("prompt missing required-tool instruction:\n%s", prompt)
	}
	if !strings.Contains(prompt, "You may call only these selected tools: weather.") {
		t.Fatalf("prompt missing selected-tool restriction:\n%s", prompt)
	}
}

func TestAgentTextTransportSelectedToolsReplaceExistingPromptToolDefinitions(t *testing.T) {
	t.Parallel()

	search := namedTool{name: "search"}
	weather := namedTool{name: "weather"}
	staleSource, err := tooldefinitions.New(&gaictx.SimpleRenderer{}, toolSignatures([]loop.Tool{search, weather}), nil)
	if err != nil {
		t.Fatalf("new stale tool source: %v", err)
	}
	builder := gaictx.New(gaictx.Definition{
		Renderer:       &gaictx.SimpleRenderer{},
		ContextSources: []gaictx.ContextSource{staleSource},
	})
	assistant := agent.New(agent.Definition{
		Model: disabledNativeToolWorkflowModel{&scriptedWorkflowModel{}},
		Tools: []loop.Tool{search, weather},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	input := textRunInput("find the weather")
	input.Execution = &agent.ExecutionOverrides{}
	input.Execution.ToolChoice = &ai.ToolChoice{
		Mode:  ai.ToolChoiceRequired,
		Names: []string{"weather"},
	}
	_, err = assistant.NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if _, err := builder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := builder.BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if strings.Contains(prompt, "tool: search") {
		t.Fatalf("prompt advertised unselected tool:\n%s", prompt)
	}
	if !strings.Contains(prompt, "tool: weather") {
		t.Fatalf("prompt omitted selected tool:\n%s", prompt)
	}
}

func TestAgentTextTransportDoesNotExecuteUnselectedRequiredTool(t *testing.T) {
	selected := &recordingTool{name: "weather"}
	unselected := &recordingTool{name: "search"}
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{
		{{
			Type: ai.TokenTypeToolCall,
			ToolCall: &ai.ToolCall{
				ID:   "call-1",
				Type: "function",
				Name: "search",
				Args: []byte(`{}`),
			},
		}},
		{},
	}}
	assistant := agent.New(agent.Definition{
		Model: disabledNativeToolWorkflowModel{model},
		Tools: []loop.Tool{selected, unselected},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}}), nil
		},
	})
	input := textRunInput("use weather")
	input.Execution = &agent.ExecutionOverrides{}
	input.Execution.ToolChoice = &ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"weather"}}

	workflow, err := assistant.NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 1 || !errors.Is(consumed.errs[0], loop.ErrMaxIterations) {
		t.Fatalf("workflow errors = %v, want ErrMaxIterations after no selected tool call", consumed.errs)
	}
	if unselected.calls != 0 {
		t.Fatalf("unselected tool was called %d times", unselected.calls)
	}
	if selected.calls != 0 {
		t.Fatalf("selected tool was called %d times", selected.calls)
	}
}

func TestAgentTextTransportDoesNotAdvertiseOrExecuteDisabledTools(t *testing.T) {
	disabled := &recordingTool{name: "search"}
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "search", Args: []byte(`{}`)}}},
		{},
	}}
	builder := gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}})
	assistant := agent.New(agent.Definition{
		Model:  disabledNativeToolWorkflowModel{model},
		Tools:  []loop.Tool{disabled},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return builder, nil },
	})
	input := textRunInput("do not use tools")
	input.Execution = &agent.ExecutionOverrides{}
	input.Execution.ToolChoice = &ai.ToolChoice{Mode: ai.ToolChoiceNone}

	workflow, err := assistant.NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if _, err := builder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := builder.BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if strings.Contains(prompt, "tool: search") {
		t.Fatalf("prompt advertised disabled tool:\n%s", prompt)
	}
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 0 {
		t.Fatalf("workflow errors: %v", consumed.errs)
	}
	if disabled.calls != 0 {
		t.Fatalf("disabled tool was called %d times", disabled.calls)
	}
}

func TestAgentTextTransportRejectsUnsatisfiableRequiredToolChoice(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		name   string
		tools  []loop.Tool
		choice ai.ToolChoice
	}{
		{
			name:   "no tools",
			tools:  nil,
			choice: ai.ToolChoice{Mode: ai.ToolChoiceRequired},
		},
		{
			name:   "selected tool is unavailable",
			tools:  []loop.Tool{namedTool{name: "search"}},
			choice: ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"weather"}},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			assistant := agent.New(agent.Definition{
				Model: disabledNativeToolWorkflowModel{&scriptedWorkflowModel{}},
				Tools: tt.tools,
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
					return gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}}), nil
				},
			})
			input := textRunInput("use a tool")
			input.Execution = &agent.ExecutionOverrides{}
			input.Execution.ToolChoice = &tt.choice

			if _, err := assistant.NewRun(context.Background(), input); err == nil {
				t.Fatal("NewRun succeeded with an unsatisfiable required tool choice")
			}
		})
	}
}

func TestAgentRejectsInvalidToolChoiceMode(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		name  string
		model ai.Model
	}{
		{
			name:  "native transport",
			model: nativeToolWorkflowModel{scriptedWorkflowModel: &scriptedWorkflowModel{}},
		},
		{
			name:  "text transport",
			model: disabledNativeToolWorkflowModel{&scriptedWorkflowModel{}},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			assistant := agent.New(agent.Definition{
				Model: tt.model,
				Tools: []loop.Tool{namedTool{name: "search"}},
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
					return gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}}), nil
				},
			})
			input := textRunInput("use a tool")
			input.Execution = &agent.ExecutionOverrides{}
			input.Execution.ToolChoice = &ai.ToolChoice{Mode: ai.ToolChoiceMode("requred")}

			if _, err := assistant.NewRun(context.Background(), input); err == nil {
				t.Fatal("NewRun succeeded with an invalid tool choice mode")
			}
		})
	}
}

func TestAgentDoesNotDuplicateExistingToolDefinitions(t *testing.T) {
	t.Parallel()

	tool := loop.NewEchoTool()
	source, err := tooldefinitions.New(&gaictx.SimpleRenderer{}, toolSignatures([]loop.Tool{tool}), nil)
	if err != nil {
		t.Fatalf("new tool source: %v", err)
	}
	builder := gaictx.New(gaictx.Definition{
		Renderer:       &gaictx.SimpleRenderer{},
		ContextSources: []gaictx.ContextSource{source},
	})
	assistant := agent.New(agent.Definition{
		Model: &mocks.MockModel{},
		Tools: []loop.Tool{tool},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	_, err = assistant.NewRun(context.Background(), agent.RunInput{})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if _, err := builder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := builder.BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if got := strings.Count(prompt, "tool: echo"); got != 1 {
		t.Fatalf("tool definitions rendered %d times:\n%s", got, prompt)
	}
}

func TestAgentPreservesExistingToolDefinitionsFromLookupOnlyBuilder(t *testing.T) {
	t.Parallel()

	builder := &hasToolDefinitionsPromptBuilder{unclonablePromptBuilder: &unclonablePromptBuilder{}}
	assistant := agent.New(agent.Definition{
		Model: &mocks.MockModel{},
		Tools: []loop.Tool{loop.NewEchoTool()},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	if _, err := assistant.NewRun(context.Background(), agent.RunInput{}); err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if builder.prependedSources != 0 {
		t.Fatalf("tool definitions were prepended %d times despite an existing source", builder.prependedSources)
	}
}

func TestAgentExecutionToolsReplaceExistingPromptToolDefinitions(t *testing.T) {
	t.Parallel()

	definitionTool := namedTool{name: "definition_tool"}
	runTool := namedTool{name: "run_tool"}
	staleSource, err := tooldefinitions.New(&gaictx.SimpleRenderer{}, toolSignatures([]loop.Tool{definitionTool}), nil)
	if err != nil {
		t.Fatalf("new stale tool source: %v", err)
	}
	builder := gaictx.New(gaictx.Definition{
		Renderer:       &gaictx.SimpleRenderer{},
		ContextSources: []gaictx.ContextSource{staleSource},
	})
	assistant := agent.New(agent.Definition{
		Model: disabledNativeToolWorkflowModel{&scriptedWorkflowModel{}},
		Tools: []loop.Tool{definitionTool},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	input := textRunInput("use the run tool")
	input.Execution = &agent.ExecutionOverrides{}
	input.Execution.Tools = []loop.Tool{runTool}
	_, err = assistant.NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	runBuilder := builder
	if len(runBuilder.ContextSources) != 1 || runBuilder.ContextSources[0].Name() != "tool_definitions" {
		t.Fatalf("context sources = %+v, want one tool-definitions source", runBuilder.ContextSources)
	}
	if _, err := builder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := builder.BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if strings.Contains(prompt, "tool: definition_tool") {
		t.Fatalf("prompt retained definition-level tool:\n%s", prompt)
	}
	if !strings.Contains(prompt, "tool: run_tool") {
		t.Fatalf("prompt missing run-level tool:\n%s", prompt)
	}
}

func TestAgentExecutionToolOverridesUseRunOwnedPromptBuilders(t *testing.T) {
	t.Parallel()

	definitionTool := namedTool{name: "definition_tool"}
	runTool := namedTool{name: "run_tool"}
	staleSource, err := tooldefinitions.New(&gaictx.SimpleRenderer{}, toolSignatures([]loop.Tool{definitionTool}), nil)
	if err != nil {
		t.Fatalf("new stale tool source: %v", err)
	}
	var builders []*gaictx.Builder
	assistant := agent.New(agent.Definition{
		Model: disabledNativeToolWorkflowModel{&scriptedWorkflowModel{}},
		Tools: []loop.Tool{definitionTool},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			builder := gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}, ContextSources: []gaictx.ContextSource{staleSource}})
			builders = append(builders, builder)
			return builder, nil
		},
	})

	firstInput := textRunInput("use the run tool")
	firstInput.Execution = &agent.ExecutionOverrides{}
	firstInput.Execution.Tools = []loop.Tool{runTool}
	if _, err := assistant.NewRun(context.Background(), firstInput); err != nil {
		t.Fatalf("first NewRun failed: %v", err)
	}
	secondInput := textRunInput("do not use tools")
	secondInput.Execution = &agent.ExecutionOverrides{}
	secondInput.Execution.Tools = []loop.Tool{}
	if _, err := assistant.NewRun(context.Background(), secondInput); err != nil {
		t.Fatalf("second NewRun failed: %v", err)
	}
	if len(builders) != 2 {
		t.Fatalf("prompt builders = %d, want 2", len(builders))
	}
	for name, builder := range map[string]*gaictx.Builder{"first": builders[0], "second": builders[1]} {
		if _, err := builder.BuildContext(context.Background()); err != nil {
			t.Fatalf("%s BuildContext failed: %v", name, err)
		}
	}
	firstPrompt, err := builders[0].BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("first BuildPrompt failed: %v", err)
	}
	secondPrompt, err := builders[1].BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("second BuildPrompt failed: %v", err)
	}
	if strings.Contains(firstPrompt, "tool: definition_tool") || !strings.Contains(firstPrompt, "tool: run_tool") {
		t.Fatalf("first prompt tool definitions = %q", firstPrompt)
	}
	if strings.Contains(secondPrompt, "tool: definition_tool") || strings.Contains(secondPrompt, "tool: run_tool") {
		t.Fatalf("second prompt tool definitions = %q", secondPrompt)
	}
}

func TestAgentExecutionEmptyToolsRemoveExistingPromptToolDefinitions(t *testing.T) {
	t.Parallel()

	definitionTool := namedTool{name: "definition_tool"}
	staleSource, err := tooldefinitions.New(&gaictx.SimpleRenderer{}, toolSignatures([]loop.Tool{definitionTool}), nil)
	if err != nil {
		t.Fatalf("new stale tool source: %v", err)
	}
	builder := gaictx.New(gaictx.Definition{
		Renderer:       &gaictx.SimpleRenderer{},
		ContextSources: []gaictx.ContextSource{staleSource},
	})
	assistant := agent.New(agent.Definition{
		Model: disabledNativeToolWorkflowModel{&scriptedWorkflowModel{}},
		Tools: []loop.Tool{definitionTool},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	input := textRunInput("do not use tools")
	input.Execution = &agent.ExecutionOverrides{}
	input.Execution.Tools = []loop.Tool{}
	_, err = assistant.NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	runBuilder := builder
	if len(runBuilder.ContextSources) != 0 {
		t.Fatalf("context sources = %+v, want no tool definitions", runBuilder.ContextSources)
	}
	if _, err := builder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := builder.BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if strings.Contains(prompt, "tool: definition_tool") {
		t.Fatalf("prompt retained definition-level tool:\n%s", prompt)
	}
}

func TestAgentNewRunUsesInputMaxTokens(t *testing.T) {
	t.Parallel()

	model := &scriptedWorkflowModel{scripts: [][]ai.Token{{}}}
	assistant := agent.New(agent.Definition{
		Model:  model,
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return &testPromptBuilder{}, nil },
		Limits: agent.Limits{MaxTokens: 9},
	})

	input := textRunInput("input")
	maxTokens := 3
	input.Execution = &agent.ExecutionOverrides{Limits: agent.LimitsOverrides{MaxTokens: &maxTokens}}
	workflow, err := assistant.NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if got := consumeWorkflow(t, workflow); len(got.errs) != 0 {
		t.Fatalf("workflow errors: %v", got.errs)
	}
	requests := model.Requests()
	if len(requests) != 1 || requests[0].MaxTokens != 3 {
		t.Fatalf("input max tokens request = %#v", requests)
	}
}

func textRunInput(text string) agent.RunInput {
	return agent.RunInput{Prompt: gaictx.PromptInput{User: ai.TextParts(text)}}
}

func promptUserText(input agent.RunInput) string {
	if input.Prompt.User == nil {
		return ""
	}
	return (ai.Message{Parts: input.Prompt.User}).Text()
}

func promptContextValue(input agent.RunInput, name string) string {
	for _, part := range input.Prompt.Context {
		if part == nil || part.Name() != name {
			continue
		}
		node, err := part.Render(context.Background())
		if err == nil {
			return node.Value
		}
	}
	return ""
}

func TestAgentNewRunUsesConfiguredTokenCounterOverride(t *testing.T) {
	t.Parallel()

	modelTokenCounter := &mocks.MockTokenCounter{IDValue: "model"}
	overrideTokenCounter := &mocks.MockTokenCounter{IDValue: "override"}
	builder := &testPromptBuilder{}
	assistant := agent.New(agent.Definition{
		Model:        &mocks.MockModel{TokenCounterValue: modelTokenCounter},
		TokenCounter: overrideTokenCounter,
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	_, err := assistant.NewRun(context.Background(), agent.RunInput{})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	runBuilder := builder
	if runBuilder.counter != overrideTokenCounter {
		t.Fatalf("expected configured counter override, got %v", runBuilder.counter)
	}
}

func TestAgentNewRunFallsBackToModelTokenCounter(t *testing.T) {
	t.Parallel()

	modelTokenCounter := &mocks.MockTokenCounter{IDValue: "model"}
	builder := &testPromptBuilder{}
	assistant := agent.New(agent.Definition{
		Model: &mocks.MockModel{TokenCounterValue: modelTokenCounter},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	_, err := assistant.NewRun(context.Background(), agent.RunInput{})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	runBuilder := builder
	if runBuilder.counter != modelTokenCounter {
		t.Fatalf("expected model counter fallback, got %v", runBuilder.counter)
	}
}

func TestAgentNewRunRequiresModelAndPrompt(t *testing.T) {
	t.Parallel()

	_, err := agent.New(agent.Definition{}).NewRun(context.Background(), agent.RunInput{})
	if err != loop.ErrModelNotConfigured {
		t.Fatalf("expected ErrModelNotConfigured, got %v", err)
	}

	_, err = agent.New(agent.Definition{Model: &mocks.MockModel{}}).NewRun(context.Background(), agent.RunInput{})
	if err != loop.ErrPromptNotConfigured {
		t.Fatalf("expected ErrPromptNotConfigured, got %v", err)
	}

	_, err = agent.New(agent.Definition{
		Model:  &mocks.MockModel{},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return nil, nil },
	}).NewRun(context.Background(), agent.RunInput{})
	if err != loop.ErrPromptNotConfigured {
		t.Fatalf("expected ErrPromptNotConfigured for nil builder, got %v", err)
	}
}

func TestAgentNewRunReturnsPromptError(t *testing.T) {
	t.Parallel()

	promptErr := errors.New("prompt failed")
	_, err := agent.New(agent.Definition{
		Model: &mocks.MockModel{},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return nil, promptErr
		},
	}).NewRun(context.Background(), agent.RunInput{})
	if !errors.Is(err, promptErr) {
		t.Fatalf("expected prompt error, got %v", err)
	}
}

func (*unclonablePromptBuilder) BuildRequest(context.Context, gaictx.Conversation) (ai.AIRequest, error) {
	return ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "")}}, nil
}
