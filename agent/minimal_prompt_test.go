package agent_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/tooldefinitions"
	"github.com/lace-ai/gai/loop"
)

// readOnlyPromptBuilder provides only the loop's construction contract.
type readOnlyPromptBuilder struct{ input gaictx.PromptInput }

func (*readOnlyPromptBuilder) BuildContext(context.Context) ([]gaictx.Part, error) { return nil, nil }
func (b *readOnlyPromptBuilder) BuildPrompt(context.Context, gaictx.Conversation) (string, error) {
	if b.input.User == nil {
		return "", nil
	}
	return (ai.Message{Parts: b.input.User}).Text(), nil
}
func (b *readOnlyPromptBuilder) Input() gaictx.PromptInput { return b.input.Clone() }

// inputOnlyPromptBuilder adds run input configuration without source mutation.
type inputOnlyPromptBuilder struct{ readOnlyPromptBuilder }

func (b *inputOnlyPromptBuilder) SetInput(input gaictx.PromptInput) { b.input = input.Clone() }

type sourceLookupPromptBuilder struct{ inputOnlyPromptBuilder }

func (*sourceLookupPromptBuilder) HasContextSource(name string) bool {
	return name == "tool_definitions"
}

func TestAgentRejectsPromptBuilderWithoutInputSetter(t *testing.T) {
	t.Parallel()
	model := &scriptedWorkflowModel{}
	assistant := agent.New(agent.Definition{
		Model: model,
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return &readOnlyPromptBuilder{}, nil
		},
	})
	workflow, err := assistant.NewRun(t.Context(), textRunInput("must not be lost"))
	if !errors.Is(err, agent.ErrPromptInputNotConfigurable) || workflow != nil {
		t.Fatalf("NewRun = (%v, %v), want ErrPromptInputNotConfigurable", workflow, err)
	}
	if len(model.Requests()) != 0 {
		t.Fatal("model ran despite invalid prompt builder")
	}
}

func TestAgentAcceptsInputOnlyPromptBuilder(t *testing.T) {
	t.Parallel()
	for _, native := range []bool{false, true} {
		t.Run(map[bool]string{false: "no tools", true: "native tools"}[native], func(t *testing.T) {
			model := &scriptedWorkflowModel{scripts: [][]ai.Token{{{Type: ai.TokenTypeText, Data: []byte("answer")}}}}
			builder := &inputOnlyPromptBuilder{}
			definition := agent.Definition{
				Model:  model,
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return builder, nil },
			}
			if native {
				definition.Model = nativeToolWorkflowModel{model}
				definition.Tools = []loop.Tool{loop.NewEchoTool()}
			}
			workflow, err := agent.New(definition).NewRun(t.Context(), textRunInput("question"))
			if err != nil {
				t.Fatalf("NewRun: %v", err)
			}
			if consumed := consumeWorkflow(t, workflow); len(consumed.errs) != 0 {
				t.Fatalf("workflow errors: %v", consumed.errs)
			}
			requests := model.Requests()
			if len(requests) != 1 || len(requests[0].Messages) != 1 || requests[0].Messages[0].Text() != "question" {
				t.Fatalf("requests = %#v, want run input", requests)
			}
			if native && (len(requests[0].Tools) != 1 || requests[0].Tools[0].Name != "echo") {
				t.Fatalf("request omitted native tools: %#v", requests[0])
			}
		})
	}
}

func TestAgentRequiresSourcePrependingOnlyForNewTextToolDefinitions(t *testing.T) {
	t.Parallel()
	for _, existing := range []bool{false, true} {
		t.Run(map[bool]string{false: "missing source", true: "existing source"}[existing], func(t *testing.T) {
			model := &scriptedWorkflowModel{}
			assistant := agent.New(agent.Definition{
				Model: model,
				Tools: []loop.Tool{loop.NewEchoTool()},
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
					if existing {
						return &sourceLookupPromptBuilder{}, nil
					}
					return &inputOnlyPromptBuilder{}, nil
				},
			})
			workflow, err := assistant.NewRun(t.Context(), textRunInput("question"))
			if existing {
				if err != nil || workflow == nil {
					t.Fatalf("existing source should require no prepender: (%v, %v)", workflow, err)
				}
			} else if err == nil || !strings.Contains(err.Error(), "PrependContextSource") || workflow != nil {
				t.Fatalf("NewRun = (%v, %v), want missing prepend capability error", workflow, err)
			}
			if len(model.Requests()) != 0 {
				t.Fatal("NewRun must not generate a response")
			}
		})
	}
}

func TestAgentModelOverrideToNativeRemovesStaleTextToolDefinitions(t *testing.T) {
	t.Parallel()
	tool := loop.NewEchoTool()
	source, err := tooldefinitions.New(nil, []gaictx.ToolSignature{tool}, nil)
	if err != nil {
		t.Fatal(err)
	}
	builder := gaictx.New(gaictx.Definition{ContextSources: []gaictx.ContextSource{source}})
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{{{Type: ai.TokenTypeText, Data: []byte("answer")}}}}
	assistant := agent.New(agent.Definition{
		Model:  &scriptedWorkflowModel{},
		Tools:  []loop.Tool{tool},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return builder, nil },
	})
	input := textRunInput("question")
	input.Execution = &agent.ExecutionOverrides{Model: nativeToolWorkflowModel{model}}
	workflow, err := assistant.NewRun(t.Context(), input)
	if err != nil {
		t.Fatalf("NewRun: %v", err)
	}
	if builder.HasContextSource("tool_definitions") {
		t.Fatal("native model override retained stale text tool definitions")
	}
	if consumed := consumeWorkflow(t, workflow); len(consumed.errs) != 0 {
		t.Fatalf("workflow errors: %v", consumed.errs)
	}
	requests := model.Requests()
	if len(requests) != 1 || len(requests[0].Tools) != 1 || strings.Contains(requests[0].Prompt, "tool_definitions") || strings.Contains(requests[0].Prompt, "standalone JSON") {
		t.Fatalf("native request retained text protocol or omitted tools: %#v", requests)
	}
}

func (b *readOnlyPromptBuilder) BuildRequest(ctx context.Context, conv gaictx.Conversation) (ai.AIRequest, error) {
	return ai.AIRequest{Messages: []ai.Message{{Role: ai.RoleUser, Parts: ai.CloneParts(b.input.User)}}}, nil
}
