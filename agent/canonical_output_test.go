package agent_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

func TestWorkflowCanonicalJSONHasConsistentLiveAndAcceptedText(t *testing.T) {
	const raw = `{"ok":true}`
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{{{Part: &ai.ContentPart{Kind: ai.ContentJSON, JSON: json.RawMessage(raw)}}}}}
	assistant := agent.New(agent.Definition{
		Model:  model,
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return &testPromptBuilder{}, nil },
	})
	workflow, err := assistant.NewRun(t.Context(), textRunInput("question"))
	if err != nil {
		t.Fatal(err)
	}
	var visible string
	for event := range workflow.RunEvents(t.Context()) {
		if event.Type == agent.EventOutput && event.Output != nil && event.Output.Kind == agent.OutputText {
			visible += event.Output.Text
		}
	}
	result, err := workflow.Wait()
	if err != nil {
		t.Fatal(err)
	}
	if visible != raw || result.Text != raw || result.AttemptedText != raw || result.Primary.Text != raw || result.Primary.AttemptedText != raw {
		t.Fatalf("JSON views disagree: live=%q workflow=%q attempted=%q primary=%q primary-attempted=%q", visible, result.Text, result.AttemptedText, result.Primary.Text, result.Primary.AttemptedText)
	}
	assertCanonicalJSONResult(t, result.Primary, raw)
}

func assertCanonicalJSONResult(t *testing.T, result agent.AgentResult, raw string) {
	t.Helper()
	if result.Text != raw || result.AttemptedText != raw {
		t.Fatalf("JSON text=%q attempted=%q, want %q", result.Text, result.AttemptedText, raw)
	}
	for _, tokens := range [][]ai.Token{result.Tokens, result.AttemptedTokens} {
		if len(tokens) != 1 || tokens[0].Type() != ai.TokenTypeText || tokens[0].Text() != raw || tokens[0].Part == nil || tokens[0].Part.Kind != ai.ContentJSON || string(tokens[0].Part.JSON) != raw {
			t.Fatalf("JSON token lost canonical kind or text projection: %#v", tokens)
		}
	}
	var assistant []ai.Message
	for _, message := range result.Messages {
		if message.Role == ai.RoleAssistant {
			assistant = append(assistant, message)
		}
	}
	if len(assistant) != 1 || len(assistant[0].Parts) != 1 || assistant[0].Parts[0].Kind != ai.ContentJSON || string(assistant[0].Parts[0].JSON) != raw {
		t.Fatalf("JSON messages = %#v", assistant)
	}
}

func TestAgentMiddlewareCanonicalJSONAppendAndReplace(t *testing.T) {
	const mainJSON = `{"source":"main"}`
	const postJSON = `{"source":"post"}`
	for _, tt := range []struct {
		name   string
		policy agent.OutputPolicy
		want   string
	}{
		{"append", agent.AppendOutput, mainJSON + postJSON},
		{"replace", agent.ReplaceOutput, postJSON},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var postInput agent.RunInput
			post := agent.New(agent.Definition{
				Name:  "post",
				Model: &scriptedWorkflowModel{scripts: [][]ai.Token{{{Part: &ai.ContentPart{Kind: ai.ContentJSON, JSON: json.RawMessage(postJSON)}}}}},
				Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
					postInput = input
					return &testPromptBuilder{}, nil
				},
			})
			main := agent.New(agent.Definition{
				Name:       "main",
				Model:      &scriptedWorkflowModel{scripts: [][]ai.Token{{{Part: &ai.ContentPart{Kind: ai.ContentJSON, JSON: json.RawMessage(mainJSON)}}}}},
				Prompt:     func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return &testPromptBuilder{}, nil },
				Middleware: []agent.Middleware{agent.NewAgentMiddleware(post, agent.AgentMiddlewareConfig{Name: "post", Output: tt.policy})},
			})
			workflow, err := main.NewRun(t.Context(), textRunInput("question"))
			if err != nil {
				t.Fatal(err)
			}
			var visible string
			for event := range workflow.RunEvents(t.Context()) {
				if event.Type == agent.EventOutput && event.Output != nil && event.Output.Kind == agent.OutputText {
					visible += event.Output.Text
				}
			}
			result, err := workflow.Wait()
			if err != nil {
				t.Fatal(err)
			}
			if visible != tt.want || result.Text != tt.want {
				t.Fatalf("JSON middleware live=%q accepted=%q, want %q", visible, result.Text, tt.want)
			}
			assertCanonicalJSONResult(t, result.Primary, mainJSON)
			if len(result.Stages) != 1 || result.Stages[0].Name != "post" {
				t.Fatalf("stages = %#v", result.Stages)
			}
			assertCanonicalJSONResult(t, result.Stages[0].Result, postJSON)
			if got := promptContextValue(postInput, "upstream_output"); got != mainJSON {
				t.Fatalf("middleware upstream JSON = %q, want %q", got, mainJSON)
			}
		})
	}
}
