package agent_test

import (
	"context"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

func TestWorkflowCanonicalEmptyTextCannotLeakStaleLegacyData(t *testing.T) {
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{{
		{Type: ai.TokenTypeText, Part: &ai.ContentPart{Kind: ai.ContentText, Text: ""}, Text: "stale visible text", Data: []byte("stale secret data")},
		{Type: ai.TokenTypeText, Text: "accepted"},
	}}}
	assistant := agent.New(agent.Definition{
		Model:  model,
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return &testPromptBuilder{}, nil },
		Limits: agent.Limits{MaxLoopIterations: 1},
	})
	workflow, err := assistant.NewRun(t.Context(), textRunInput("question"))
	if err != nil {
		t.Fatal(err)
	}
	var visible string
	for event := range workflow.RunEvents(t.Context()) {
		if event.Output != nil && event.Output.Kind == agent.OutputText {
			visible += event.Output.Text
		}
	}
	result, err := workflow.Wait()
	if err != nil {
		t.Fatal(err)
	}
	if visible != "accepted" || result.Text != "accepted" || result.Primary.Text != "accepted" || result.AttemptedText != "accepted" || result.Primary.AttemptedText != "accepted" {
		t.Fatalf("legacy content leaked: visible=%q result=%q primary=%q attempted=%q primary-attempted=%q", visible, result.Text, result.Primary.Text, result.AttemptedText, result.Primary.AttemptedText)
	}
	if len(result.Primary.Messages) != 2 || result.Primary.Messages[1].Text() != "accepted" {
		t.Fatalf("canonical stored messages = %#v", result.Primary.Messages)
	}
}
