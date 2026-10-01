package loop_test

import (
	"context"
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

// fixedPromptBuilder deliberately implements only the loop's three methods.
type fixedPromptBuilder struct {
	input        gaictx.PromptInput
	contextCalls int
}

var _ gaictx.PromptBuilder = (*fixedPromptBuilder)(nil)

func (b *fixedPromptBuilder) BuildContext(context.Context) ([]gaictx.Part, error) {
	b.contextCalls++
	return nil, nil
}

func (b *fixedPromptBuilder) BuildPrompt(context.Context, gaictx.Conversation) (string, error) {
	return (ai.Message{Parts: b.input.User}).Text(), nil
}

func (b *fixedPromptBuilder) Input() gaictx.PromptInput { return b.input.Clone() }

func TestLoopAcceptsMinimalPromptBuilderAndRecordsUserInput(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Type: ai.TokenTypeText, Data: []byte("answer")}},
	}}
	builder := &fixedPromptBuilder{input: gaictx.PromptInput{User: ai.TextParts("question")}}
	l := loop.New(model, nil, builder, nil)
	if err := loopError(collectLoopEvents(t, l, t.Context())); err != nil {
		t.Fatalf("loop failed: %v", err)
	}
	if builder.contextCalls != 1 {
		t.Fatalf("BuildContext calls = %d, want 1", builder.contextCalls)
	}
	requests := model.Requests()
	if len(requests) != 1 || len(requests[0].Messages) != 1 || requests[0].Messages[0].Text() != "question" {
		t.Fatalf("requests = %#v, want rendered question", requests)
	}
	if len(requests[0].Messages) != 1 {
		t.Fatalf("minimal builder unexpectedly supplied native messages: %#v", requests[0].Messages)
	}
	messages := l.Messages()
	if len(messages) != 2 || messages[0].Role != gaictx.RoleUser || messages[0].Text() != "question" || messages[1].Role != gaictx.RoleAssistant || messages[1].Text() != "answer" {
		t.Fatalf("conversation = %#v, want the user question and assistant answer", messages)
	}
}

func (b *fixedPromptBuilder) BuildRequest(ctx context.Context, conv gaictx.Conversation) (ai.AIRequest, error) {
	return ai.AIRequest{Messages: []ai.Message{{Role: ai.RoleUser, Parts: ai.CloneParts(b.input.User)}}}, nil
}
