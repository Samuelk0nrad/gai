package context_test

import (
	"context"
	"testing"

	gaictx "github.com/lace-ai/gai/context"
)

// plainRenderer has no callback registration or tool signature methods.
type plainRenderer struct{}

var _ gaictx.Renderer = plainRenderer{}

func (plainRenderer) Render(ctx context.Context, parts []gaictx.Part) (string, error) {
	var text string
	for _, part := range parts {
		if message, ok := part.(gaictx.MessagePart); ok {
			node, err := message.Render(ctx)
			if err != nil {
				return "", err
			}
			text += node.Value
			for _, content := range node.Children {
				text += content.Value
			}
		}
	}
	return text, nil
}

func TestBuilderAcceptsRenderOnlyRenderer(t *testing.T) {
	t.Parallel()
	builder := gaictx.New(gaictx.Definition{
		Renderer:    plainRenderer{},
		PromptInput: gaictx.PromptInput{User: gaictx.NewTextContent("question")},
	})
	if _, err := builder.BuildContext(t.Context()); err != nil {
		t.Fatalf("BuildContext: %v", err)
	}
	prompt, messages, err := builder.BuildRequest(t.Context(), nil)
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	if prompt != "question" || len(messages) != 1 || messages[0].Text != prompt {
		t.Fatalf("BuildRequest = (%q, %#v), want custom rendered question", prompt, messages)
	}
}
