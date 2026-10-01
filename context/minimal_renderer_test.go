package context_test

import (
	"context"
	"github.com/lace-ai/gai/ai"
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
		PromptInput: gaictx.PromptInput{User: ai.TextParts("question")},
	})
	if _, err := builder.BuildContext(t.Context()); err != nil {
		t.Fatalf("BuildContext: %v", err)
	}
	request, err := builder.BuildRequest(t.Context(), nil)
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	if len(request.Messages) != 1 || request.Messages[0].Text() != "question" {
		t.Fatalf("BuildRequest = %#v, want canonical question", request)
	}
}
