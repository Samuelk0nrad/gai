package context_test

import (
	"context"
	"errors"
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

type legacyBuilder struct {
	input  gaictx.PromptInput
	prompt string
	err    error
	calls  int
}

func (b *legacyBuilder) BuildContext(context.Context) ([]gaictx.Part, error) { return nil, b.err }
func (b *legacyBuilder) BuildPrompt(context.Context, gaictx.Conversation) (string, error) {
	b.calls++
	return b.prompt, b.err
}
func (b *legacyBuilder) Input() gaictx.PromptInput { return b.input }

func TestLegacyBuilderAdapterExplicitlyLiftsRenderedText(t *testing.T) {
	t.Parallel()
	old := &legacyBuilder{input: gaictx.PromptInput{User: ai.TextParts("question")}, prompt: "custom rendered question"}
	builder := gaictx.AdaptLegacyPromptBuilder(old)
	if _, err := builder.BuildContext(t.Context()); err != nil {
		t.Fatal(err)
	}
	request, err := builder.BuildRequest(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if old.calls != 1 || request.Prompt != "" || len(request.Messages) != 1 || request.Messages[0].Role != ai.RoleUser || request.Messages[0].Text() != old.prompt {
		t.Fatalf("legacy request = %#v (%d render calls)", request, old.calls)
	}
	input := builder.Input()
	input.User[0].Text = "changed"
	if old.input.User[0].Text != "question" {
		t.Fatal("adapter returned aliased input")
	}
}

func TestLegacyBuilderAdapterPropagatesErrorsAndCancellation(t *testing.T) {
	t.Parallel()
	want := errors.New("legacy render failed")
	old := &legacyBuilder{err: want}
	builder := gaictx.AdaptLegacyPromptBuilder(old)
	if _, err := builder.BuildRequest(t.Context(), nil); !errors.Is(err, want) {
		t.Fatalf("render error = %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := builder.BuildRequest(ctx, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation = %v", err)
	}
	if old.calls != 1 {
		t.Fatal("canceled request invoked legacy renderer")
	}
	if _, err := gaictx.AdaptLegacyPromptBuilder(nil).BuildRequest(t.Context(), nil); !errors.Is(err, gaictx.ErrPromptBuilderNil) {
		t.Fatalf("nil builder error = %v", err)
	}
}
