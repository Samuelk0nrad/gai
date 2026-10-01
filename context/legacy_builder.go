package context

import (
	"context"

	"github.com/lace-ai/gai/ai"
)

// LegacyPromptBuilder is the rendered-prompt contract used before canonical
// request construction. Its Conversation and PromptInput values use the current
// canonical types so migrating a custom builder need not duplicate message data.
// Deprecated: implement PromptBuilder.BuildRequest directly.
type LegacyPromptBuilder interface {
	BuildContext(context.Context) ([]Part, error)
	BuildPrompt(context.Context, Conversation) (string, error)
	Input() PromptInput
}

// AdaptLegacyPromptBuilder explicitly lifts an already rendered prompt into one
// canonical user message. It cannot recover roles, tool IDs, or opaque provider
// state lost by the legacy renderer. Use only for text-only custom builders;
// native tool and multimodal builders must implement BuildRequest directly.
func AdaptLegacyPromptBuilder(builder LegacyPromptBuilder) PromptBuilder {
	return legacyBuilderAdapter{builder: builder}
}

type legacyBuilderAdapter struct{ builder LegacyPromptBuilder }

func (a legacyBuilderAdapter) BuildContext(ctx context.Context) ([]Part, error) {
	if a.builder == nil {
		return nil, ErrPromptBuilderNil
	}
	return a.builder.BuildContext(ctx)
}

func (a legacyBuilderAdapter) BuildRequest(ctx context.Context, conv Conversation) (ai.AIRequest, error) {
	if err := ctx.Err(); err != nil {
		return ai.AIRequest{}, err
	}
	if a.builder == nil {
		return ai.AIRequest{}, ErrPromptBuilderNil
	}
	prompt, err := a.builder.BuildPrompt(ctx, conv)
	if err != nil {
		return ai.AIRequest{}, err
	}
	if err := ctx.Err(); err != nil {
		return ai.AIRequest{}, err
	}
	return ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, prompt)}}, nil
}

func (a legacyBuilderAdapter) Input() PromptInput {
	if a.builder == nil {
		return PromptInput{}
	}
	return a.builder.Input().Clone()
}
