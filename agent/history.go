package agent

import (
	"context"
	"errors"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

// ErrHistoryCompactionNotConfigurable means automatic history compaction was
// enabled with a prompt builder other than the standard *context.Builder.
var ErrHistoryCompactionNotConfigurable = errors.New("automatic history compaction requires *context.Builder")

// The structural capability avoids an import cycle: history's summarizer uses
// agent. Ordinary context sources and ordinary context builds remain read-only.
type historyCompactionSource interface {
	CompactHistory(context.Context, int, ai.Model, ai.TokenCounter) error
}

// Embedding the concrete builder preserves the loop's budget/counter and
// request-building capabilities. The loop applies its resolved allocation before
// this initial build, outside its model iteration and retry loops.
type compactingPromptBuilder struct {
	*gaictx.Builder
	model ai.Model
}

func (b *compactingPromptBuilder) BuildContext(ctx context.Context) ([]gaictx.Part, error) {
	return b.Builder.BuildContextWithPreparation(ctx, func(ctx context.Context, source gaictx.ContextSource, budget int, counter ai.TokenCounter) error {
		if source, ok := source.(historyCompactionSource); ok {
			return source.CompactHistory(ctx, budget, b.model, counter)
		}
		return nil
	})
}
