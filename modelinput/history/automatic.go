package history

import (
	"context"

	"github.com/lace-ai/gai/ai"
)

// CompactHistory is the explicit preparation capability used by
// agent.Definition.AutoCompactHistory. Ordinary source functions never invoke it.
// It uses the effective run model/counter and the default compaction policy.
// At most one summary and one CAS are attempted; errors are not retried.
// ErrHistoryPressureRemaining can follow a successful compaction commit.
// Use NewCompactor directly for custom summarizers, amounts, or pressure policy.
func (s *HistorySource) CompactHistory(ctx context.Context, historyBudget int, model ai.Model, counter ai.TokenCounter) error {
	if s == nil {
		return ErrHistoryStoreRequired
	}
	store, ok := s.historyStateStore.(HistoryStore)
	if !ok {
		return ErrHistoryStoreRequired
	}
	compactor, err := NewCompactor(s.sessionID, store, CompactorDefinition{
		Model: model, TokenCounter: counter, ObservationSink: s.debug,
	})
	if err != nil {
		return err
	}
	result, err := compactor.Compact(ctx, historyBudget)
	if err != nil {
		return err
	}
	if result.PressureRemaining {
		return ErrHistoryPressureRemaining
	}
	return nil
}
