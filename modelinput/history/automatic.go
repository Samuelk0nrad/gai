package history

import (
	"context"

	"github.com/Samuelk0nrad/gai/ai"
)

// AutoCompaction is one run-start compaction requested by the agent's
// HistoryCompaction policy.
type AutoCompaction struct {
	// HistoryBudget is the allocation left for this source.
	HistoryBudget int
	// Model is the run's effective model. With a nil Summarizer it selects
	// NewContinuationSummarizer(Model).
	Model   ai.Model
	Counter ai.TokenCounter
	// Summarizer overrides the continuation summarizer.
	Summarizer       Summarizer
	Amount           float32
	SummaryMaxTokens int
	ResponseFormat   SummaryResponseFormat
	// Prefix holds the run's request settings and the messages that precede
	// this source. CompactHistory appends the full stored history to it, so the
	// summary request continues the previous run's cached prompt.
	Prefix ai.AIRequest
	// RequestBudget limits the summary request; zero skips the check.
	RequestBudget ai.RequestBudgetConfig
}

// CompactHistory is the explicit preparation capability used by the agent's
// HistoryCompaction policy. Ordinary source functions never invoke it. At most
// one summary and one CAS are attempted; errors are not retried. Remaining
// pressure is reported in the result, not as an error. The result's Revision
// is the history revision the run then builds from, unless a later writer
// changes it first.
func (s *HistorySource) CompactHistory(ctx context.Context, req AutoCompaction) (CompactionResult, error) {
	if s == nil {
		return CompactionResult{}, ErrHistoryStoreRequired
	}
	store, ok := s.historyStateStore.(HistoryStore)
	if !ok {
		return CompactionResult{}, ErrHistoryStoreRequired
	}
	compactor, err := NewCompactor(s.sessionID, store, CompactorDefinition{
		Model: req.Model, Summarizer: req.Summarizer, Amount: req.Amount,
		SummaryMaxTokens: req.SummaryMaxTokens, ResponseFormat: req.ResponseFormat,
		TokenCounter: req.Counter, ObservationSink: s.debug,
	})
	if err != nil {
		return CompactionResult{}, err
	}
	prefix := req.Prefix.Copy()
	return compactor.compact(ctx, req.HistoryBudget, compactOptions{
		prefix: &prefix, projectHistory: true, requestBudget: req.RequestBudget,
	})
}
