package history

import (
	"context"
	"errors"
	"math"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/agent/summary"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

// CompactorDefinition configures explicit compaction of completed turns.
// Calling Compact is the opt-in; context building never invokes it.
type CompactorDefinition struct {
	// Provide a Summarizer or a Model used to construct the default summarizer.
	// When both are supplied, Summarizer takes precedence.
	Model      ai.Model
	Summarizer *summary.Summarizer
	// Amount is the oldest fraction of turns to summarize, in (0,1]. Zero
	// selects the default 0.7. At least one turn is compacted under pressure.
	Amount float32
	// SummaryMaxTokens is the summary output limit; zero inherits the summarizer's default.
	SummaryMaxTokens int
	// TokenCounter estimates the selected history projection, including framing.
	// Nil uses ai.TextTokenEstimator. It does not count the entire model request.
	TokenCounter    ai.TokenCounter
	ObservationSink gai.ObservationSink
}

// CompactionResult describes a successfully completed compaction operation.
// On error the result is zero: the candidate must not be treated as committed.
type CompactionResult struct {
	// Changed is true only after a successful CAS commit.
	Changed bool
	// Revision is the loaded revision for a no-op or the committed revision.
	Revision Revision
	// PressureRemaining means some working history still does not fit the
	// supplied history budget. Successful compaction need not eliminate it.
	PressureRemaining bool
}

// Compactor performs one explicit load/summarize/CAS operation. Dependencies
// must support concurrent calls if a Compactor is shared. Conflicts never cause
// automatic retries; configured summarizer/provider retries remain their policy.
type Compactor struct {
	sessionID        string
	store            HistoryStore
	summarizer       summary.Summarizer
	counter          ai.TokenCounter
	amount           float32
	summaryMaxTokens int
	sink             gai.ObservationSink
}

// NewCompactor validates configuration and creates an explicit compactor.
func NewCompactor(sessionID string, store HistoryStore, def CompactorDefinition) (*Compactor, error) {
	if store == nil {
		return nil, gaictx.ErrSessionStoreNotFound
	}
	if math.IsNaN(float64(def.Amount)) || math.IsInf(float64(def.Amount), 0) || def.Amount < 0 || def.Amount > 1 {
		return nil, ErrInvalidSummaryAmount
	}
	if def.SummaryMaxTokens < 0 {
		return nil, ErrInvalidSummaryMaxTokens
	}
	if def.Amount == 0 {
		def.Amount = .7
	}
	if def.TokenCounter == nil {
		def.TokenCounter = ai.TextTokenEstimator{}
	}
	var summarizer summary.Summarizer
	if def.Summarizer != nil {
		summarizer = *def.Summarizer
	} else if def.Model != nil {
		summarizer = summary.New(def.Model, summary.WithTokenCounter(def.TokenCounter))
	} else {
		return nil, ErrSummarizerRequired
	}
	return &Compactor{sessionID: sessionID, store: store, summarizer: summarizer, counter: def.TokenCounter,
		amount: def.Amount, summaryMaxTokens: def.SummaryMaxTokens, sink: def.ObservationSink}, nil
}

// Compact summarizes at most once and attempts at most one CAS, only if the
// working history exceeds historyBudget. The budget is the history allocation,
// not a model's full context window. Zero is a valid allocation; negative is not.
// Summary-only pressure is an unchanged result. Unsupported text summarization
// (media/opaque content) and pre-commit failures leave storage untouched.
func (c *Compactor) Compact(ctx context.Context, historyBudget int) (result CompactionResult, err error) {
	if c == nil {
		return result, ErrCompactorNil
	}
	if c.store == nil {
		return result, gaictx.ErrSessionStoreNotFound
	}
	ctx, obs := newHistoryCompactionObserver(ctx, c.sink, c.sessionID, historyBudget, c.amount)
	defer func() { obs.CompactionFinished(ctx, result, err); obs.Finish(err) }()
	if historyBudget < 0 {
		return result, ErrInvalidHistoryBudget
	}
	if err = ctx.Err(); err != nil {
		return result, err
	}
	snapshot, err := c.store.LoadHistory(ctx, c.sessionID)
	if err != nil {
		obs.StateLoadFailed(ctx, err)
		return result, err
	}
	if err = snapshot.validate(); err != nil {
		return result, err
	}
	obs.SetTokenCounterID(c.counter.ID())
	if snapshot.State != nil {
		obs.ObserveState(len(snapshot.State.Turns), snapshot.State.Summary != nil)
	}
	selected, err := selectHistory(ctx, snapshot.State, historyBudget, c.counter)
	if err != nil {
		return result, err
	}
	if !selected.budgetReached || snapshot.State == nil || len(snapshot.State.Turns) == 0 {
		return CompactionResult{Revision: snapshot.Revision, PressureRemaining: selected.budgetReached}, nil
	}
	next, changed, err := summarizeHistory(ctx, snapshot.State, &c.summarizer, c.counter, c.amount, c.summaryMaxTokens, obs)
	if err != nil {
		return result, err
	}
	if !changed {
		return CompactionResult{Revision: snapshot.Revision, PressureRemaining: selected.budgetReached}, nil
	}
	selected, err = selectHistory(ctx, next, historyBudget, c.counter)
	if err != nil {
		return result, err
	}
	if err = ctx.Err(); err != nil {
		return result, err
	}
	revision, err := c.store.CompareAndSwapHistory(ctx, c.sessionID, snapshot.Revision, next)
	if err != nil {
		var conflict *RevisionConflictError
		if errors.As(err, &conflict) {
			obs.CompactionConflict(ctx, conflict)
		}
		return result, err
	}
	return CompactionResult{Changed: true, Revision: revision, PressureRemaining: selected.budgetReached}, nil
}
