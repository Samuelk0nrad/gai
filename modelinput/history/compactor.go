package history

import (
	"context"
	"errors"
	"math"
	"reflect"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput"
)

// CompactorDefinition configures explicit compaction of completed turns.
// Calling Compact is the explicit opt-in; ordinary context building never invokes it.
type CompactorDefinition struct {
	// Provide a Summarizer or a Model. A Model alone selects
	// NewContinuationSummarizer(Model), which needs WithPrefix on every
	// Compact call. When both are supplied, Summarizer takes precedence.
	Model      ai.Model
	Summarizer Summarizer
	// Amount is the oldest fraction of turns to summarize, in (0,1]. Zero
	// selects the default 0.7. At least one turn is compacted under pressure.
	Amount float32
	// SummaryMaxTokens is the summary output limit; zero inherits the summarizer's default.
	SummaryMaxTokens int
	// ResponseFormat selects how a continuation summary treats the continued
	// request's structured-output setting. The zero value omits it.
	ResponseFormat SummaryResponseFormat
	// TokenCounter estimates the selected history projection, including framing.
	// Nil uses ai.TextTokenEstimator. It does not count the entire model request.
	TokenCounter    ai.TokenCounter
	ObservationSink gai.ObservationSink
}

// CompactionResult describes a successfully completed compaction operation.
// On error the result is zero except Early and the summary call's Usage: the
// candidate must not be treated as committed.
type CompactionResult struct {
	// Changed is true only after a successful CAS commit.
	Changed bool
	// Revision is the loaded revision for a no-op or the committed revision.
	Revision Revision
	// PressureRemaining means some working history still does not fit the
	// supplied history budget. Successful compaction need not eliminate it.
	PressureRemaining bool
	// Early means the history still fit its budget and compaction ran because
	// it crossed the AutoCompaction threshold. It is set on errors too.
	Early bool
	// SummarizedTurns is how many turns the committed summary folded.
	SummarizedTurns int
	// Usage is the provider usage of the summary request, when reported, also
	// when compaction failed after the call. It is not part of any agent
	// run's usage.
	Usage         ai.Usage
	UsageReported bool
}

// CompactOption configures one Compact call.
type CompactOption func(*compactOptions)

type compactOptions struct {
	prefix *ai.AIRequest
	// prefixFunc renders the prefix only once compaction is needed.
	prefixFunc     func(context.Context) (ai.AIRequest, error)
	projectHistory bool
	expected       Revision
	expectedSet    bool
	requestBudget  ai.RequestBudgetConfig
	// threshold, when in (0,1), also compacts history that fits the budget
	// but exceeds this fraction of it.
	threshold float32
}

// WithPrefix supplies the request a continuation summary continues, usually
// AgentResult.Transcript of the run that produced the newest turn. It must
// contain the stored summary and the turns to fold, as that run sent them.
// Pair it with WithExpectedRevision, so the summary applies to that state.
// A nil request supplies none: the transcript is nil when the run did not
// complete, and a continuation summarizer then returns ErrSummaryPrefixRequired.
func WithPrefix(request *ai.AIRequest) CompactOption {
	return func(o *compactOptions) {
		if request == nil {
			o.prefix = nil
			return
		}
		copy := request.Copy()
		o.prefix = &copy
	}
}

// WithExpectedRevision makes Compact fail with *RevisionConflictError, before
// any model call, unless the stored history is still at revision. Use the
// revision the application committed the newest turn with.
func WithExpectedRevision(revision Revision) CompactOption {
	return func(o *compactOptions) { o.expected, o.expectedSet = revision, true }
}

// WithRequestBudget limits a continuation summary request, usually to the
// agent run's request budget. A request that does not fit drops its cached
// prefix after the last folded turn, then folds fewer turns. Summarizers with
// their own prompt, such as agent/summary, do not use it.
func WithRequestBudget(budget ai.RequestBudgetConfig) CompactOption {
	return func(o *compactOptions) { o.requestBudget = budget }
}

// Compactor performs one explicit load/summarize/CAS operation. Dependencies
// must support concurrent calls if a Compactor is shared. Conflicts never cause
// automatic retries; configured summarizer/provider retries remain their policy.
type Compactor struct {
	sessionID        string
	store            HistoryStore
	summarizer       Summarizer
	counter          ai.TokenCounter
	amount           float32
	summaryMaxTokens int
	responseFormat   SummaryResponseFormat
	sink             gai.ObservationSink
	options          options
}

// NewCompactor validates configuration and creates an explicit compactor.
// Give it the WithToolResultPreview value of the session's HistorySource, so
// it selects and counts the history the prompt carries.
func NewCompactor(sessionID string, store HistoryStore, def CompactorDefinition, opts ...Option) (*Compactor, error) {
	if store == nil {
		return nil, modelinput.ErrSessionStoreNotFound
	}
	if math.IsNaN(float64(def.Amount)) || math.IsInf(float64(def.Amount), 0) || def.Amount < 0 || def.Amount > 1 {
		return nil, ErrInvalidSummaryAmount
	}
	if def.SummaryMaxTokens < 0 {
		return nil, ErrInvalidSummaryMaxTokens
	}
	if err := def.ResponseFormat.Validate(); err != nil {
		return nil, err
	}
	if def.Amount == 0 {
		def.Amount = .7
	}
	if def.TokenCounter == nil {
		def.TokenCounter = ai.TextTokenEstimator{}
	}
	summarizer := def.Summarizer
	if summarizer != nil && nilValue(summarizer) {
		return nil, ErrSummarizerRequired
	}
	if summarizer == nil {
		if def.Model == nil || nilValue(def.Model) {
			return nil, ErrSummarizerRequired
		}
		summarizer = NewContinuationSummarizer(def.Model)
	}
	return &Compactor{sessionID: sessionID, store: store, summarizer: summarizer, counter: def.TokenCounter,
		amount: def.Amount, summaryMaxTokens: def.SummaryMaxTokens, responseFormat: def.ResponseFormat, sink: def.ObservationSink,
		options: applyOptions(opts)}, nil
}

// Compact summarizes at most once and attempts at most one CAS, only if the
// working history exceeds historyBudget. The budget is the history allocation,
// not a model's full context window. Zero is a valid allocation; negative is not.
// Summary-only pressure is an unchanged result. Unsupported summaries and
// pre-commit failures leave storage untouched.
//
// A continuation summarizer requires WithPrefix, checked before loading.
func (c *Compactor) Compact(ctx context.Context, historyBudget int, opts ...CompactOption) (result CompactionResult, err error) {
	if c == nil {
		return result, ErrCompactorNil
	}
	var options compactOptions
	for _, opt := range opts {
		if opt != nil {
			opt(&options)
		}
	}
	return c.compact(ctx, historyBudget, options)
}

func (c *Compactor) compact(ctx context.Context, historyBudget int, options compactOptions) (result CompactionResult, err error) {
	if c.store == nil {
		return result, modelinput.ErrSessionStoreNotFound
	}
	ctx, obs := newHistoryCompactionObserver(ctx, c.sink, c.sessionID, historyBudget, c.amount)
	defer func() { obs.CompactionFinished(ctx, result, err); obs.Finish(err) }()
	if historyBudget < 0 {
		return result, ErrInvalidHistoryBudget
	}
	if options.expectedSet && options.expected == "" {
		return result, ErrInvalidExpectedRevision
	}
	if err = options.requestBudget.Validate(); err != nil {
		return result, err
	}
	if t := float64(options.threshold); math.IsNaN(t) || t < 0 || t > 1 {
		return result, ErrInvalidCompactionThreshold
	}
	if _, continuation := c.summarizer.(*ContinuationSummarizer); continuation && options.prefix == nil && options.prefixFunc == nil {
		return result, ErrSummaryPrefixRequired
	}
	if err = ctx.Err(); err != nil {
		return result, err
	}
	obs.SetTokenCounterID(c.counter.ID())
	snapshot, err := c.store.LoadHistory(ctx, c.sessionID)
	if err != nil {
		obs.StateLoadFailed(ctx, err)
		return result, err
	}
	if err = snapshot.validate(); err != nil {
		obs.StateLoadFailed(ctx, err)
		return result, err
	}
	if options.expectedSet && snapshot.Revision != options.expected {
		err = &RevisionConflictError{SessionID: c.sessionID, Expected: options.expected, Actual: snapshot.Revision}
		obs.CompactionConflict(ctx, err.(*RevisionConflictError))
		return result, err
	}
	if snapshot.State != nil {
		obs.ObserveState(len(snapshot.State.Turns), snapshot.State.Summary != nil)
	}
	selected, err := selectHistory(ctx, snapshot.State, historyBudget, c.counter, c.options.previewRunes)
	obs.CompactionSelection(ctx, snapshot.State, selected, err, selectionStageLoaded)
	if err != nil {
		return result, err
	}
	trigger := selected.budgetReached
	if !trigger && options.threshold > 0 && options.threshold < 1 && snapshot.State != nil && len(snapshot.State.Turns) > 0 {
		var early selection
		early, err = selectHistory(ctx, snapshot.State, thresholdBudget(historyBudget, options.threshold), c.counter, c.options.previewRunes)
		if err != nil {
			return result, err
		}
		// A stored summary that alone exceeds the threshold cannot get below
		// it by folding turns, so it would compact on every run. It waits for
		// real pressure instead.
		trigger = early.budgetReached && (snapshot.State.Summary == nil || early.summaryIncluded)
		result.Early = trigger
	}
	if !trigger || snapshot.State == nil || len(snapshot.State.Turns) == 0 {
		return CompactionResult{Revision: snapshot.Revision, PressureRemaining: selected.budgetReached}, nil
	}
	prefix := options.prefix
	if options.prefixFunc != nil {
		var rendered ai.AIRequest
		if rendered, err = options.prefixFunc(ctx); err != nil {
			return result, err
		}
		prefix = &rendered
	}
	if prefix != nil && options.projectHistory {
		// The prefix and the summary come from the same snapshot.
		extended := prefix.Copy()
		extended.Messages = append(extended.Messages, fullProjection(snapshot.State, c.options.previewRunes)...)
		prefix = &extended
	}
	outcome, changed, err := summarizeHistory(ctx, snapshot.State, c.summarizer, summaryRequest{
		amount: c.amount, maxTokens: c.summaryMaxTokens, prefix: prefix, responseFormat: c.responseFormat,
		requestBudget: options.requestBudget, counter: c.counter,
	}, obs)
	// The call is billed even when compaction fails after it.
	result.Usage, result.UsageReported = outcome.usage, outcome.usageReported
	if err != nil {
		return result, err
	}
	if !changed {
		return CompactionResult{Revision: snapshot.Revision, PressureRemaining: selected.budgetReached, Early: result.Early}, nil
	}
	next := outcome.state
	selected, err = selectHistory(ctx, next, historyBudget, c.counter, c.options.previewRunes)
	obs.CompactionSelection(ctx, next, selected, err, selectionStageCandidate)
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
	return CompactionResult{Changed: true, Revision: revision, PressureRemaining: selected.budgetReached, Early: result.Early,
		SummarizedTurns: outcome.summarizedTurns, Usage: result.Usage, UsageReported: result.UsageReported}, nil
}

// thresholdBudget is the largest history size that does not exceed threshold
// of budget. The tolerance absorbs float32 representation error, so 0.7 of 100
// is 70, not 69.
func thresholdBudget(budget int, threshold float32) int {
	limit := float64(budget) * float64(threshold)
	return int(math.Floor(limit + limit*1e-6))
}

func nilValue(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	}
	return false
}
