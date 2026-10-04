package history

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/agent/summary"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

// HistoryState is the persisted conversation state consumed by HistorySource.
// Turns contains the unsummarized tail of the conversation; Summary contains
// older turns that have already been compacted.
const HistorySchemaVersion = 1

type HistoryState struct {
	SchemaVersion int `json:"schema_version"`
	Turns         []gaictx.Turn
	Summary       *Summary
}

// MarshalJSON versions newly persisted state while retaining the store API.
func (s HistoryState) MarshalJSON() ([]byte, error) {
	type state HistoryState
	if s.SchemaVersion != 0 && s.SchemaVersion != HistorySchemaVersion {
		return nil, fmt.Errorf("unsupported history schema version: %d", s.SchemaVersion)
	}
	s.SchemaVersion = HistorySchemaVersion
	return json.Marshal(state(s))
}

// UnmarshalJSON requires the current versioned history schema.
func (s *HistoryState) UnmarshalJSON(data []byte) error {
	type state HistoryState
	var decoded state
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	if decoded.SchemaVersion != HistorySchemaVersion {
		return fmt.Errorf("unsupported history schema version: %d", decoded.SchemaVersion)
	}
	*s = HistoryState(decoded)
	return nil
}

// HistoryStore loads and saves history state for a session.
// Calculated token counts are build-local and are never written to the store.
type HistoryStore interface {
	GetLastHistoryState(ctx context.Context, sessionID string) (*HistoryState, error)
	SaveHistoryState(ctx context.Context, sessionID string, state *HistoryState) error
}

// HistorySource renders persisted conversation history as prompt context.
type HistorySource struct {
	historyStateStore HistoryStore
	sessionID         string

	debug   gai.ObservationSink
	counter ai.TokenCounter

	summarizer       *summary.Summarizer
	summarize        bool
	summaryAmount    float32
	summaryMaxTokens int
}

// SummarizerDefinition configures history summarization.
// When Enabled is true, HistorySource first tries to fit history normally. If
// the token budget is reached, it summarizes the oldest Amount of unsummarized
// turns. Amount is a fraction from 0 to 1 and defaults to 0.7 when left unset.
// Provide either Summarizer or Model when Enabled is true.
type SummarizerDefinition struct {
	Model            ai.Model
	Summarizer       *summary.Summarizer
	Enabled          bool
	SummaryMaxTokens int
	Amount           float32
}

// NewHistory creates a HistorySource without summarization.
func NewHistory(sessionId string, historyStateStore HistoryStore) *HistorySource {
	return &HistorySource{
		historyStateStore: historyStateStore,
		sessionID:         sessionId,
	}
}

// New creates a HistorySource for sessionId.
// Pass nil summaryDef to disable summarization. When summarization is enabled,
// New validates the configuration and builds a default summary agent from Model
// if Summarizer is not provided.
func New(sessionId string, historyStateStore HistoryStore, summaryDef *SummarizerDefinition) (*HistorySource, error) {
	if summaryDef == nil {
		return NewHistory(sessionId, historyStateStore), nil
	}
	config := *summaryDef
	if config.Amount < 0 || config.Amount > 1 {
		return nil, ErrInvalidSummaryAmount
	}
	if config.Amount == 0 {
		config.Amount = 0.7
	}
	if config.Enabled && config.Summarizer == nil {
		if config.Model != nil {
			summarizer := summary.New(config.Model)
			config.Summarizer = &summarizer
		} else {
			return nil, ErrSummarizerRequired
		}
	}

	return &HistorySource{
		historyStateStore: historyStateStore,
		sessionID:         sessionId,
		summarizer:        config.Summarizer,
		summarize:         config.Enabled,
		summaryAmount:     config.Amount,
		summaryMaxTokens:  config.SummaryMaxTokens,
	}, nil
}

func (p *HistorySource) Name() string {
	return "history"
}

func (s *HistorySource) SetTokenCounter(counter ai.TokenCounter) {
	s.counter = counter
}

func (s *HistorySource) ObservationSink(debug gai.ObservationSink, conv gaictx.Conversation) {
	s.debug = debug
}

// Function returns a semantic history part selected for tokenBudget.
func (s *HistorySource) Function(ctx context.Context, tokenBudget int) (gaictx.Part, error) {
	part, _, err := s.build(ctx, tokenBudget)
	return part, err
}

// FunctionWithTokens returns the selected part and the token total calculated
// during this invocation with the configured counter. Builder can consume that
// total without recounting the part; the part itself remains stateless.
func (s *HistorySource) FunctionWithTokens(ctx context.Context, tokenBudget int) (gaictx.Part, int, error) {
	return s.build(ctx, tokenBudget)
}

// FunctionWithBudget selects canonical history whose build-local count already
// includes emitted message framing. History bypasses the arbitrary-part renderer,
// so its existing projection is identical to the builder's project callback.
func (s *HistorySource) FunctionWithBudget(ctx context.Context, tokenBudget int, _ func(context.Context, gaictx.Part) (int, error)) (gaictx.Part, int, error) {
	return s.build(ctx, tokenBudget)
}

// build selects and counts one snapshot. Its token total is local to this call.
func (s *HistorySource) build(ctx context.Context, tokenBudget int) (result gaictx.Part, tokens int, err error) {
	ctx, obs := newHistoryBuildObserver(ctx, s.debug, s.sessionID, tokenBudget, s.summarize)
	defer func() {
		obs.Finish(err)
	}()

	if s.historyStateStore == nil {
		obs.StoreMissing(ctx)
		return nil, 0, gaictx.ErrSessionStoreNotFound
	}
	if s.counter == nil {
		obs.TokenCounterMissing(ctx)
		return nil, 0, gaictx.ErrTokenCounterNotFound
	}
	counterID := s.counter.ID()
	obs.SetTokenCounterID(counterID)
	lastHistoryState, err := s.historyStateStore.GetLastHistoryState(ctx, s.sessionID)
	if err != nil {
		obs.StateLoadFailed(ctx, err)
		return nil, 0, err
	}
	state := lastHistoryState
	if state == nil {
		obs.StateMissing(ctx)
	} else {
		obs.MarkStatePresent()
	}
	selected, err := selectHistory(ctx, state, tokenBudget, s.counter)
	obs.Selection(ctx, state, selected, err)
	if err != nil {
		return nil, 0, err
	}
	if selected.budgetReached && s.summarize {
		obs.SummaryAttempted(ctx, len(state.Turns))
		next, changed, summaryErr := s.summarizeState(ctx, state, tokenBudget)
		if summaryErr != nil {
			obs.SummaryFailed(ctx, summaryErr)
			return nil, 0, summaryErr
		}
		if changed {
			obs.MarkSummaryGenerated()
			selected, err = selectHistory(ctx, next, tokenBudget, s.counter)
			obs.Selection(ctx, next, selected, err)
			if err != nil {
				return nil, 0, err
			}
			if err = ctx.Err(); err != nil {
				return nil, 0, err
			}
			if err = s.historyStateStore.SaveHistoryState(ctx, s.sessionID, next); err != nil {
				obs.StateSaveFailed(ctx, err)
				return nil, 0, err
			}
			obs.MarkStateSaved()
		}
	} else if selected.budgetReached {
		obs.SummarySkippedDisabled(ctx)
	}
	obs.BuildFinished(ctx, &selected.part, selected.tokens, selected.turnsVisited, selected.turnsIncluded, selected.messages)
	return &selected.part, selected.tokens, nil
}
