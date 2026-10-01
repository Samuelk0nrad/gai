package history

import (
	"context"
	"sort"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/agent/summary"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

// HistoryState is the persisted conversation state consumed by HistorySource.
// Turns contains the unsummarized tail of the conversation; Summary contains
// older turns that have already been compacted.
type HistoryState struct {
	Turns   []gaictx.Turn
	Summary *Summary
}

// HistoryStore loads and saves history state for a session.
// Implementations also store cached per-turn token counts through TurnTokenStore.
type HistoryStore interface {
	GetLastHistoryState(ctx context.Context, sessionID string) (*HistoryState, error)
	SaveHistoryState(ctx context.Context, sessionID string, state *HistoryState) error

	gaictx.TurnTokenStore
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

func (s *HistorySource) Function(ctx context.Context, tokenBudget int) (result gaictx.Part, err error) {
	ctx, obs := newHistoryBuildObserver(ctx, s.debug, s.sessionID, tokenBudget, s.summarize)
	defer func() {
		obs.Finish(err)
	}()

	if s.historyStateStore == nil {
		obs.StoreMissing(ctx)
		return nil, gaictx.ErrSessionStoreNotFound
	}
	if s.counter == nil {
		obs.TokenCounterMissing(ctx)
		return nil, gaictx.ErrTokenCounterNotFound
	}
	counterID := s.counter.ID()
	obs.SetTokenCounterID(counterID)
	lastHistoryState, err := s.historyStateStore.GetLastHistoryState(ctx, s.sessionID)
	if err != nil {
		obs.StateLoadFailed(ctx, err)
		return nil, err
	}
	var part Part
	summaryIncluded := false
	budgetReached := false
	tokenCount := 0
	turnCount := 0
	messageCount := 0
	includedTurnCount := 0
	if lastHistoryState == nil {
		obs.StateMissing(ctx)
	} else {
		lastHistoryState.Turns = sortTurnsByCount(lastHistoryState.Turns)
		obs.MarkStatePresent()
		state := lastHistoryState
		summarized := false
		for {
			part = Part{}
			summaryIncluded = false
			budgetReached = false
			tokenCount = 0
			turnCount = 0
			messageCount = 0
			includedTurnCount = 0

			buildBudgetReached, err := s.buildPart(ctx, state, tokenBudget, s.counter, &part, obs, &tokenCount, &turnCount, &messageCount, &includedTurnCount, &summaryIncluded)
			if err != nil {
				return nil, err
			}
			budgetReached = buildBudgetReached

			if budgetReached && s.summarize && !summarized {
				obs.SummaryAttempted(ctx, len(lastHistoryState.Turns))
				state, err = s.summarizeState(ctx, lastHistoryState, tokenBudget)
				if err != nil {
					obs.SummaryFailed(ctx, err)
					return nil, err
				}
				if state != lastHistoryState && state.Summary != nil {
					obs.MarkSummaryGenerated()
				}
				summarized = true
				continue
			}
			if budgetReached && !s.summarize {
				obs.SummarySkippedDisabled(ctx)
			}

			if state.Summary != nil || len(state.Turns) > 0 {
				if err := s.historyStateStore.SaveHistoryState(ctx, s.sessionID, state); err != nil {
					obs.StateSaveFailed(ctx, err)
					return nil, err
				}
				obs.MarkStateSaved()
			}
			break
		}
	}
	part.saveTokens(counterID, tokenCount)
	obs.BuildFinished(ctx, &part, tokenCount, turnCount, includedTurnCount, messageCount)
	result = &part
	return result, nil
}

func (s *HistorySource) buildPart(
	ctx context.Context,
	state *HistoryState,
	tokenBudget int,
	counter ai.TokenCounter,
	part *Part,
	obs *historyObserver,
	tokenCount,
	turnCount,
	messageCount,
	includedTurnCount *int,
	summaryIncluded *bool,
) (bool, error) {
	if state.Summary != nil {
		*summaryIncluded = true
		summaryContent := Content{
			Text:  state.Summary.Content.String(),
			Role:  "summary",
			Value: state.Summary.Content,
		}
		part.Contents = append(part.Contents, summaryContent)
		summaryTokenCount, err := state.Summary.TokenCount(counter)
		if err != nil {
			obs.SummaryTokenCountFailed(ctx, state.Summary, err)
			return false, err
		}
		if *tokenCount+summaryTokenCount > tokenBudget {
			obs.BudgetReached(ctx, *tokenCount, nil)
			return true, nil
		}
		*tokenCount += summaryTokenCount
		obs.SummaryIncluded(ctx, state.Summary)
	} else {
		obs.SummaryMissing(ctx)
	}

	firstIncluded := len(state.Turns)
	budgetReached := false
	for i := len(state.Turns) - 1; i >= 0; i-- {
		turn := &state.Turns[i]
		*turnCount++
		tokens, err := turn.Tokenize(ctx, s.counter, s.historyStateStore)
		if err != nil {
			turnCopy := *turn
			obs.TurnTokenizeFailed(ctx, &turnCopy, err)
			return false, err
		}
		if *tokenCount+tokens > tokenBudget {
			turnCopy := *turn
			obs.BudgetReached(ctx, *tokenCount, &turnCopy)
			budgetReached = true
			break
		}
		*tokenCount += tokens
		firstIncluded = i
		*includedTurnCount++
	}

	for _, turn := range state.Turns[firstIncluded:] {
		if turn.UserMessage != nil {
			part.Contents = append(part.Contents, MapMessageToContent(*turn.UserMessage))
			*messageCount++
		}
		for _, message := range turn.Messages {
			part.Contents = append(part.Contents, MapMessageToContent(message))
			*messageCount++
		}
	}

	return budgetReached, nil
}

// sortTurnsByCount() Sort turns by Count in ascending order (oldest first)
func sortTurnsByCount(turns []gaictx.Turn) []gaictx.Turn {
	sort.SliceStable(turns, func(i, j int) bool {
		return turns[i].Count < turns[j].Count
	})
	return turns
}
