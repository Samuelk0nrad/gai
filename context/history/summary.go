package history

import (
	"context"
	"fmt"
	"strings"

	"github.com/lace-ai/gai/agent/summary"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

// Summary is the compact representation of older conversation turns.
type Summary struct {
	ID             string
	StartTurnID    string
	EndTurnID      string
	StartTurnCount int
	EndTurnCount   int
	Content        gaictx.TextContent
	tokenCount     map[string]int
}

func NewSummary(id, startTurnID, endTurnID string, startTurnCount, endTurnCount int, content gaictx.TextContent) *Summary {
	return &Summary{
		ID:             id,
		StartTurnID:    startTurnID,
		EndTurnID:      endTurnID,
		StartTurnCount: startTurnCount,
		EndTurnCount:   endTurnCount,
		Content:        content,
		tokenCount:     map[string]int{},
	}
}

func (s *HistorySource) summarizeState(ctx context.Context, state *HistoryState, maxTokens int) (*HistoryState, error) {
	if s == nil {
		return nil, ErrHistorySourceNil
	}
	ctx, obs := newHistorySummaryObserver(ctx, s.debug, s.sessionID, maxTokens, s.summaryAmount)
	var err error
	defer func() {
		obs.Finish(err)
	}()

	if state == nil {
		err = ErrHistoryStateRequired
		return nil, err
	}
	if s.summarizer == nil {
		err = ErrSummarizerMissing
		return nil, err
	}
	obs.ObserveState(len(state.Turns), state.Summary != nil)
	obs.SetTokenCounterID(s.counter.ID())
	if len(state.Turns) == 0 {
		obs.SummarySkippedNoTurns(ctx)
		return state, nil
	}

	var builder strings.Builder

	if state.Summary != nil {
		builder.WriteString(state.Summary.Content.String())
		builder.WriteString("\n")
	}

	summarizedTurnCount := s.summarizedTurnCount(len(state.Turns))
	if summarizedTurnCount == 0 {
		obs.SummarySkippedAmountZero(ctx)
		return state, nil
	}
	summarizedTurns := state.Turns[:summarizedTurnCount]
	for i := range summarizedTurns {
		writeTurn(&builder, &summarizedTurns[i])
	}

	req := summary.Request{
		ID:        "history",
		Text:      builder.String(),
		MaxTokens: s.summaryMaxTokens,
	}

	res, err := s.summarizer.Summarize(ctx, req)
	if err != nil {
		return nil, err
	}

	firstTurn := summarizedTurns[0]
	lastTurn := summarizedTurns[len(summarizedTurns)-1]
	nextSummary := &Summary{
		StartTurnID:    firstTurn.ID,
		EndTurnID:      lastTurn.ID,
		StartTurnCount: firstTurn.Count,
		EndTurnCount:   lastTurn.Count,
		Content:        gaictx.NewTextContent(res),
	}
	if state.Summary != nil {
		nextSummary.StartTurnID = state.Summary.StartTurnID
		nextSummary.StartTurnCount = state.Summary.StartTurnCount
	}
	tokenCount, err := s.counter.CountTokens(ctx, nextSummary.Content.String())
	if err != nil {
		obs.SummaryTokenCountFailed(ctx, nextSummary, err)
		return nil, err
	}
	nextSummary.SetTokenCount(s.counter.ID(), tokenCount)
	obs.SummaryGenerated(ctx, nextSummary, summarizedTurnCount, len(state.Turns)-summarizedTurnCount, state.Summary != nil)

	nextState := &HistoryState{
		Summary: nextSummary,
		Turns:   append([]gaictx.Turn(nil), state.Turns[summarizedTurnCount:]...),
	}
	return nextState, nil
}

func (s *HistorySource) summarizedTurnCount(turnCount int) int {
	if turnCount == 0 || s.summaryAmount <= 0 {
		return 0
	}
	count := int(float32(turnCount) * s.summaryAmount)
	if count == 0 {
		return 1
	}
	if count > turnCount {
		return turnCount
	}
	return count
}

func writeTurn(builder *strings.Builder, turn *gaictx.Turn) {
	if turn.UserMessage != nil {
		builder.WriteString("user: ")
		if turn.UserMessage.Content != nil {
			builder.WriteString(turn.UserMessage.Content.String())
		}
		builder.WriteString("\n")
	}
	for _, message := range turn.Messages {
		builder.WriteString(string(message.Role))
		builder.WriteString(": ")
		if message.Content != nil {
			builder.WriteString(message.Content.String())
		}
		builder.WriteString("\n")
	}
}

func (s *Summary) TokenCount(counter ai.TokenCounter) (int, error) {
	if s == nil {
		return 0, fmt.Errorf("summary is nil")
	}
	if counter == nil {
		return 0, fmt.Errorf("counter is required")
	}
	if s.tokenCount == nil {
		s.tokenCount = map[string]int{}
	}
	counterID := counter.ID()
	if count, ok := s.tokenCount[counterID]; ok && count >= 0 {
		return count, nil
	} else if ok {
		delete(s.tokenCount, counterID)
	}
	count, err := counter.CountTokens(context.Background(), s.Content.String())
	if err != nil {
		return 0, err
	}
	s.SetTokenCount(counterID, count)
	return count, nil
}

func (s *Summary) SetTokenCount(counterID string, tokens int) {
	if s == nil {
		return
	}
	if s.tokenCount == nil {
		s.tokenCount = map[string]int{}
	}
	if tokens < 0 {
		delete(s.tokenCount, counterID)
		return
	}
	s.tokenCount[counterID] = tokens
}

func (s *Summary) SetTokenCounts(tokenCounts map[string]int) {
	if s == nil {
		return
	}
	s.tokenCount = make(map[string]int, len(tokenCounts))
	for counterID, tokens := range tokenCounts {
		if tokens < 0 {
			continue
		}
		s.tokenCount[counterID] = tokens
	}
}
