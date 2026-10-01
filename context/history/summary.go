package history

import (
	"context"
	"encoding/json"
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
	Content        ai.ContentPart
	tokenCount     map[string]int
}

// MarshalJSON rejects content that the summary reader cannot retain. Storage
// must not successfully save a state that its own decoder will reject.
func (s Summary) MarshalJSON() ([]byte, error) {
	type summaryValue Summary
	if s.Content.Kind != ai.ContentText {
		return nil, fmt.Errorf("summary requires a text content part")
	}
	if err := s.Content.Validate(); err != nil {
		return nil, err
	}
	return json.Marshal(summaryValue(s))
}

// UnmarshalJSON upgrades legacy {"Text": ...} summary payloads to canonical
// text parts while rejecting structured content that a summary cannot retain.
func (s *Summary) UnmarshalJSON(data []byte) error {
	type summaryValue Summary
	var decoded summaryValue
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	if decoded.Content.Kind == "" {
		decoded.Content.Kind = ai.ContentText
	}
	if decoded.Content.Kind != ai.ContentText {
		return fmt.Errorf("summary requires a text content part")
	}
	if err := decoded.Content.Validate(); err != nil {
		return err
	}
	*s = Summary(decoded)
	return nil
}

func NewSummary(id, startTurnID, endTurnID string, startTurnCount, endTurnCount int, content ai.ContentPart) *Summary {
	return &Summary{
		ID:             id,
		StartTurnID:    startTurnID,
		EndTurnID:      endTurnID,
		StartTurnCount: startTurnCount,
		EndTurnCount:   endTurnCount,
		Content:        ai.CloneParts([]ai.ContentPart{content})[0],
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
		text, renderErr := ai.RenderMessages(ctx, []ai.Message{{Role: ai.RoleUser, Parts: []ai.ContentPart{state.Summary.Content}}})
		if renderErr != nil {
			err = renderErr
			return nil, err
		}
		builder.WriteString(text)
		builder.WriteString("\n")
	}

	summarizedTurnCount := s.summarizedTurnCount(len(state.Turns))
	if summarizedTurnCount == 0 {
		obs.SummarySkippedAmountZero(ctx)
		return state, nil
	}
	summarizedTurns := state.Turns[:summarizedTurnCount]
	for i := range summarizedTurns {
		if err = writeTurn(ctx, &builder, &summarizedTurns[i]); err != nil {
			return nil, err
		}
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
		Content:        ai.ContentPart{Kind: ai.ContentText, Text: res},
	}
	if state.Summary != nil {
		nextSummary.StartTurnID = state.Summary.StartTurnID
		nextSummary.StartTurnCount = state.Summary.StartTurnCount
	}
	tokenCount, err := s.counter.CountTokens(ctx, nextSummary.Content.Text)
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

func writeTurn(ctx context.Context, builder *strings.Builder, turn *gaictx.Turn) error {
	var messages []ai.Message
	if turn.UserMessage != nil {
		messages = append(messages, turn.UserMessage.Message)
	}
	for _, message := range turn.Messages {
		messages = append(messages, message.Message)
	}
	text, err := ai.RenderMessages(ctx, messages)
	if err != nil {
		return err
	}
	builder.WriteString(text)
	builder.WriteString("\n")
	return nil
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
	count, err := counter.CountTokens(context.Background(), s.Content.Text)
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
