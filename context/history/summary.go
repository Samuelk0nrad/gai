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

// UnmarshalJSON requires a valid canonical text part for summary content.
func (s *Summary) UnmarshalJSON(data []byte) error {
	type summaryValue Summary
	var decoded summaryValue
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
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
	}
}

// summarizeHistory creates a detached replacement without accessing a store.
func summarizeHistory(ctx context.Context, state *HistoryState, summarizer *summary.Summarizer, counter ai.TokenCounter, amount float32, summaryMaxTokens int, obs *historyObserver) (*HistoryState, bool, error) {
	var err error
	if state == nil {
		err = ErrHistoryStateRequired
		return nil, false, err
	}
	if summarizer == nil {
		err = ErrSummarizerMissing
		return nil, false, err
	}
	obs.ObserveState(len(state.Turns), state.Summary != nil)
	obs.SetTokenCounterID(counter.ID())
	if len(state.Turns) == 0 {
		obs.SummarySkippedNoTurns(ctx)
		return nil, false, nil
	}

	state = state.Clone()
	state.Turns = sortTurnsByCount(state.Turns)

	var builder strings.Builder

	if state.Summary != nil {
		text, renderErr := ai.RenderMessages(ctx, []ai.Message{{Role: ai.RoleUser, Parts: []ai.ContentPart{state.Summary.Content}}})
		if renderErr != nil {
			err = renderErr
			return nil, false, err
		}
		builder.WriteString(text)
		builder.WriteString("\n")
	}

	summarizedTurnCount := summaryTurnCount(len(state.Turns), amount)
	if summarizedTurnCount == 0 {
		obs.SummarySkippedAmountZero(ctx)
		return nil, false, nil
	}
	summarizedTurns := state.Turns[:summarizedTurnCount]
	for i := range summarizedTurns {
		if err = writeTurn(ctx, &builder, &summarizedTurns[i]); err != nil {
			return nil, false, err
		}
	}

	req := summary.Request{
		ID:        "history",
		Text:      builder.String(),
		MaxTokens: summaryMaxTokens,
	}

	res, err := summarizer.Summarize(ctx, req)
	if err != nil {
		return nil, false, err
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
	tokenCount, err := counter.CountTokens(ctx, nextSummary.Content.Text)
	if err != nil {
		obs.SummaryTokenCountFailed(ctx, nextSummary, err)
		return nil, false, err
	}
	obs.SummaryGenerated(ctx, nextSummary, tokenCount, summarizedTurnCount, len(state.Turns)-summarizedTurnCount, state.Summary != nil)

	nextState := &HistoryState{
		SchemaVersion: HistorySchemaVersion,
		Summary:       nextSummary,
		Turns:         append([]gaictx.Turn(nil), state.Turns[summarizedTurnCount:]...),
	}
	return nextState, true, nil
}

func summaryTurnCount(turnCount int, amount float32) int {
	if turnCount == 0 || amount <= 0 {
		return 0
	}
	count := int(float32(turnCount) * amount)
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

// Tokens counts the summary text on demand using the caller's context. Counting
// does not mutate the summary or persist calculated token counts.
func (s *Summary) Tokens(ctx context.Context, counter ai.TokenCounter) (int, error) {
	if s == nil {
		return 0, fmt.Errorf("summary is nil")
	}
	if counter == nil {
		return 0, gaictx.ErrTokenCounterNotFound
	}
	return counter.CountTokens(ctx, s.Content.Text)
}
