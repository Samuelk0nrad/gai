package history

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput"
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

// summaryRequest carries the per-operation summary settings.
type summaryRequest struct {
	amount         float32
	maxTokens      int
	prefix         *ai.AIRequest
	responseFormat SummaryResponseFormat
	requestBudget  ai.RequestBudgetConfig
	counter        ai.TokenCounter
}

// summaryOutcome is a summarized candidate and the summary call's accounting.
type summaryOutcome struct {
	state           *HistoryState
	summarizedTurns int
	usage           ai.Usage
	usageReported   bool
}

// summarizeHistory creates a detached replacement without accessing a store.
func summarizeHistory(ctx context.Context, state *HistoryState, summarizer Summarizer, req summaryRequest, obs *historyObserver) (summaryOutcome, bool, error) {
	var err error
	if state == nil {
		err = ErrHistoryStateRequired
		return summaryOutcome{}, false, err
	}
	if summarizer == nil {
		err = ErrSummarizerMissing
		return summaryOutcome{}, false, err
	}
	counter := req.counter
	if counter == nil {
		counter = ai.TextTokenEstimator{}
	}
	obs.ObserveState(len(state.Turns), state.Summary != nil)
	obs.SetTokenCounterID(counter.ID())
	if len(state.Turns) == 0 {
		obs.SummarySkippedNoTurns(ctx)
		return summaryOutcome{}, false, nil
	}

	state = state.Clone()
	state.Turns = sortTurnsByCount(state.Turns)

	summarizedTurnCount := summaryTurnCount(len(state.Turns), req.amount)
	if summarizedTurnCount == 0 {
		obs.SummarySkippedAmountZero(ctx)
		return summaryOutcome{}, false, nil
	}
	res, err := summarizer.SummarizeHistory(ctx, SummaryInput{
		Prefix:          req.prefix,
		PreviousSummary: state.Summary,
		Turns:           state.Turns[:summarizedTurnCount],
		Kept:            state.Turns[summarizedTurnCount:],
		MaxTokens:       req.maxTokens,
		ResponseFormat:  req.responseFormat,
		RequestBudget:   req.requestBudget,
		Counter:         counter,
	})
	if err != nil {
		return summaryOutcome{}, false, err
	}
	if res.FoldedTurns < 0 || res.FoldedTurns > summarizedTurnCount {
		return summaryOutcome{}, false, fmt.Errorf("%w: summarizer folded %d of %d turns", ErrInvalidSummaryReply, res.FoldedTurns, summarizedTurnCount)
	}
	if res.FoldedTurns > 0 {
		summarizedTurnCount = res.FoldedTurns
	}
	if strings.TrimSpace(res.Text) == "" {
		return summaryOutcome{}, false, fmt.Errorf("%w: empty summary", ErrInvalidSummaryReply)
	}
	summarizedTurns := state.Turns[:summarizedTurnCount]

	firstTurn := summarizedTurns[0]
	lastTurn := summarizedTurns[len(summarizedTurns)-1]
	nextSummary := &Summary{
		StartTurnID:    firstTurn.ID,
		EndTurnID:      lastTurn.ID,
		StartTurnCount: firstTurn.Count,
		EndTurnCount:   lastTurn.Count,
		Content:        ai.ContentPart{Kind: ai.ContentText, Text: res.Text},
	}
	if state.Summary != nil {
		nextSummary.StartTurnID = state.Summary.StartTurnID
		nextSummary.StartTurnCount = state.Summary.StartTurnCount
	}
	tokenCount, err := counter.CountTokens(ctx, nextSummary.Content.Text)
	if err != nil {
		obs.SummaryTokenCountFailed(ctx, nextSummary, err)
		return summaryOutcome{}, false, err
	}
	obs.SummaryGenerated(ctx, nextSummary, tokenCount, summarizedTurnCount, len(state.Turns)-summarizedTurnCount, state.Summary != nil)

	nextState := &HistoryState{
		SchemaVersion: HistorySchemaVersion,
		Summary:       nextSummary,
		Turns:         append([]modelinput.Turn(nil), state.Turns[summarizedTurnCount:]...),
	}
	return summaryOutcome{state: nextState, summarizedTurns: summarizedTurnCount, usage: res.Usage, usageReported: res.UsageReported}, true, nil
}

// RenderSummaryText renders an earlier summary and turns as one text document,
// the input of summarizers with their own prompt. Provider state, such as
// reasoning items, signatures, hosted tool calls and citations, has no text
// form and is dropped. Media and unknown required state are rejected.
func RenderSummaryText(ctx context.Context, previous *Summary, turns []modelinput.Turn) (string, error) {
	var builder strings.Builder
	if previous != nil {
		text, err := ai.RenderMessages(ctx, []ai.Message{{Role: ai.RoleUser, Parts: []ai.ContentPart{previous.Content}}})
		if err != nil {
			return "", err
		}
		builder.WriteString(text)
		builder.WriteString("\n")
	}
	for i := range turns {
		if err := writeTurn(ctx, &builder, &turns[i]); err != nil {
			return "", err
		}
	}
	return builder.String(), nil
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

func writeTurn(ctx context.Context, builder *strings.Builder, turn *modelinput.Turn) error {
	var messages []ai.Message
	// The summary prompt needs only the visible content. Provider state, such
	// as reasoning items, signatures, hosted tool calls and citations, has no
	// text form and would make RenderMessages reject the turn. Unknown
	// required state still does.
	add := func(message ai.Message) {
		if text, ok := message.WithoutProviderState(); ok {
			messages = append(messages, text)
		}
	}
	if turn.UserMessage != nil {
		add(turn.UserMessage.Message)
	}
	for _, message := range turn.Messages {
		add(message.Message)
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
		return 0, modelinput.ErrTokenCounterNotFound
	}
	return counter.CountTokens(ctx, s.Content.Text)
}
