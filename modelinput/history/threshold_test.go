package history_test

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput/history"
)

// historyTokens is the size selection counts for the stored turns.
func historyTokens(t *testing.T, state *history.HistoryState) int {
	t.Helper()
	total := 0
	for _, turn := range state.Turns {
		part := history.Part{Messages: []ai.Message{turn.UserMessage.Message}}
		for _, message := range turn.Messages {
			part.Messages = append(part.Messages, message.Message)
		}
		tokens, err := part.Tokens(t.Context(), ai.TextTokenEstimator{})
		if err != nil {
			t.Fatal(err)
		}
		total += tokens
	}
	return total
}

func TestAutoCompactionStartsAtTheThreshold(t *testing.T) {
	size := historyTokens(t, threeTurnState())
	for _, tc := range []struct {
		name      string
		budget    int
		threshold float32
		changed   bool
		early     bool
	}{
		// The default threshold is 0.85 of the budget.
		{"below the default threshold", size * 10 / 8, 0, false, false},
		{"above the default threshold", size * 10 / 9, 0, true, true},
		{"threshold one waits for pressure", size * 10 / 9, 1, false, false},
		{"custom threshold", size * 10 / 6, .5, true, true},
		{"over the budget", size / 2, 0, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store := &sharedHistoryStore{state: threeTurnState()}
			model := &recordingModel{reply: textReply("summary")}
			rendered := 0
			result, err := history.NewHistory("s", store).CompactHistory(t.Context(), history.AutoCompaction{
				HistoryBudget: tc.budget, Threshold: tc.threshold, Model: model, Counter: ai.TextTokenEstimator{},
				Prefix: func(context.Context) (ai.AIRequest, error) {
					rendered++
					return ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleSystem, "system")}}, nil
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			want := map[bool]int{false: 0, true: 1}[tc.changed]
			// The prefix is rendered only when compaction runs.
			if result.Changed != tc.changed || result.Early != tc.early || len(model.calls()) != want || rendered != want {
				t.Fatalf("result=%+v calls=%d rendered=%d", result, len(model.calls()), rendered)
			}
			// Pressure is measured against the whole budget, not the threshold.
			if tc.early && result.PressureRemaining {
				t.Fatal("early compaction reported pressure")
			}
		})
	}
}

func TestAutoCompactionReportsEarlyFailures(t *testing.T) {
	size := historyTokens(t, threeTurnState())
	failure := errors.New("provider down")
	store := &sharedHistoryStore{state: threeTurnState()}
	model := &recordingModel{reply: func(ai.AIRequest) []ai.Token { return []ai.Token{{Err: failure}} }}
	result, err := history.NewHistory("s", store).CompactHistory(t.Context(), history.AutoCompaction{
		HistoryBudget: size * 10 / 9, Model: model, Counter: ai.TextTokenEstimator{},
		Prefix: fixedPrefix(ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleSystem, "system")}}),
	})
	if !errors.Is(err, failure) || !result.Early || result.Changed || store.casCalls != 0 {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

func TestAutoCompactionRejectsAnInvalidThreshold(t *testing.T) {
	for _, threshold := range []float32{-.1, 1.5, float32(math.NaN()), float32(math.Inf(1))} {
		store := &sharedHistoryStore{state: threeTurnState()}
		model := &recordingModel{reply: textReply("summary")}
		_, err := history.NewHistory("s", store).CompactHistory(t.Context(), history.AutoCompaction{
			HistoryBudget: 1, Threshold: threshold, Model: model,
			Prefix: fixedPrefix(ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleSystem, "system")}}),
		})
		if !errors.Is(err, history.ErrInvalidCompactionThreshold) || len(model.calls()) != 0 {
			t.Fatalf("threshold %v: err=%v calls=%d", threshold, err, len(model.calls()))
		}
	}
}

func TestAutoCompactionWaitsForPressureWhenTheSummaryAloneExceedsTheThreshold(t *testing.T) {
	state := threeTurnState()
	state.Summary = history.NewSummary("s", "t0", "t0", 0, 0, ai.ContentPart{Kind: ai.ContentText, Text: strings.Repeat("summary ", 300)})
	summaryTokens, err := (&history.Part{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "Conversation summary:\n"+state.Summary.Content.Text)}}).Tokens(t.Context(), ai.TextTokenEstimator{})
	if err != nil {
		t.Fatal(err)
	}
	total := summaryTokens + historyTokens(t, state)
	// The history fits, and the summary alone is over 0.85 of the budget:
	// folding turns cannot get below the threshold, so it must not compact.
	budget := total + 1
	if float64(summaryTokens) <= .85*float64(budget) {
		t.Fatalf("summary %d is not over the threshold of %d", summaryTokens, budget)
	}
	store := &sharedHistoryStore{state: state}
	model := &recordingModel{reply: textReply("summary")}
	result, err := history.NewHistory("s", store).CompactHistory(t.Context(), history.AutoCompaction{
		HistoryBudget: budget, Model: model, Counter: ai.TextTokenEstimator{},
		Prefix: fixedPrefix(ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleSystem, "system")}}),
	})
	if err != nil || result.Changed || result.Early || len(model.calls()) != 0 {
		t.Fatalf("result=%+v err=%v calls=%d", result, err, len(model.calls()))
	}
}

func TestSelectionKeepsTheNewestTurnsWhenTheSummaryDoesNotFit(t *testing.T) {
	state := threeTurnState()
	state.Summary = history.NewSummary("s", "t0", "t0", 0, 0, ai.ContentPart{Kind: ai.ContentText, Text: strings.Repeat("summary ", 100)})
	newest := history.HistoryState{Turns: state.Turns[2:]}
	budget := historyTokens(t, &newest)
	source := history.NewHistory("s", &sharedHistoryStore{state: state})
	source.SetTokenCounter(ai.TextTokenEstimator{})
	part, tokens, err := source.FunctionWithTokens(t.Context(), budget)
	if err != nil {
		t.Fatal(err)
	}
	messages := part.(*history.Part).Messages
	if tokens != budget || len(messages) != 2 || messages[0].Text() != state.Turns[2].UserMessage.Message.Text() {
		t.Fatalf("tokens=%d/%d messages=%+v", tokens, budget, messages)
	}
}
