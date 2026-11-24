package history

import (
	"context"
	"reflect"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput"
)

func TestHistoryStateCloneOwnsNestedPayloads(t *testing.T) {
	state := &HistoryState{
		Summary: &Summary{Content: ai.ContentPart{Kind: ai.ContentText, Text: "summary", Extensions: []ai.Extension{{Data: []byte(`{"summary":1}`)}}}},
		Turns:   []modelinput.Turn{{ID: "turn", UserMessage: &modelinput.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "user")}, Messages: []modelinput.StoredMessage{{Message: ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{Args: []byte(`{"x":1}`)}}}, Extensions: []ai.Extension{{Data: []byte(`{"opaque":1}`)}}}}}}},
	}
	before := state.Clone()
	clone := state.Clone()
	clone.Summary.Content.Text = "changed"
	clone.Summary.Content.Extensions[0].Data[0] = '!'
	clone.Turns[0].ID = "changed"
	clone.Turns[0].UserMessage.Message.Parts[0].Text = "changed"
	clone.Turns[0].Messages[0].Message.Parts[0].ToolCall.Args[0] = '!'
	clone.Turns[0].Messages[0].Message.Extensions[0].Data[0] = '!'
	if !reflect.DeepEqual(state, before) {
		t.Fatal("clone aliases original history")
	}
	if (*HistoryState)(nil).Clone() != nil {
		t.Fatal("nil clone is nonnil")
	}
}

func TestSummaryCandidateIsDetachedAndPreservesChronology(t *testing.T) {
	state := &HistoryState{
		Summary: NewSummary("old-summary", "t0", "t0", 0, 0, ai.ContentPart{Kind: ai.ContentText, Text: "previous"}),
		Turns: []modelinput.Turn{
			{ID: "t2", Count: 2, UserMessage: &modelinput.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "newest")}},
			{ID: "t1", Count: 1, UserMessage: &modelinput.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "oldest")}},
		},
	}
	before := state.Clone()
	summarizer := &stubSummarizer{text: "combined"}
	ctx, obs := newHistoryCompactionObserver(t.Context(), nil, "session", 10, .5)
	defer obs.Finish(nil)
	outcome, changed, err := summarizeHistory(ctx, state, summarizer, summaryRequest{amount: .5, counter: ai.TextTokenEstimator{}}, obs)
	next := outcome.state
	if err != nil || !changed {
		t.Fatalf("summarize = %v, %v", changed, err)
	}
	if next.Summary.StartTurnID != "t0" || next.Summary.EndTurnID != "t1" || len(next.Turns) != 1 || next.Turns[0].ID != "t2" {
		t.Fatalf("incorrect summary coverage/tail: %+v", next)
	}
	next.Turns[0].UserMessage.Message.Parts[0].Text = "mutated candidate"
	next.Summary.Content.Text = "mutated summary"
	if !reflect.DeepEqual(state, before) {
		t.Fatal("summary generation or candidate changed input")
	}
}

type stubSummarizer struct {
	text  string
	input SummaryInput
}

func (s *stubSummarizer) SummarizeHistory(_ context.Context, input SummaryInput) (SummaryOutput, error) {
	s.input = input
	return SummaryOutput{Text: s.text}, nil
}

func TestThresholdBudgetAbsorbsFloat32Error(t *testing.T) {
	for _, tc := range []struct {
		budget    int
		threshold float32
		want      int
	}{
		{100, .7, 70}, {100, .9, 90}, {100, .85, 85}, {100, .699, 69}, {0, .85, 0}, {3, .5, 1}, {128000, .85, 108800},
	} {
		if got := thresholdBudget(tc.budget, tc.threshold); got != tc.want {
			t.Errorf("thresholdBudget(%d, %v) = %d, want %d", tc.budget, tc.threshold, got, tc.want)
		}
	}
}
