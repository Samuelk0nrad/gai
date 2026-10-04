package history

import (
	"reflect"
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestHistoryStateCloneOwnsNestedPayloads(t *testing.T) {
	state := &HistoryState{
		Summary: &Summary{Content: ai.ContentPart{Kind: ai.ContentText, Text: "summary", Extensions: []ai.Extension{{Data: []byte(`{"summary":1}`)}}}},
		Turns:   []gaictx.Turn{{ID: "turn", UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "user")}, Messages: []gaictx.StoredMessage{{Message: ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{Args: []byte(`{"x":1}`)}}}, Extensions: []ai.Extension{{Data: []byte(`{"opaque":1}`)}}}}}}},
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
		Turns: []gaictx.Turn{
			{ID: "t2", Count: 2, UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "newest")}},
			{ID: "t1", Count: 1, UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "oldest")}},
		},
	}
	before := state.Clone()
	model := &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Message: ai.TextMessage(ai.RoleAssistant, "combined")}}}}
	source, err := New("session", nil, &SummarizerDefinition{Enabled: true, Amount: .5, Model: model})
	if err != nil {
		t.Fatal(err)
	}
	source.SetTokenCounter(ai.TextTokenEstimator{})
	next, changed, err := source.summarizeState(t.Context(), state, 10)
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
