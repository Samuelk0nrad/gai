package history_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/history"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestPlainSavedAndGeneratedSummariesFitTightBudget(t *testing.T) {
	t.Parallel()
	for _, generated := range []bool{false, true} {
		name := "saved"
		if generated {
			name = "generated"
		}
		// The literal prefixed summary costs 6 text tokens + 4 framing tokens.
		for _, budget := range []int{9, 10} {
			t.Run(fmt.Sprintf("%s/budget_%d", name, budget), func(t *testing.T) {
				t.Parallel()
				store := &sharedHistoryStore{state: &history.HistoryState{}}
				var source *history.HistorySource
				var model *mocks.MockModel
				if generated {
					store.state.Turns = []gaictx.Turn{{ID: "old", Count: 1, UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, strings.Repeat("old ", 40))}}}
					model = &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Message: ai.TextMessage(ai.RoleAssistant, "x")}}}}
					compactor, err := history.NewCompactor("session", store, history.CompactorDefinition{Amount: 1, Model: model})
					if err != nil {
						t.Fatal(err)
					}
					if _, err := compactor.Compact(t.Context(), budget); err != nil {
						t.Fatal(err)
					}
					source = history.NewHistory("session", store)
				} else {
					store.state.Summary = history.NewSummary("summary", "old", "old", 1, 1, ai.ContentPart{Kind: ai.ContentText, Text: "x"})
					source = history.NewHistory("session", store)
				}
				counter := ai.TextTokenEstimator{}
				source.SetTokenCounter(counter)
				part, tokens, err := source.FunctionWithTokens(t.Context(), budget)
				if err != nil {
					t.Fatal(err)
				}
				messages := part.(*history.Part).ConversationMessages()
				if budget == 9 {
					if len(messages) != 0 || tokens != 0 {
						t.Fatalf("oversized summary included: messages %d, tokens %d", len(messages), tokens)
					}
				} else {
					if len(messages) != 1 || len(messages[0].Parts) != 1 || messages[0].Text() != "Conversation summary:\nx" || tokens != 10 {
						t.Fatalf("exactly fitting plain summary omitted or overcounted: messages %#v, tokens %d", messages, tokens)
					}
				}
				actual, err := part.Tokens(t.Context(), counter)
				if err != nil || actual != tokens {
					t.Fatalf("handed-off total %d differs from snapshot count %d: %v", tokens, actual, err)
				}
				if generated && model.Count != 1 {
					t.Fatalf("summarizer calls = %d, want 1", model.Count)
				}
				if generated && (len(store.saved) != 1 || store.saved[0].Summary == nil || store.saved[0].Summary.Content.Text != "x") {
					t.Fatal("semantic summary was not saved")
				}
				if !generated && len(store.saved) != 0 {
					t.Fatal("selecting a saved summary wrote history")
				}
			})
		}
	}
}

type reviewRecordingCounter struct{ inputs []string }

func (*reviewRecordingCounter) ID() string { return "test/review-recording-v1" }
func (*reviewRecordingCounter) Fidelity() ai.TokenCountFidelity {
	return ai.TokenCountEstimated
}
func (c *reviewRecordingCounter) CountTokens(ctx context.Context, text string) (int, error) {
	c.inputs = append(c.inputs, text)
	return (ai.TextTokenEstimator{}).CountTokens(ctx, text)
}

func TestOpaqueSummaryRetainsOriginalPartAndCountingProjection(t *testing.T) {
	t.Parallel()
	content := ai.ContentPart{Kind: ai.ContentText, Text: " x ", Extensions: []ai.Extension{{Namespace: "test", Type: "signature", Data: json.RawMessage(`"opaque"`), Required: true}}}
	store := &sharedHistoryStore{state: &history.HistoryState{Summary: history.NewSummary("summary", "old", "old", 1, 1, content)}}
	before, err := json.Marshal(store.state)
	if err != nil {
		t.Fatal(err)
	}
	source := history.NewHistory("session", store)
	counter := &reviewRecordingCounter{}
	source.SetTokenCounter(counter)
	result, tokens, err := source.FunctionWithTokens(t.Context(), 10000)
	if err != nil {
		t.Fatal(err)
	}
	part := result.(*history.Part)
	want := ai.Message{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "Conversation summary:\n"}, content}}
	if len(part.Messages) != 1 || !reflect.DeepEqual(part.Messages[0], want) {
		t.Fatalf("opaque summary part changed: %#v", part.Messages)
	}
	encoded, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	if len(counter.inputs) != 1 || counter.inputs[0] != string(encoded) {
		t.Fatalf("opaque summary counted through a lossy projection: %q", counter.inputs)
	}
	expected, err := (ai.TextTokenEstimator{}).CountTokens(t.Context(), string(encoded))
	if err != nil || tokens != expected {
		t.Fatalf("opaque summary total = %d, want %d: %v", tokens, expected, err)
	}
	part.Messages[0].Parts[1].Text = "changed"
	part.Messages[0].Parts[1].Extensions[0].Data[1] = 'X'
	after, err := json.Marshal(store.state)
	if err != nil || string(after) != string(before) {
		t.Fatal("prompt snapshot aliases persisted opaque summary")
	}
}

func TestHistoryBuilderCountsSelectedMessagesOncePerBuild(t *testing.T) {
	t.Parallel()
	store := &sharedHistoryStore{state: &history.HistoryState{Turns: []gaictx.Turn{{ID: "turn", Count: 1,
		UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "first")},
		Messages:    []gaictx.StoredMessage{{Message: ai.TextMessage(ai.RoleAssistant, "second")}, {Message: ai.TextMessage(ai.RoleAssistant, "third")}},
	}}}}
	source := history.NewHistory("session", store)
	counter := &mocks.MockTokenCounter{}
	next := &budgetRecorder{}
	build := func() error {
		builder := gaictx.New(gaictx.Definition{TokenBudget: 100, TokenCounter: counter, ContextSources: []gaictx.ContextSource{source, next}})
		_, err := builder.BuildContext(t.Context())
		return err
	}
	for _, calls := range []int{3, 6} {
		if err := build(); err != nil {
			t.Fatal(err)
		}
		if counter.CountCalls != calls || next.remaining != 85 {
			t.Fatalf("build counted %d messages, want %d; remaining %d, want 85", counter.CountCalls, calls, next.remaining)
		}
	}
	store.state.Turns[0].UserMessage.Message.Parts[0].Text = "one two three four"
	if err := build(); err != nil {
		t.Fatal(err)
	}
	if counter.CountCalls != 9 || next.remaining != 82 {
		t.Fatalf("changed content reused an old total: calls %d, remaining %d", counter.CountCalls, next.remaining)
	}
	failure := errors.New("later count failed")
	counter.Err = failure
	next.remaining = -1
	if err := build(); !errors.Is(err, failure) {
		t.Fatalf("later counting failure = %v, want %v", err, failure)
	}
	if counter.CountCalls != 10 || next.remaining != -1 {
		t.Fatal("counting failure was hidden by a previous build total")
	}
}

func TestConcurrentHistoryBuildTotalsStayInvocationLocal(t *testing.T) {
	t.Parallel()
	store := &sharedHistoryStore{state: &history.HistoryState{
		Summary: history.NewSummary("summary", "old", "old", 1, 1, ai.ContentPart{Kind: ai.ContentText, Text: "x"}),
		Turns:   []gaictx.Turn{{ID: "recent", Count: 2, UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "recent question")}}},
	}}
	source := history.NewHistory("session", store)
	counter := ai.TextTokenEstimator{}
	source.SetTokenCounter(counter)
	var group sync.WaitGroup
	// Exercise both sides of the summary-only (10) and full-history (18) fits.
	for _, budget := range []int{1, 9, 10, 17, 18, 100} {
		group.Go(func() {
			for range 20 {
				part, tokens, err := source.FunctionWithTokens(t.Context(), budget)
				if err != nil {
					t.Error(err)
					return
				}
				actual, err := part.Tokens(t.Context(), counter)
				if err != nil || tokens != actual || tokens > budget {
					t.Errorf("build total %d, snapshot count %d, budget %d: %v", tokens, actual, budget, err)
					return
				}
			}
		})
	}
	group.Wait()
}
