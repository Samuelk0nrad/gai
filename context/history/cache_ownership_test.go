package history_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/history"
	"github.com/lace-ai/gai/testutil/mocks"
)

// This store owns its immutable loaded state. Saving records only a separate
// build result and is safe for concurrent callers. No token-persistence API is
// needed to implement HistoryStore.
type sharedHistoryStore struct {
	state *history.HistoryState
	mu    sync.Mutex
	saved []*history.HistoryState
}

var _ history.HistoryStore = (*sharedHistoryStore)(nil)

func (s *sharedHistoryStore) GetLastHistoryState(context.Context, string) (*history.HistoryState, error) {
	return s.state, nil
}
func (s *sharedHistoryStore) SaveHistoryState(_ context.Context, _ string, state *history.HistoryState) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.saved = append(s.saved, state)
	return nil
}

func TestConcurrentHistoryBuildsLeaveLoadedStateUntouched(t *testing.T) {
	t.Parallel()
	state := &history.HistoryState{
		Summary: history.NewSummary("summary", "t0", "t0", 0, 0, ai.ContentPart{Kind: ai.ContentText, Text: "earlier turns"}),
		Turns: []gaictx.Turn{
			{ID: "new", Count: 2, UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "new")}},
			{ID: "old", Count: 1, Messages: []gaictx.StoredMessage{{Message: ai.TextMessage(ai.RoleAssistant, "old")}}},
		},
	}
	before, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	store := &sharedHistoryStore{state: state}
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			source := history.NewHistory("session", store)
			source.SetTokenCounter(ai.TextTokenEstimator{})
			result, err := source.Function(t.Context(), 1000)
			if err != nil {
				t.Error(err)
				return
			}
			part := result.(*history.Part)
			if len(part.Messages) != 3 || part.Messages[1].Text() != "old" || part.Messages[2].Text() != "new" {
				t.Errorf("unsorted result: %#v", part.Messages)
				return
			}
			// Prompt values are snapshots of the persisted payload.
			part.Messages[1].Parts[0].Text = "changed prompt"
		})
	}
	wg.Wait()
	after, err := json.Marshal(state)
	if err != nil || string(after) != string(before) || state.SchemaVersion != 0 {
		t.Fatal("build changed the store's loaded value")
	}
	if len(store.saved) != 16 {
		t.Fatalf("saves = %d, want 16", len(store.saved))
	}
	for _, saved := range store.saved {
		if saved == state || saved.SchemaVersion != history.HistorySchemaVersion || saved.Turns[0].ID != "old" || saved.Turns[1].ID != "new" {
			t.Fatal("wrong saved snapshot")
		}
	}
}

func TestLegacyCalculatedCountsAreIgnoredAndNotPersisted(t *testing.T) {
	t.Parallel()
	payload := `{"schema_version":1,"Turns":[{"ID":"t1","Count":1,"TokenCount":{"counter":999999},"UserMessage":{"schema_version":1,"token_count":{"counter":999999},"message":{"role":"user","parts":[{"kind":"text","text":"hello"}]}}}],"Summary":{"Content":{"kind":"text","text":"earlier"},"tokenCount":{"counter":999999}}}`
	var state history.HistoryState
	if err := json.Unmarshal([]byte(payload), &state); err != nil {
		t.Fatal(err)
	}
	store := &sharedHistoryStore{state: &state}
	source := history.NewHistory("session", store)
	counter := &mocks.MockTokenCounter{IDValue: "counter"}
	source.SetTokenCounter(counter)
	result, err := source.Function(t.Context(), 3)
	if err != nil {
		t.Fatal(err)
	}
	part := result.(*history.Part)
	if len(part.Messages) != 2 || part.Messages[1].Text() != "hello" || counter.CountCalls != 2 {
		t.Fatalf("old counts affected selection: %#v; calls %d", part, counter.CountCalls)
	}
	encoded, err := json.Marshal(store.saved[0])
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(strings.ToLower(string(encoded)), "token_count") || strings.Contains(strings.ToLower(string(encoded)), "tokencount") {
		t.Fatalf("calculated counts persisted: %s", encoded)
	}
	var roundTrip history.HistoryState
	if err := json.Unmarshal(encoded, &roundTrip); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(roundTrip.Turns, state.Turns) || !reflect.DeepEqual(roundTrip.Summary, state.Summary) {
		t.Fatal("semantic history changed on write")
	}
}

func TestHistoryBuildPropagatesCountingFailures(t *testing.T) {
	t.Parallel()
	for _, state := range []*history.HistoryState{
		{Summary: history.NewSummary("summary", "", "", 0, 0, ai.ContentPart{Kind: ai.ContentText, Text: "summary"})},
		{Turns: []gaictx.Turn{{Messages: []gaictx.StoredMessage{{Message: ai.TextMessage(ai.RoleUser, "question")}}}}},
	} {
		store := &sharedHistoryStore{state: state}
		source := history.NewHistory("session", store)
		failure := errors.New("local count failed")
		source.SetTokenCounter(&mocks.MockTokenCounter{Err: failure})
		if _, err := source.Function(t.Context(), 100); !errors.Is(err, failure) {
			t.Fatalf("count error = %v", err)
		}
		source.SetTokenCounter(ai.TextTokenEstimator{})
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := source.Function(ctx, 100); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation error = %v", err)
		}
		if len(store.saved) != 0 {
			t.Fatal("saved after failed count")
		}
	}
}

func TestSelectedHistoryPartFitsItsAllocatedBudget(t *testing.T) {
	t.Parallel()
	for _, state := range []*history.HistoryState{
		{Turns: []gaictx.Turn{{UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "a")}, Messages: []gaictx.StoredMessage{{Message: ai.TextMessage(ai.RoleAssistant, "b")}}}}},
		{Summary: history.NewSummary("summary", "", "", 0, 0, ai.ContentPart{Kind: ai.ContentText, Text: "x"})},
		{Turns: []gaictx.Turn{{Messages: []gaictx.StoredMessage{{Message: ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "call", Name: "search", Parts: ai.TextParts(strings.Repeat("x", 8000))}}}}}}}}},
	} {
		store := &sharedHistoryStore{state: state}
		source := history.NewHistory("session", store)
		counter := ai.TextTokenEstimator{}
		source.SetTokenCounter(counter)
		all, err := source.Function(t.Context(), 10000)
		if err != nil {
			t.Fatal(err)
		}
		required, err := all.Tokens(t.Context(), counter)
		if err != nil || required <= 1 {
			t.Fatalf("full part count = %d, %v", required, err)
		}
		for _, budget := range []int{1, required - 1, required} {
			selected, err := source.Function(t.Context(), budget)
			if err != nil {
				t.Fatal(err)
			}
			got, err := selected.Tokens(t.Context(), counter)
			if err != nil || got > budget {
				t.Fatalf("selected part costs %d with allocation %d: %v", got, budget, err)
			}
			if budget == required && got != required {
				t.Fatalf("exactly fitting history excluded: cost %d, budget %d", got, budget)
			}
		}
	}
}

type budgetRecorder struct{ remaining int }

func (*budgetRecorder) Name() string { return "next" }
func (s *budgetRecorder) Function(_ context.Context, budget int) (gaictx.Part, error) {
	s.remaining = budget
	return nil, nil
}

func TestHistoryLeavesAccurateRemainingBudgetForNextSource(t *testing.T) {
	t.Parallel()
	for _, state := range []*history.HistoryState{
		{Turns: []gaictx.Turn{{UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "a")}, Messages: []gaictx.StoredMessage{{Message: ai.TextMessage(ai.RoleAssistant, "b")}}}}},
		{Summary: history.NewSummary("summary", "", "", 0, 0, ai.ContentPart{Kind: ai.ContentText, Text: "x"})},
	} {
		next := &budgetRecorder{}
		source := history.NewHistory("session", &sharedHistoryStore{state: state})
		builder := gaictx.New(gaictx.Definition{TokenBudget: 100, ContextSources: []gaictx.ContextSource{source, next}})
		parts, err := builder.BuildContext(t.Context())
		if err != nil || len(parts) != 1 {
			t.Fatalf("BuildContext = %v, %v", parts, err)
		}
		tokens, err := parts[0].Tokens(t.Context(), builder.TokenCounter())
		if err != nil || next.remaining != 100-tokens {
			t.Fatalf("remaining budget = %d, want %d: %v", next.remaining, 100-tokens, err)
		}
	}
}
