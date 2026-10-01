package context_test

import (
	"context"
	"errors"
	"github.com/lace-ai/gai/ai"
	"testing"

	"github.com/lace-ai/gai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

type turnTokenUpdate struct {
	turnID  string
	counter string
	tokens  int
}

type turnTokenStore struct {
	updates []turnTokenUpdate
	err     error
}

func (s *turnTokenStore) UpdateTurnTokens(ctx context.Context, turnID string, counter string, tokens int) error {
	s.updates = append(s.updates, turnTokenUpdate{
		turnID:  turnID,
		counter: counter,
		tokens:  tokens,
	})
	return s.err
}

func TestTurnTokenizeUsesExistingTurnCount(t *testing.T) {
	t.Parallel()

	counter := &mocks.MockTokenCounter{}
	store := &turnTokenStore{}
	turn := gaictx.Turn{
		ID:         "turn-1",
		TokenCount: map[string]int{"mock.counter": 7},
		Messages: []gaictx.StoredMessage{
			{Message: ai.Message{Parts: ai.TextParts("should not be counted")}},
		},
	}

	tokens, err := turn.Tokenize(context.Background(), counter, store)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}
	if tokens != 7 {
		t.Fatalf("expected cached turn tokens, got %d", tokens)
	}
	if counter.CountCalls != 0 {
		t.Fatalf("expected counter not to be called, got %d calls", counter.CountCalls)
	}
	if len(store.updates) != 0 {
		t.Fatalf("expected cached count not to be saved again, got %+v", store.updates)
	}
}

func TestTurnTokenizeSumsExistingMessageCounts(t *testing.T) {
	t.Parallel()

	counter := &mocks.MockTokenCounter{}
	store := &turnTokenStore{}
	turn := gaictx.Turn{
		ID: "turn-1",
		UserMessage: &gaictx.StoredMessage{
			TokenCount: map[string]int{"mock.counter": 1}, Message: ai.Message{Parts: ai.TextParts("hello")},
		},
		Messages: []gaictx.StoredMessage{
			{

				TokenCount: map[string]int{"mock.counter": 2}, Message: ai.Message{Parts: ai.TextParts("assistant response")},
			},
			{

				TokenCount: map[string]int{"mock.counter": 3}, Message: ai.Message{Parts: []ai.ContentPart{
					{Kind: ai.ContentToolResult,
						ToolResult: &ai.ToolResult{ToolCallID: "call_search", Name: "tool", Parts: ai.TextParts("result text")}}}},
			},
		},
	}

	tokens, err := turn.Tokenize(context.Background(), counter, store)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}
	if tokens != 6 {
		t.Fatalf("expected summed message tokens, got %d", tokens)
	}
	if counter.CountCalls != 0 {
		t.Fatalf("expected counter not to be called, got %d calls", counter.CountCalls)
	}
	if turn.TokenCount["mock.counter"] != 6 {
		t.Fatalf("expected turn token count to be cached, got %+v", turn.TokenCount)
	}
	if len(store.updates) != 1 {
		t.Fatalf("expected one turn token update, got %+v", store.updates)
	}
	if store.updates[0].turnID != "turn-1" || store.updates[0].tokens != 6 {
		t.Fatalf("unexpected turn token update: %+v", store.updates[0])
	}
}

func TestTurnTokenizeCountsCombinedMessagesWithoutUpdatingMessages(t *testing.T) {
	t.Parallel()

	counter := &mocks.MockTokenCounter{}
	store := &turnTokenStore{}
	turn := gaictx.Turn{
		ID:          "turn-1",
		UserMessage: &gaictx.StoredMessage{Message: ai.Message{Parts: ai.TextParts("hello user")}},
		Messages: []gaictx.StoredMessage{
			{Message: ai.Message{Parts: ai.TextParts("assistant response")}},
			{Message: ai.Message{Parts: []ai.ContentPart{
				{Kind: ai.ContentToolResult,
					ToolResult: &ai.ToolResult{ToolCallID: "call_search", Name: "tool", Parts: ai.TextParts("tool result")}}}},
			},
		},
	}

	tokens, err := turn.Tokenize(context.Background(), counter, store)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}
	if tokens == 0 {
		t.Fatal("expected combined messages to be counted")
	}
	if counter.CountCalls != 1 {
		t.Fatalf("expected one combined counter call, got %d calls", counter.CountCalls)
	}
	if turn.UserMessage.TokenCount != nil {
		t.Fatalf("expected user message token count to stay untouched, got %+v", turn.UserMessage.TokenCount)
	}
	for _, message := range turn.Messages {
		if message.TokenCount != nil {
			t.Fatalf("expected message token count to stay untouched, got %+v", message.TokenCount)
		}
	}
	if turn.TokenCount["mock.counter"] != tokens {
		t.Fatalf("expected turn token count to be cached, got %+v", turn.TokenCount)
	}
	if len(store.updates) != 1 {
		t.Fatalf("expected one turn token update, got %+v", store.updates)
	}
}

func TestTurnTokenizeHandlesNilMessageContent(t *testing.T) {
	t.Parallel()

	counter := &mocks.MockTokenCounter{}
	turn := gaictx.Turn{
		ID:          "turn-1",
		UserMessage: &gaictx.StoredMessage{Message: ai.Message{Role: ai.RoleUser}},
		Messages:    []gaictx.StoredMessage{{Message: ai.Message{Role: ai.RoleAssistant}}},
	}

	tokens, err := turn.Tokenize(context.Background(), counter, nil)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}
	if tokens != 0 {
		t.Fatalf("expected nil content to contribute no tokens, got %d", tokens)
	}
	if counter.CountCalls != 1 {
		t.Fatalf("expected one counter call, got %d", counter.CountCalls)
	}
}

func TestTurnTokenizeRequiresTokenCounter(t *testing.T) {
	t.Parallel()

	turn := gaictx.Turn{ID: "turn-1"}
	_, err := turn.Tokenize(context.Background(), nil, nil)
	if !errors.Is(err, gaictx.ErrTokenCounterNotFound) {
		t.Fatalf("expected ErrTokenCounterNotFound, got %v", err)
	}
}

func TestMessageTokensRecountsNegativeCachedValue(t *testing.T) {
	t.Parallel()

	counter := &mocks.MockTokenCounter{Count: 4}
	message := gaictx.StoredMessage{
		TokenCount: map[string]int{"mock.counter": -1}, Message: ai.Message{Parts: ai.TextParts("hello world")},
	}

	tokens, err := message.Tokens(context.Background(), counter)
	if err != nil {
		t.Fatalf("Tokens failed: %v", err)
	}
	if tokens != 4 {
		t.Fatalf("expected counter to recount invalid cached value, got %d", tokens)
	}
	if counter.CountCalls != 1 {
		t.Fatalf("expected one counter call, got %d", counter.CountCalls)
	}
	if message.TokenCount["mock.counter"] != 4 {
		t.Fatalf("expected cache to be updated, got %+v", message.TokenCount)
	}
}

func TestMessageTokensHandlesNilContent(t *testing.T) {
	t.Parallel()

	counter := &mocks.MockTokenCounter{}
	message := gaictx.StoredMessage{Message: ai.Message{}}

	tokens, err := message.Tokens(context.Background(), counter)
	if err != nil {
		t.Fatalf("Tokens failed: %v", err)
	}
	if tokens != 0 {
		t.Fatalf("expected nil content to contribute no tokens, got %d", tokens)
	}
}

func TestTurnTokenizeIgnoresNegativeCachedMessageCounts(t *testing.T) {
	t.Parallel()

	counter := &mocks.MockTokenCounter{}
	store := &turnTokenStore{}
	turn := gaictx.Turn{
		ID: "turn-1",
		UserMessage: &gaictx.StoredMessage{
			TokenCount: map[string]int{"mock.counter": -1}, Message: ai.Message{Parts: ai.TextParts("hello")},
		},
		Messages: []gaictx.StoredMessage{
			{

				TokenCount: map[string]int{"mock.counter": 2}, Message: ai.Message{Parts: ai.TextParts("assistant response")},
			},
		},
	}

	tokens, err := turn.Tokenize(context.Background(), counter, store)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}
	if tokens != 3 {
		t.Fatalf("expected turn tokens to be recounted from messages, got %d", tokens)
	}
	if counter.CountCalls != 1 {
		t.Fatalf("expected counter to be called once, got %d calls", counter.CountCalls)
	}
	if turn.TokenCount["mock.counter"] != 3 {
		t.Fatalf("expected turn token count to be cached, got %+v", turn.TokenCount)
	}
	if len(store.updates) != 1 || store.updates[0].tokens != 3 {
		t.Fatalf("expected turn token update with repaired count, got %+v", store.updates)
	}
}

func TestTurnTokenizeEmitsObservationWhenSavingTokensFails(t *testing.T) {
	t.Parallel()

	saveErr := errors.New("save tokens")
	store := &turnTokenStore{err: saveErr}
	turn := gaictx.Turn{
		ID:    "turn-1",
		Count: 2,
		Messages: []gaictx.StoredMessage{
			{Message: ai.Message{Parts: ai.TextParts("three token message")}},
		},
	}
	var event gai.Observation
	turn.SetObservationSink(gai.ObservationSinkFunc(func(_ context.Context, emitted gai.Observation) {
		event = emitted
	}))

	tokens, err := turn.Tokenize(context.Background(), &mocks.MockTokenCounter{}, store)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}
	if tokens != 3 {
		t.Fatalf("expected calculated token count despite save failure, got %d", tokens)
	}
	if event.Name != "turn_token_save_failed" {
		t.Fatalf("expected token save failure event, got %+v", event)
	}
	if event.Source != "context:Turn.Tokenize" {
		t.Fatalf("unexpected event source: %q", event.Source)
	}
	if event.Err != nil || event.Fields["outcome"] != "error" {
		t.Fatalf("expected safe save-error observation, got %#v", event)
	}
	if event.Fields["turn_id"] != "turn-1" || event.Fields["turn_count"] != 2 ||
		event.Fields["counter_id"] != "mock.counter" || event.Fields["token_count"] != 3 {
		t.Fatalf("unexpected event fields: %+v", event.Fields)
	}
}
