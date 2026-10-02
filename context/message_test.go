package context_test

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestTurnTokensCountsCombinedContentOnEveryCall(t *testing.T) {
	t.Parallel()
	turn := gaictx.Turn{
		UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "hello")},
		Messages: []gaictx.StoredMessage{
			{Message: ai.TextMessage(ai.RoleAssistant, "assistant response")},
			{Message: ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "call", Name: "search", Parts: ai.TextParts("found docs")}}}}},
		},
	}
	encoded, err := json.Marshal(turn.Messages[1].Message)
	if err != nil {
		t.Fatal(err)
	}
	combined := "hello\nassistant response\n" + string(encoded)
	want, err := (ai.TextTokenEstimator{}).CountTokens(t.Context(), combined)
	if err != nil {
		t.Fatal(err)
	}
	before, err := json.Marshal(turn)
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		got, err := turn.Tokens(t.Context(), ai.TextTokenEstimator{})
		if err != nil || got != want {
			t.Fatalf("Tokens = %d, %v; want %d", got, err, want)
		}
	}
	after, err := json.Marshal(turn)
	if err != nil || string(after) != string(before) {
		t.Fatalf("counting changed turn: %s, %v", after, err)
	}
}

func TestMessageAndTurnTokensObserveCurrentContentAndCounter(t *testing.T) {
	t.Parallel()
	message := gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "one")}
	turn := gaictx.Turn{UserMessage: &message}
	for _, count := range []func(context.Context, ai.TokenCounter) (int, error){message.Tokens, turn.Tokens} {
		first, second := &mocks.MockTokenCounter{Count: 3}, &mocks.MockTokenCounter{Count: 8}
		for _, counter := range []*mocks.MockTokenCounter{first, second, first} {
			got, err := count(t.Context(), counter)
			if err != nil || got != counter.Count {
				t.Fatalf("Tokens = %d, %v; want %d", got, err, counter.Count)
			}
		}
		if first.CountCalls != 2 || second.CountCalls != 1 {
			t.Fatal("count was reused by counter ID")
		}
	}
	copy := message
	copy.Message = ai.TextMessage(ai.RoleUser, "one two three")
	counter := &mocks.MockTokenCounter{}
	got, err := copy.Tokens(t.Context(), counter)
	if err != nil || got != 3 {
		t.Fatalf("copied message count = %d, %v", got, err)
	}
	got, err = message.Tokens(t.Context(), counter)
	if err != nil || got != 1 {
		t.Fatalf("original message count = %d, %v", got, err)
	}
	turn.UserMessage = &copy
	got, err = turn.Tokens(t.Context(), counter)
	if err != nil || got != 3 {
		t.Fatalf("changed turn count = %d, %v", got, err)
	}
}

func TestMessageAndTurnTokensPropagateErrors(t *testing.T) {
	t.Parallel()
	message := gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "text")}
	turn := gaictx.Turn{UserMessage: &message}
	for _, count := range []func(context.Context, ai.TokenCounter) (int, error){message.Tokens, turn.Tokens} {
		if _, err := count(t.Context(), nil); !errors.Is(err, gaictx.ErrTokenCounterNotFound) {
			t.Fatalf("nil counter error = %v", err)
		}
		failure := errors.New("count failed")
		if _, err := count(t.Context(), &mocks.MockTokenCounter{Err: failure}); !errors.Is(err, failure) {
			t.Fatalf("counter error = %v", err)
		}
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := count(ctx, ai.TextTokenEstimator{}); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation error = %v", err)
		}
	}
	var missing *gaictx.Turn
	if _, err := missing.Tokens(t.Context(), ai.TextTokenEstimator{}); !errors.Is(err, gaictx.ErrMessageNotFound) {
		t.Fatalf("nil turn error = %v", err)
	}
	invalid := gaictx.StoredMessage{Message: ai.Message{Parts: []ai.ContentPart{{Kind: ai.ContentJSON, JSON: json.RawMessage(`{`)}}}}
	if _, err := invalid.Tokens(t.Context(), ai.TextTokenEstimator{}); err == nil {
		t.Fatal("invalid JSON count succeeded")
	}
	if _, err := (&gaictx.Turn{Messages: []gaictx.StoredMessage{invalid}}).Tokens(t.Context(), ai.TextTokenEstimator{}); err == nil {
		t.Fatal("invalid turn count succeeded")
	}
}

func TestEmptyMessageAndTurnTokens(t *testing.T) {
	t.Parallel()
	message := gaictx.StoredMessage{}
	turn := gaictx.Turn{UserMessage: &message, Messages: []gaictx.StoredMessage{{}}}
	for _, count := range []func(context.Context, ai.TokenCounter) (int, error){message.Tokens, turn.Tokens, (&gaictx.Turn{}).Tokens} {
		got, err := count(t.Context(), &mocks.MockTokenCounter{})
		if err != nil || got != 0 {
			t.Fatalf("empty count = %d, %v", got, err)
		}
	}
}

func TestCopiedMessagesAndTurnsCanBeCountedConcurrently(t *testing.T) {
	t.Parallel()
	message := gaictx.StoredMessage{Message: ai.Message{Role: ai.RoleAssistant, Parts: ai.TextParts("hello"), Extensions: []ai.Extension{{Namespace: "test", Type: "opaque", Data: json.RawMessage(`{"state":1}`)}}}}
	turn := gaictx.Turn{Messages: []gaictx.StoredMessage{message}}
	before, err := json.Marshal(turn)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			copyMessage, copyTurn := message, turn
			for range 20 {
				for _, count := range []func(context.Context, ai.TokenCounter) (int, error){copyMessage.Tokens, copyTurn.Tokens} {
					if _, err := count(t.Context(), ai.TextTokenEstimator{}); err != nil {
						t.Error(err)
						return
					}
				}
			}
		})
	}
	wg.Wait()
	after, err := json.Marshal(turn)
	if err != nil || string(after) != string(before) {
		t.Fatalf("shared turn changed: %s, %v", after, err)
	}
}
