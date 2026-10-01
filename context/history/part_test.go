package history

import (
	"context"
	"github.com/lace-ai/gai/ai"
	"testing"

	"github.com/lace-ai/gai/testutil/mocks"
)

func TestPartTokensRecountsNegativeCachedValue(t *testing.T) {
	t.Parallel()

	counter := &mocks.MockTokenCounter{Count: 6}
	part := &Part{
		Messages: []ai.Message{
			{Parts: ai.TextParts("hello")},
		},
		TokenCount: map[string]int{"mock.counter": -1},
	}

	tokens, err := part.Tokens(context.Background(), counter)
	if err != nil {
		t.Fatalf("Tokens failed: %v", err)
	}
	if tokens != 6 {
		t.Fatalf("expected counter to recount invalid cached value, got %d", tokens)
	}
	if counter.CountCalls != 1 {
		t.Fatalf("expected one counter call, got %d", counter.CountCalls)
	}
	if part.TokenCount["mock.counter"] != 6 {
		t.Fatalf("expected cache to be updated, got %+v", part.TokenCount)
	}
}
