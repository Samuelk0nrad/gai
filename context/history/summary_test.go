package history

import (
	"github.com/lace-ai/gai/ai"
	"strings"
	"testing"

	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestWriteTurnHandlesNilMessageContent(t *testing.T) {
	t.Parallel()

	turn := gaictx.Turn{
		UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "")},
		Messages:    []gaictx.StoredMessage{{Message: ai.TextMessage(ai.RoleAssistant, "")}},
	}
	var builder strings.Builder

	if err := writeTurn(t.Context(), &builder, &turn); err != nil {
		t.Fatal(err)
	}

	if got, want := builder.String(), "<user>\n\n</user>\n<assistant>\n\n</assistant>\n\n"; got != want {
		t.Fatalf("unexpected serialized turn: want %q got %q", want, got)
	}
}

func TestSummaryTokenCountRecountsNegativeCachedValue(t *testing.T) {
	t.Parallel()

	counter := &mocks.MockTokenCounter{Count: 5}
	summary := &Summary{
		Content:    ai.ContentPart{Kind: ai.ContentText, Text: "older turns"},
		tokenCount: map[string]int{"mock.counter": -1},
	}

	tokens, err := summary.TokenCount(counter)
	if err != nil {
		t.Fatalf("TokenCount failed: %v", err)
	}
	if tokens != 5 {
		t.Fatalf("expected counter to recount invalid cached value, got %d", tokens)
	}
	if counter.CountCalls != 1 {
		t.Fatalf("expected one counter call, got %d", counter.CountCalls)
	}
	if summary.tokenCount["mock.counter"] != 5 {
		t.Fatalf("expected cache to be updated, got %+v", summary.tokenCount)
	}
}

func TestSummarySetTokenCountStoresAndClearsCache(t *testing.T) {
	t.Parallel()

	summary := &Summary{}
	summary.SetTokenCount("mock.counter", 7)

	if summary.tokenCount["mock.counter"] != 7 {
		t.Fatalf("expected cache to store token count, got %+v", summary.tokenCount)
	}

	summary.SetTokenCount("mock.counter", -1)
	if _, ok := summary.tokenCount["mock.counter"]; ok {
		t.Fatalf("expected negative token count to clear cache entry, got %+v", summary.tokenCount)
	}
}

func TestSummarySetTokenCountsReplacesCache(t *testing.T) {
	t.Parallel()

	summary := &Summary{
		tokenCount: map[string]int{
			"old.counter": 9,
		},
	}

	summary.SetTokenCounts(map[string]int{
		"mock.counter": 7,
		"bad.counter":  -1,
	})

	if _, ok := summary.tokenCount["old.counter"]; ok {
		t.Fatalf("expected old cache entry to be replaced, got %+v", summary.tokenCount)
	}
	if summary.tokenCount["mock.counter"] != 7 {
		t.Fatalf("expected new cache entry to be stored, got %+v", summary.tokenCount)
	}
	if _, ok := summary.tokenCount["bad.counter"]; ok {
		t.Fatalf("expected negative cache entry to be omitted, got %+v", summary.tokenCount)
	}
}
