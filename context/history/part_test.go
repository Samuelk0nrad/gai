package history

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestPartAndSummaryTokensRecountCurrentContent(t *testing.T) {
	t.Parallel()
	part := Part{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "one")}}
	summary := Summary{Content: ai.ContentPart{Kind: ai.ContentText, Text: "one"}}
	for _, count := range []func(context.Context, ai.TokenCounter) (int, error){part.Tokens, summary.Tokens} {
		counter := &mocks.MockTokenCounter{Count: 6}
		got, err := count(t.Context(), counter)
		if err != nil || got != 6 {
			t.Fatalf("first count = %d, %v", got, err)
		}
		counter.Count = 8
		got, err = count(t.Context(), counter)
		if err != nil || got != 8 || counter.CountCalls != 2 {
			t.Fatalf("second count = %d, %v; calls %d", got, err, counter.CountCalls)
		}
		counter.Err = errors.New("failed")
		if _, err := count(t.Context(), counter); !errors.Is(err, counter.Err) {
			t.Fatalf("later error = %v", err)
		}
	}
	copyPart, copySummary := part, summary
	copyPart.Messages = []ai.Message{ai.TextMessage(ai.RoleUser, "one two three")}
	copySummary.Content = ai.ContentPart{Kind: ai.ContentText, Text: "one two three"}
	for _, tc := range []struct {
		count func(context.Context, ai.TokenCounter) (int, error)
		want  int
	}{
		{part.Tokens, 1}, {summary.Tokens, 1}, {copyPart.Tokens, 3}, {copySummary.Tokens, 3},
	} {
		got, err := tc.count(t.Context(), &mocks.MockTokenCounter{})
		if err != nil || got != tc.want {
			t.Fatalf("copied count = %d, %v; want %d", got, err, tc.want)
		}
	}
}

func TestPartAndSummaryTokensHonorContextAndErrors(t *testing.T) {
	t.Parallel()
	part := Part{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "text")}}
	summary := Summary{Content: ai.ContentPart{Kind: ai.ContentText, Text: "text"}}
	for _, count := range []func(context.Context, ai.TokenCounter) (int, error){part.Tokens, summary.Tokens} {
		if _, err := count(t.Context(), nil); !errors.Is(err, gaictx.ErrTokenCounterNotFound) {
			t.Fatalf("nil counter error = %v", err)
		}
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := count(ctx, ai.TextTokenEstimator{}); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation error = %v", err)
		}
	}
	var nilSummary *Summary
	if _, err := nilSummary.Tokens(t.Context(), ai.TextTokenEstimator{}); err == nil {
		t.Fatal("nil summary count succeeded")
	}
	var nilPart *Part
	if got, err := nilPart.Tokens(t.Context(), ai.TextTokenEstimator{}); err != nil || got != 0 {
		t.Fatalf("nil part = %d, %v", got, err)
	}
}

func TestHistoryPartTokensCountsPreviewWithoutMutatingContent(t *testing.T) {
	t.Parallel()
	full := strings.Repeat("界", 700)
	part := Part{Messages: []ai.Message{{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "call", Name: "search", Parts: ai.TextParts(full)}}}}}}
	before, err := json.Marshal(part)
	if err != nil {
		t.Fatal(err)
	}
	preview := part.ConversationMessages()[0]
	if preview.ToolResults()[0].Text() != strings.Repeat("界", 500)+"\n[tool result truncated]" {
		t.Fatal("wrong preview")
	}
	want, err := (gaictx.StoredMessage{Message: preview}).Tokens(t.Context(), ai.TextTokenEstimator{})
	if err != nil {
		t.Fatal(err)
	}
	summary := Summary{Content: ai.ContentPart{Kind: ai.ContentText, Text: "earlier turns"}}
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			copyPart, copySummary := part, summary
			for range 20 {
				got, err := copyPart.Tokens(t.Context(), ai.TextTokenEstimator{})
				if err != nil || got != want {
					t.Errorf("preview count = %d, %v; want %d", got, err, want)
					return
				}
				if _, err := copySummary.Tokens(t.Context(), ai.TextTokenEstimator{}); err != nil {
					t.Error(err)
					return
				}
			}
		})
	}
	wg.Wait()
	after, err := json.Marshal(part)
	if err != nil || string(before) != string(after) {
		t.Fatal("counting changed source payload")
	}
}
