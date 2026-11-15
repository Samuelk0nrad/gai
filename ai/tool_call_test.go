package ai_test

import (
	"context"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
)

func TestToolCallStringNilReceiver(t *testing.T) {
	var tc *ai.ToolCall

	if got := tc.String(); got != "<nil>" {
		t.Fatalf("expected <nil>, got %q", got)
	}
}

// TestToolCallDetectorCloseReportsFailedFlush checks that Close reports a
// buffered token it could not deliver, so a provider can mark the generation
// as failed.
func TestToolCallDetectorCloseReportsFailedFlush(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	out := make(chan ai.Token)
	detector := ai.NewToolCallDetector(ctx, nil, func(token ai.Token) bool {
		return ai.SendToken(ctx, out, token)
	})
	// An unfinished JSON candidate stays buffered until Close.
	if !detector.Push(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"type":"function"`}}) {
		t.Fatal("Push failed")
	}
	cancel()
	if detector.Close() {
		t.Fatal("Close = true, want false after the flush failed")
	}
	if !detector.Close() {
		t.Fatal("second Close = false, want true")
	}
}

// TestToolCallDetectorCloseReportsDelivery checks that Close reports true
// when it delivers every buffered token.
func TestToolCallDetectorCloseReportsDelivery(t *testing.T) {
	var got []ai.Token
	detector := ai.NewToolCallDetector(context.Background(), nil, func(token ai.Token) bool {
		got = append(got, token)
		return true
	})
	if !detector.Push(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"type":"function"`}}) {
		t.Fatal("Push failed")
	}
	if !detector.Close() {
		t.Fatal("Close = false, want true")
	}
	if len(got) != 1 {
		t.Fatalf("delivered %d tokens, want 1", len(got))
	}
}
