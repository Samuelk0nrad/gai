package ai

import (
	"context"
	"errors"
	"testing"
)

func TestTextTokenEstimator(t *testing.T) {
	counter := TextTokenEstimator{}
	if counter.ID() != "gai.estimate/chars-v1" || counter.Fidelity() != TokenCountEstimated {
		t.Fatalf("unexpected estimator identity or fidelity: %s %v", counter.ID(), counter.Fidelity())
	}
	for _, tc := range []struct {
		text string
		want int
	}{{"", 0}, {"abcd", 1}, {"abcde", 2}, {"🙂界öé", 1}, {"🙂界öéa", 2}, {"\xff", 1}} {
		got, err := counter.CountTokens(t.Context(), tc.text)
		if err != nil || got != tc.want {
			t.Fatalf("CountTokens(%q) = %d, %v; want %d", tc.text, got, err, tc.want)
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := counter.CountTokens(ctx, "hello"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled count: %v", err)
	}
}
