package loop

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"

	"github.com/lace-ai/gai/ai"
)

func TestCanonicalCanceledToolStartWaitsForAlreadyRunningTool(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		entered := make(chan struct{})
		release := make(chan struct{})
		var releaseOnce sync.Once
		unblock := func() { releaseOnce.Do(func() { close(release) }) }
		defer unblock()
		var executed atomic.Int32
		tool := observedTestTool{name: "gated", call: func(context.Context, ai.ToolCall) (string, error) {
			executed.Add(1)
			close(entered)
			<-release
			return "completed first", nil
		}}
		l := &Loop{Tools: []Tool{tool}}
		iteration := &Iteration{Parts: make([]IterationPart, 2)}
		calls := []pendingToolCall{
			{partIndex: 0, call: ai.ToolCall{ID: "first", Type: "function", Name: "gated", Args: json.RawMessage(`{}`)}},
			{partIndex: 1, call: ai.ToolCall{ID: "second", Type: "function", Name: "gated", Args: json.RawMessage(`{}`)}},
		}
		// The second start event has no receiver; cancellation stops its send.
		events := make(chan Event)
		done := make(chan error, 1)
		go func() { done <- l.executeToolCalls(ctx, iteration, calls, l.Tools, events, 1, 1, 0) }()
		first := <-events
		if first.Type != EventToolStart || first.ToolCall.ID != "first" {
			t.Fatalf("first event = %#v", first)
		}
		<-entered
		cancel()
		synctest.Wait()
		select {
		case err := <-done:
			t.Fatalf("returned before the running tool stopped: %v", err)
		default:
		}

		unblock()
		err := <-done
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("executeToolCalls error = %v, want cancellation", err)
		}
		if executed.Load() != 1 {
			t.Fatalf("executed %d tools; second tool must not start", executed.Load())
		}
		// Terminal snapshots must only be read after all writers have stopped.
		snapshot := iteration.Clone()
		if snapshot.Parts[0].ToolResp == nil || snapshot.Parts[0].ToolResp.Text != "completed first" || snapshot.Parts[1].ToolResp != nil {
			t.Fatalf("terminal snapshot = %#v", snapshot.Parts)
		}
		snapshot.Parts[0].ToolResp.Text = "mutated snapshot"
		if iteration.Parts[0].ToolResp.Text != "completed first" {
			t.Fatal("terminal tool response snapshot aliases execution state")
		}
	})
}
