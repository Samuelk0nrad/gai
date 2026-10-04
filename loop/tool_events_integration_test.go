package loop_test

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
)

// eventTool registers an isolated handler for public loop-event regressions.
func eventTool(t *testing.T, name string, serial bool, fn loop.ToolFunc) loop.Tool {
	t.Helper()
	tool, err := loop.NewTool(name, name, ai.ToolParameters{}, fn)
	if err != nil {
		t.Fatal(err)
	}
	tool, err = loop.WithToolOptions(tool, loop.ToolOptions{Serial: serial})
	if err != nil {
		t.Fatal(err)
	}
	return tool
}

// eventModel emits one batch and then a normal final answer.
func eventModel(names ...string) *scriptedStreamModel {
	tokens := make([]ai.Token, len(names))
	for i, name := range names {
		tokens[i] = ai.Token{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: fmt.Sprint(i), Type: "function", Name: name, Args: []byte(`{}`)}}}
	}
	return &scriptedStreamModel{sequences: [][]ai.Token{tokens, {{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "done"}}}}}
}

// assertToolEventOrder checks every invoked tool starts before its published result.
func assertToolEventOrder(t *testing.T, events []loop.Event) []string {
	t.Helper()
	started := map[string]bool{}
	var results []string
	for _, e := range events {
		if e.Type == loop.EventToolStart {
			started[e.ToolCall.ID] = true
		}
		if e.Type == loop.EventToolResult || e.Type == loop.EventToolError {
			if !started[e.ToolCall.ID] {
				t.Fatalf("result without invocation start: %#v", e)
			}
			results = append(results, e.ToolCall.ID)
		}
	}
	return results
}

// TestRunSchedulesBoundedSerialCallsAndPublishesOrderedTranscript covers scheduling through the public stream.
func TestRunSchedulesBoundedSerialCallsAndPublishesOrderedTranscript(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		gates := []chan struct{}{make(chan struct{}), make(chan struct{}), make(chan struct{})}
		fn := func(_ context.Context, call ai.ToolCall) (string, error) {
			<-gates[int(call.ID[0]-'0')]
			return call.ID, nil
		}
		model := eventModel("a", "a", "b")
		l := loop.New(model, []loop.Tool{eventTool(t, "a", true, fn), eventTool(t, "b", false, fn)}, testPromptBuilder(), nil)
		l.ToolExecution.MaxConcurrent = 2
		var events []loop.Event
		done := make(chan struct{})
		go func() { events = collectLoopEvents(t, l, t.Context()); close(done) }()
		synctest.Wait()
		var starts []string
		// Read the stream only after synctest has synchronized its writes.
		// The draining goroutine retains events locally until closed; use handler gates
		// and observed starts below after completing each controlled stage instead.
		close(gates[2])
		synctest.Wait()
		close(gates[0])
		synctest.Wait()
		close(gates[1])
		<-done
		for _, e := range events {
			if e.Type == loop.EventToolStart {
				starts = append(starts, e.ToolCall.ID)
			}
		}
		if !reflect.DeepEqual(starts, []string{"0", "2", "1"}) {
			t.Fatalf("starts=%v", starts)
		}
		if got := assertToolEventOrder(t, events); !reflect.DeepEqual(got, []string{"2", "0", "1"}) {
			t.Fatalf("results=%v", got)
		}
		if events[len(events)-1].Type != loop.EventDone {
			t.Fatalf("terminal=%v", events[len(events)-1])
		}
		requests := model.Requests()
		if len(requests) != 2 {
			t.Fatalf("requests=%d", len(requests))
		}
		var ids []string
		for _, message := range requests[1].Messages {
			for _, part := range message.Parts {
				if part.ToolResult != nil {
					ids = append(ids, part.ToolResult.ToolCallID)
				}
			}
		}
		if !reflect.DeepEqual(ids, []string{"0", "1", "2"}) {
			t.Fatalf("transcript=%v", ids)
		}
	})
}

// TestRunCancellationJoinsActiveToolAndLeavesQueuedCallUnstarted covers cancellation at scheduler capacity.
func TestRunCancellationJoinsActiveToolAndLeavesQueuedCallUnstarted(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		release := make(chan struct{})
		var invoked atomic.Int32
		tool := eventTool(t, "test", false, func(ctx context.Context, _ ai.ToolCall) (string, error) {
			invoked.Add(1)
			<-ctx.Done()
			<-release
			return "", ctx.Err()
		})
		model := eventModel("test", "test")
		l := loop.New(model, []loop.Tool{tool}, testPromptBuilder(), nil)
		l.ToolExecution.MaxConcurrent = 1
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		var events []loop.Event
		done := make(chan struct{})
		go func() { events = collectLoopEvents(t, l, ctx); close(done) }()
		synctest.Wait()
		cancel()
		synctest.Wait()
		select {
		case <-done:
			t.Fatal("stream closed before active tool returned")
		default:
		}
		if invoked.Load() != 1 {
			t.Fatalf("invoked=%d", invoked.Load())
		}
		close(release)
		<-done
		final := events[len(events)-1]
		if final.Type != loop.EventCanceled || !errors.Is(final.Err, context.Canceled) || final.Iteration == nil {
			t.Fatalf("terminal=%#v", final)
		}
		var results int
		for _, part := range final.Iteration.Parts {
			if part.ToolResp != nil {
				results++
				if !errors.Is(part.ToolResp.Err, context.Canceled) {
					t.Fatalf("result=%v", part.ToolResp)
				}
			}
		}
		if results != 1 || len(model.Requests()) != 1 {
			t.Fatalf("results=%d requests=%d", results, len(model.Requests()))
		}
	})
}

// TestRunToolTimeoutContinuesWithSiblingResult keeps local deadlines separate from run cancellation.
func TestRunToolTimeoutContinuesWithSiblingResult(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		slow := eventTool(t, "slow", false, func(ctx context.Context, _ ai.ToolCall) (string, error) { <-ctx.Done(); return "", ctx.Err() })
		fast := eventTool(t, "fast", false, func(context.Context, ai.ToolCall) (string, error) { return "ok", nil })
		l := loop.New(eventModel("slow", "fast"), []loop.Tool{slow, fast}, testPromptBuilder(), nil)
		l.ToolExecution = loop.ToolExecutionConfig{MaxConcurrent: 2, DefaultTimeout: time.Second}
		events := collectLoopEvents(t, l, t.Context())
		assertToolEventOrder(t, events)
		var timedOut, success, done int
		for _, e := range events {
			switch e.Type {
			case loop.EventToolError:
				if errors.Is(e.Err, context.DeadlineExceeded) {
					timedOut++
				}
			case loop.EventToolResult:
				success++
			case loop.EventDone:
				done++
			case loop.EventCanceled, loop.EventError:
				t.Fatalf("unexpected terminal=%v", e)
			}
		}
		if timedOut != 1 || success != 1 || done != 1 {
			t.Fatalf("timeout=%d success=%d done=%d", timedOut, success, done)
		}
	})
}

// TestRunPanicJoinsSiblingAndDoesNotAdmitQueuedCall checks terminal failure and payload withholding.
func TestRunPanicJoinsSiblingAndDoesNotAdmitQueuedCall(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		trigger, release := make(chan struct{}), make(chan struct{})
		var invoked atomic.Int32
		tool := eventTool(t, "test", false, func(ctx context.Context, call ai.ToolCall) (string, error) {
			invoked.Add(1)
			if call.ID == "0" {
				<-trigger
				panic("private-panic-marker")
			}
			<-ctx.Done()
			<-release
			return "", ctx.Err()
		})
		model := eventModel("test", "test", "test")
		l := loop.New(model, []loop.Tool{tool}, testPromptBuilder(), nil)
		l.ToolExecution.MaxConcurrent = 2
		var events []loop.Event
		done := make(chan struct{})
		go func() { events = collectLoopEvents(t, l, t.Context()); close(done) }()
		synctest.Wait()
		close(trigger)
		synctest.Wait()
		select {
		case <-done:
			t.Fatal("panic returned without joining sibling")
		default:
		}
		if invoked.Load() != 2 {
			t.Fatalf("invoked=%d", invoked.Load())
		}
		close(release)
		<-done
		final := events[len(events)-1]
		if final.Type != loop.EventError || !errors.Is(final.Err, loop.ErrToolPanic) || final.Iteration == nil {
			t.Fatalf("terminal=%#v", final)
		}
		for _, e := range events {
			if e.Err != nil && strings.Contains(e.Err.Error(), "private-panic-marker") {
				t.Fatal("panic payload escaped")
			}
		}
		if len(model.Requests()) != 1 {
			t.Fatal("generation continued after panic")
		}
	})
}

// TestRunUnknownToolPublishesErrorAndFinishedObservation preserves non-invocation lifecycle diagnostics.
func TestRunUnknownToolPublishesErrorAndFinishedObservation(t *testing.T) {
	var observed []gai.Observation
	l := loop.New(eventModel("missing"), nil, testPromptBuilder(), nil)
	l.ObservationSink = gai.ObservationSinkFunc(func(_ context.Context, o gai.Observation) {
		if o.Name == "loop_tool_finished" {
			observed = append(observed, o)
		}
	})
	events := collectLoopEvents(t, l, t.Context())
	var failures int
	for _, e := range events {
		if e.Type == loop.EventToolStart {
			t.Fatal("unavailable tool emitted invocation start")
		}
		if e.Type == loop.EventToolError && errors.Is(e.Err, loop.ErrToolNotFound) {
			failures++
		}
	}
	if failures != 1 || len(observed) != 1 || observed[0].Fields["status"] != "error" || observed[0].Fields["duration_ms"] != int64(0) {
		t.Fatalf("failures=%d observations=%v", failures, observed)
	}
	if events[len(events)-1].Type != loop.EventDone {
		t.Fatal("unknown-tool result did not continue to final answer")
	}
}

func TestRunSyntheticResultObservationUsesProcessedResult(t *testing.T) {
	var finished []gai.Observation
	l := loop.New(eventModel("missing"), nil, testPromptBuilder(), nil)
	l.ToolResultProcessor = loop.ToolResultProcessorFunc(func(context.Context, loop.ToolPolicyInput, loop.ToolResult) (loop.ToolResult, error) {
		return loop.ToolResult{Text: "recovered"}, nil
	})
	l.ObservationSink = gai.ObservationSinkFunc(func(_ context.Context, o gai.Observation) {
		if o.Name == "loop_tool_finished" {
			finished = append(finished, o)
		}
	})
	events := collectLoopEvents(t, l, t.Context())
	var recovered bool
	for _, e := range events {
		if e.Type == loop.EventToolResult && e.ToolResult.Text == "recovered" {
			recovered = true
		}
		if e.Type == loop.EventToolError {
			t.Fatalf("unexpected error: %v", e.Err)
		}
	}
	if !recovered || len(finished) != 1 {
		t.Fatalf("recovered=%v observations=%v", recovered, finished)
	}
	observation := finished[0]
	if observation.Err != nil || observation.Fields["tool_outcome"] != "success" || observation.Fields["status"] != "success" || observation.Fields["invoked"] != false {
		t.Fatalf("observation contradicts processed result: %#v", observation)
	}
	if _, exists := observation.Fields["error_code"]; exists {
		t.Fatal("successful result has an error code")
	}
	if got := l.Iterations[0].Conversation; len(got) == 0 || got[len(got)-1].Parts[0].ToolResult.Text() != "recovered" {
		t.Fatalf("processed result absent from transcript: %#v", got)
	}
}

func TestRunCancellationUnblocksFullToolEventBuffer(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var invoked atomic.Int32
		tool := eventTool(t, "test", false, func(context.Context, ai.ToolCall) (string, error) {
			invoked.Add(1)
			return "ok", nil
		})
		names := make([]string, 100)
		for i := range names {
			names[i] = "test"
		}
		model := eventModel(names...)
		l := loop.New(model, []loop.Tool{tool}, testPromptBuilder(), nil)
		l.ToolExecution.MaxConcurrent = 1
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		stream := l.Run(ctx)
		for e := range stream {
			if e.Type == loop.EventToolStart {
				break
			}
		}
		// The first result plus subsequent start/result pairs fill the buffer.
		// With no handlers blocking, Wait leaves the scheduler blocked on delivery.
		synctest.Wait()
		before := invoked.Load()
		if len(stream) != cap(stream) || before == 0 || before == int32(len(names)) {
			t.Fatalf("did not reach blocked delivery: buffer=%d/%d invoked=%d", len(stream), cap(stream), before)
		}
		cancel()
		synctest.Wait()
		for range stream {
		}
		if invoked.Load() != before || len(model.Requests()) != 1 {
			t.Fatal("canceled delivery admitted another call or model request")
		}
	})
}
