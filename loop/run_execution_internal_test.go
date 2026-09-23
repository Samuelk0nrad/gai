package loop

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestAttemptExecutionFinalizesLifecycleOnce(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })

	_, span := provider.Tracer("test").Start(context.Background(), "loop.iteration")
	state := &loopIterationState{obs: &iterationObserver{span: span}}
	var cancelCalls atomic.Int32
	attempt := &attemptExecution{
		state:  state,
		cancel: func() { cancelCalls.Add(1) },
	}

	attempt.cancelAttempt()
	attempt.cancelAttempt()
	attempt.finish(nil)
	attempt.finish(context.Canceled)

	if got := cancelCalls.Load(); got != 1 {
		t.Fatalf("cancel calls = %d, want 1", got)
	}
	if got := len(recorder.Ended()); got != 1 {
		t.Fatalf("ended attempt spans = %d, want 1", got)
	}
}

func TestPostAttemptDiscardSendFailureRecordsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	runState := &loopRunState{}
	run := &runExecution{
		owner:  &Loop{},
		ctx:    ctx,
		events: make(chan Event),
		state:  runState,
	}
	attemptState := &loopIterationState{}
	var cancelCalls atomic.Int32
	attempt := &attemptExecution{
		run:       run,
		ctx:       ctx,
		state:     attemptState,
		iteration: Iteration{Count: 1},
		cancel:    func() { cancelCalls.Add(1) },
	}

	if outcome := run.postAttempt(attempt, true); outcome != iterationTerminal {
		t.Fatalf("outcome = %v, want terminal", outcome)
	}
	if !errors.Is(runState.cancelErr, context.Canceled) {
		t.Fatalf("run cancellation = %v, want context canceled", runState.cancelErr)
	}
	if !attemptState.stats.Canceled {
		t.Fatal("attempt was not marked canceled")
	}
	if got := cancelCalls.Load(); got != 1 {
		t.Fatalf("cancel calls = %d, want 1", got)
	}
}
