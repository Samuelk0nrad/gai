package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/lace-ai/gai/ai"
)

func schedulerTool(t *testing.T, name string, options ToolOptions, fn ToolFunc) Tool {
	t.Helper()
	tool, err := NewTool(name, name, ai.ToolParameters{}, fn)
	if err != nil {
		t.Fatal(err)
	}
	tool, err = WithToolOptions(tool, options)
	if err != nil {
		t.Fatal(err)
	}
	return tool
}
func schedulerCalls(names ...string) (*Iteration, []pendingToolCall) {
	iteration := &Iteration{Parts: make([]IterationPart, len(names))}
	calls := make([]pendingToolCall, len(names))
	for i, name := range names {
		calls[i] = pendingToolCall{partIndex: i, call: ai.ToolCall{ID: fmt.Sprint(i), Type: "function", Name: name, Args: json.RawMessage(`{}`)}}
	}
	return iteration, calls
}

func TestToolSchedulerBoundsAndAvoidsSerialHeadOfLineBlocking(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		started := make(chan string, 4)
		release := make(chan struct{})
		var active, maximum atomic.Int32
		fn := func(ctx context.Context, call ai.ToolCall) (string, error) {
			n := active.Add(1)
			defer active.Add(-1)
			for old := maximum.Load(); n > old; old = maximum.Load() {
				if maximum.CompareAndSwap(old, n) {
					break
				}
			}
			started <- call.Name + call.ID
			select {
			case <-release:
				return call.ID, nil
			case <-ctx.Done():
				return "", ctx.Err()
			}
		}
		a := schedulerTool(t, "a", ToolOptions{Serial: true}, fn)
		b := schedulerTool(t, "b", ToolOptions{}, fn)
		l := &Loop{Tools: []Tool{a, b}, ToolExecution: ToolExecutionConfig{MaxConcurrent: 2}}
		iteration, calls := schedulerCalls("a", "a", "b", "b")
		done := make(chan error, 1)
		go func() { done <- l.executeToolCalls(t.Context(), iteration, calls, l.Tools, nil, 1, 1, 0) }()
		synctest.Wait()
		first, second := <-started, <-started
		if (first != "a0" || second != "b2") && (first != "b2" || second != "a0") {
			t.Fatalf("initial admissions=%s,%s", first, second)
		}
		if active.Load() != 2 {
			t.Fatalf("active=%d", active.Load())
		}
		close(release)
		if err := <-done; err != nil {
			t.Fatal(err)
		}
		if maximum.Load() != 2 {
			t.Fatalf("max=%d", maximum.Load())
		}
		for i, message := range iteration.Conversation {
			if message.Parts[0].ToolResult.ToolCallID != fmt.Sprint(i) {
				t.Fatal("conversation order changed")
			}
		}
	})
}

func TestToolSchedulerSharedGuardAcrossRuns(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		guard := &ToolGuard{}
		started := make(chan struct{}, 2)
		release := make(chan struct{})
		var active, maximum atomic.Int32
		tool := schedulerTool(t, "shared", ToolOptions{Guard: guard}, func(context.Context, ai.ToolCall) (string, error) {
			n := active.Add(1)
			defer active.Add(-1)
			if n > maximum.Load() {
				maximum.Store(n)
			}
			started <- struct{}{}
			<-release
			return "ok", nil
		})
		done := make(chan error, 2)
		for range 2 {
			go func() {
				l := &Loop{Tools: []Tool{tool}}
				i, c := schedulerCalls("shared")
				done <- l.executeToolCalls(t.Context(), i, c, l.Tools, nil, 1, 1, 0)
			}()
		}
		synctest.Wait()
		if len(started) != 1 {
			t.Fatalf("shared guard admitted %d", len(started))
		}
		release <- struct{}{}
		synctest.Wait()
		if len(started) != 2 {
			t.Fatal("guard waiter failed to wake")
		}
		release <- struct{}{}
		for range 2 {
			if err := <-done; err != nil {
				t.Fatal(err)
			}
		}
		if maximum.Load() != 1 {
			t.Fatalf("cross-run overlap=%d", maximum.Load())
		}
	})
}

func TestToolSchedulerCancelWaitingForSharedGuard(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		guard := &ToolGuard{}
		release, _ := guard.tryAcquire()
		defer release()
		var invoked atomic.Bool
		tool := schedulerTool(t, "blocked", ToolOptions{Guard: guard}, func(context.Context, ai.ToolCall) (string, error) { invoked.Store(true); return "", nil })
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		l := &Loop{Tools: []Tool{tool}}
		iteration, calls := schedulerCalls("blocked")
		done := make(chan error, 1)
		go func() { done <- l.executeToolCalls(ctx, iteration, calls, l.Tools, nil, 1, 1, 0) }()
		synctest.Wait()
		cancel()
		if err := <-done; !errors.Is(err, context.Canceled) {
			t.Fatalf("error=%v", err)
		}
		if invoked.Load() {
			t.Fatal("queued tool executed")
		}
	})
}

func TestToolDeadlineFiltersWithParentAndPreservesSiblingSuccess(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		slow := schedulerTool(t, "slow", ToolOptions{}, func(ctx context.Context, _ ai.ToolCall) (string, error) {
			<-ctx.Done()
			return "", ctx.Err()
		})
		fast := schedulerTool(t, "fast", ToolOptions{}, func(context.Context, ai.ToolCall) (string, error) { return "ok", nil })
		l := &Loop{Tools: []Tool{slow, fast}, ToolExecution: ToolExecutionConfig{MaxConcurrent: 2, DefaultTimeout: time.Second}, ToolResultProcessor: ToolResultProcessorFunc(func(ctx context.Context, _ ToolPolicyInput, r ToolResult) (ToolResult, error) {
			if ctx.Err() != nil {
				t.Error("processor received expired handler context")
			}
			return r, nil
		})}
		iteration, calls := schedulerCalls("slow", "fast")
		if err := l.executeToolCalls(t.Context(), iteration, calls, l.Tools, nil, 1, 1, 0); err != nil {
			t.Fatal(err)
		}
		if !errors.Is(iteration.Parts[0].ToolResp.Err, context.DeadlineExceeded) || iteration.Parts[1].ToolResp.Text != "ok" {
			t.Fatalf("results=%#v", iteration.Parts)
		}
	})
}

func TestToolSchedulerJoinsAfterPanic(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		entered := make(chan struct{})
		exited := make(chan struct{})
		blocker := schedulerTool(t, "blocker", ToolOptions{}, func(ctx context.Context, _ ai.ToolCall) (string, error) {
			close(entered)
			<-ctx.Done()
			close(exited)
			return "", ctx.Err()
		})
		panicker := schedulerTool(t, "panic", ToolOptions{}, func(context.Context, ai.ToolCall) (string, error) { <-entered; panic("secret panic payload") })
		l := &Loop{Tools: []Tool{blocker, panicker}}
		iteration, calls := schedulerCalls("blocker", "panic")
		if err := l.executeToolCalls(t.Context(), iteration, calls, l.Tools, nil, 1, 1, 0); !errors.Is(err, ErrToolPanic) {
			t.Fatalf("error=%v", err)
		}
		select {
		case <-exited:
		default:
			t.Fatal("sibling was not joined")
		}
	})
}

func TestToolOptionsAreCopiedAndExecutionOnly(t *testing.T) {
	timeout := time.Second
	options := ToolOptions{Serial: true, Timeout: &timeout, Traits: ToolTraits{Effect: ToolEffectDestructive, Idempotent: true}}
	base := NewEchoTool()
	tool, err := WithToolOptions(base, options)
	if err != nil {
		t.Fatal(err)
	}
	timeout = 2 * time.Second
	copied := tool.(ToolOptionsProvider).ToolOptions()
	*copied.Timeout = 3 * time.Second
	again := tool.(ToolOptionsProvider).ToolOptions()
	if *again.Timeout != time.Second {
		t.Fatal("aliased options")
	}
	plain, _ := ToolDefinitions([]Tool{base})
	configured, _ := ToolDefinitions([]Tool{tool})
	p, _ := json.Marshal(plain)
	c, _ := json.Marshal(configured)
	if string(p) != string(c) {
		t.Fatal("execution options entered model definition")
	}
	wrapped, err := WithToolOptions(tool, ToolOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if wrapped.(ToolOptionsProvider).ToolOptions().Serial {
		t.Fatal("nested options unexpectedly merged")
	}
}

func TestUninvokedProcessorPanicIsTerminal(t *testing.T) {
	l := &Loop{ToolResultProcessor: ToolResultProcessorFunc(func(context.Context, ToolPolicyInput, ToolResult) (ToolResult, error) { panic("private panic") })}
	iteration, calls := schedulerCalls("missing")
	if err := l.executeToolCalls(t.Context(), iteration, calls, nil, nil, 1, 1, 0); !errors.Is(err, ErrToolPanic) {
		t.Fatalf("error=%v", err)
	}
	if iteration.Parts[0].ToolResp != nil {
		t.Fatal("panic retained raw result")
	}
}

func TestSerialCallsRemainFIFOWhenSharedGuardOpens(t *testing.T) {
	for range 50 {
		guard := &ToolGuard{}
		release, _ := guard.tryAcquire()
		order := make(chan string, 2)
		a := schedulerTool(t, "a", ToolOptions{Serial: true, Guard: guard}, func(_ context.Context, c ai.ToolCall) (string, error) { order <- c.ID; return "ok", nil })
		b := schedulerTool(t, "b", ToolOptions{}, func(context.Context, ai.ToolCall) (string, error) { release(); return "ok", nil })
		l := &Loop{Tools: []Tool{a, b}, ToolExecution: ToolExecutionConfig{MaxConcurrent: 2}}
		iteration, calls := schedulerCalls("a", "b", "a")
		if err := l.executeToolCalls(t.Context(), iteration, calls, l.Tools, nil, 1, 1, 0); err != nil {
			t.Fatal(err)
		}
		if first, second := <-order, <-order; first != "0" || second != "2" {
			t.Fatalf("serial order=%s,%s", first, second)
		}
	}
}

func TestSerialIncludesProcessing(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		entered := make(chan struct{})
		release := make(chan struct{})
		var invoked atomic.Int32
		tool := schedulerTool(t, "serial", ToolOptions{Serial: true}, func(context.Context, ai.ToolCall) (string, error) { invoked.Add(1); return "ok", nil })
		l := &Loop{Tools: []Tool{tool}, ToolResultProcessor: ToolResultProcessorFunc(func(_ context.Context, input ToolPolicyInput, result ToolResult) (ToolResult, error) {
			if input.Call.ID == "0" {
				close(entered)
				<-release
			}
			return result, nil
		})}
		iteration, calls := schedulerCalls("serial", "serial")
		done := make(chan error, 1)
		go func() { done <- l.executeToolCalls(t.Context(), iteration, calls, l.Tools, nil, 1, 1, 0) }()
		<-entered
		synctest.Wait()
		if invoked.Load() != 1 {
			t.Fatal("next serialized handler overlapped processing")
		}
		close(release)
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	})
}

func TestToolTimeoutExplicitZeroDisablesDefault(t *testing.T) {
	zero := time.Duration(0)
	tool := schedulerTool(t, "test", ToolOptions{Timeout: &zero}, func(ctx context.Context, _ ai.ToolCall) (string, error) {
		if _, ok := ctx.Deadline(); ok {
			t.Error("zero timeout inherited execution default")
		}
		return "ok", nil
	})
	l := &Loop{Tools: []Tool{tool}, ToolExecution: ToolExecutionConfig{DefaultTimeout: time.Second}}
	iteration, calls := schedulerCalls("test")
	if err := l.executeToolCalls(context.Background(), iteration, calls, l.Tools, nil, 1, 1, 0); err != nil {
		t.Fatal(err)
	}
}

func TestSerialMalformedCallProcessingWaitsForEarlierCall(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		entered := make(chan struct{})
		release := make(chan struct{})
		processed := make(chan string, 2)
		tool := schedulerTool(t, "serial", ToolOptions{Serial: true}, func(context.Context, ai.ToolCall) (string, error) { close(entered); <-release; return "ok", nil })
		l := &Loop{Tools: []Tool{tool}, ToolResultProcessor: ToolResultProcessorFunc(func(_ context.Context, input ToolPolicyInput, result ToolResult) (ToolResult, error) {
			processed <- input.Call.ID
			return result, nil
		})}
		iteration, calls := schedulerCalls("serial", "serial")
		calls[1].call.Args = json.RawMessage("broken")
		done := make(chan error, 1)
		go func() { done <- l.executeToolCalls(t.Context(), iteration, calls, l.Tools, nil, 1, 1, 0) }()
		<-entered
		synctest.Wait()
		if len(processed) != 0 {
			t.Fatal("malformed call bypassed serial processing")
		}
		close(release)
		if err := <-done; err != nil {
			t.Fatal(err)
		}
		if first := <-processed; first != "0" {
			t.Fatalf("processor order starts with %s", first)
		}
	})
}

func TestSchedulerDeduplicatesLargeSharedGuardBatch(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		guard := &ToolGuard{}
		release, _ := guard.tryAcquire()
		defer release()
		tool := schedulerTool(t, "test", ToolOptions{Guard: guard}, func(context.Context, ai.ToolCall) (string, error) {
			t.Error("held guard admitted handler")
			return "", nil
		})
		names := make([]string, 65535)
		for i := range names {
			names[i] = "test"
		}
		iteration, calls := schedulerCalls(names...)
		l := &Loop{Tools: []Tool{tool}, ToolExecution: ToolExecutionConfig{MaxConcurrent: 1}}
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		done := make(chan error, 1)
		go func() { done <- l.executeToolCalls(ctx, iteration, calls, l.Tools, nil, 1, 1, 0) }()
		synctest.Wait()
		cancel()
		if err := <-done; !errors.Is(err, context.Canceled) {
			t.Fatalf("err=%v", err)
		}
	})
}

func TestGuardCountRejectsTooManyDistinctWaiters(t *testing.T) {
	guards := make([]ToolGuard, maxToolGuardWaits+1)
	tasks := make([]scheduledTool, len(guards))
	for i := range tasks {
		tasks[i].options.Guard = &guards[i]
	}
	if err := validateToolGuardCount(tasks); !errors.Is(err, ErrToolExecutionConfig) {
		t.Fatalf("err=%v", err)
	}
	if err := validateToolGuardCount(tasks[:maxToolGuardWaits]); err != nil {
		t.Fatal(err)
	}
}

// TestHandlerResultSurvivesItsExpiredDeadline avoids replacing successful side effects with a false error.
func TestHandlerResultSurvivesItsExpiredDeadline(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tool := schedulerTool(t, "test", ToolOptions{}, func(ctx context.Context, _ ai.ToolCall) (string, error) { <-ctx.Done(); return "completed", nil })
		l := &Loop{Tools: []Tool{tool}, ToolExecution: ToolExecutionConfig{DefaultTimeout: time.Second}}
		iteration, calls := schedulerCalls("test")
		if err := l.executeToolCalls(t.Context(), iteration, calls, l.Tools, nil, 1, 1, 0); err != nil {
			t.Fatal(err)
		}
		if result := iteration.Parts[0].ToolResp; result.Err != nil || result.Text != "completed" {
			t.Fatalf("result=%#v", result)
		}
	})
}
