package appclock

import (
	"context"
	"testing"
	"time"
)

// TestClockExcludesPausedTime checks that time inside Pause is subtracted and
// time outside it is kept.
func TestClockExcludesPausedTime(t *testing.T) {
	const active, paused = 20 * time.Millisecond, 200 * time.Millisecond
	ctx, clock := With(context.Background())
	start := time.Now()

	time.Sleep(active)
	resume := Pause(ctx)
	time.Sleep(paused)
	resume()

	if got := clock.Since(start); got < active || got >= paused {
		t.Fatalf("Since = %v, want at least %v and less than the %v pause", got, active, paused)
	}
}

// TestPauseWithoutClockIsNoop checks that contexts without a Clock, and a nil
// Clock, measure plain elapsed time.
func TestPauseWithoutClockIsNoop(t *testing.T) {
	Pause(context.Background())()
	var nilCtx context.Context // Pause must tolerate a nil context.
	Pause(nilCtx)()

	var clock *Clock
	start := time.Now().Add(-time.Second)
	if got := clock.Since(start); got < time.Second {
		t.Fatalf("nil Clock Since = %v, want at least 1s", got)
	}
}

// TestClockCountsOverlappingPausesOnce checks that a pause nested inside
// another, as when a sink calls a redactor, is not subtracted twice.
func TestClockCountsOverlappingPausesOnce(t *testing.T) {
	const active, paused = 50 * time.Millisecond, 50 * time.Millisecond
	ctx, clock := With(context.Background())
	start := time.Now()

	outer := Pause(ctx)
	inner := Pause(ctx)
	time.Sleep(paused)
	inner()
	outer()
	outer() // A second resume has no effect.
	time.Sleep(active)

	if got := clock.Since(start); got < active || got >= active+paused {
		t.Fatalf("Since = %v, want at least %v and less than %v", got, active, active+paused)
	}
}

// TestClockExcludesOpenPause checks that Since subtracts a pause that has not
// ended yet.
func TestClockExcludesOpenPause(t *testing.T) {
	const active, paused = 20 * time.Millisecond, 200 * time.Millisecond
	ctx, clock := With(context.Background())
	start := time.Now()

	time.Sleep(active)
	resume := Pause(ctx)
	defer resume()
	time.Sleep(paused)

	if got := clock.Since(start); got < active || got >= paused {
		t.Fatalf("Since = %v, want at least %v and less than the %v pause", got, active, paused)
	}
}
