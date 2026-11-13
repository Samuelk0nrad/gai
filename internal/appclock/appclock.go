// Package appclock measures time GAI spends waiting on application code, such
// as observation sinks, content redactors, and consumers that accept streamed
// tokens. Instrumentation uses it to keep that time out of latencies it
// attributes to a provider.
package appclock

import (
	"context"
	"sync"
	"time"
)

// Clock accumulates application time measured under one context. Pauses that
// overlap, whether nested or on different goroutines, count once: the Clock
// excludes the time during which at least one pause is open.
type Clock struct {
	mu       sync.Mutex
	open     int
	openedAt time.Time
	excluded time.Duration
}

type contextKey struct{}

// With returns a context that carries a new Clock. Pause calls made with the
// returned context, or a context derived from it, add to that Clock.
func With(ctx context.Context) (context.Context, *Clock) {
	clock := &Clock{}
	return context.WithValue(ctx, contextKey{}, clock), clock
}

// Pause starts measuring application time under ctx. Call the returned
// function when control returns from the application; calling it more than
// once has no further effect. Without a Clock in ctx it measures nothing.
func Pause(ctx context.Context) (resume func()) {
	if ctx == nil {
		return func() {}
	}
	clock, _ := ctx.Value(contextKey{}).(*Clock)
	if clock == nil {
		return func() {}
	}
	clock.mu.Lock()
	if clock.open == 0 {
		clock.openedAt = time.Now()
	}
	clock.open++
	clock.mu.Unlock()

	resumed := false
	return func() {
		clock.mu.Lock()
		defer clock.mu.Unlock()
		if resumed {
			return
		}
		resumed = true
		clock.open--
		if clock.open == 0 {
			clock.excluded += time.Since(clock.openedAt)
		}
	}
}

// Since returns the time elapsed since start minus the application time
// measured so far, including a pause that is still open. It never returns a
// negative duration. A nil Clock excludes nothing.
func (c *Clock) Since(start time.Time) time.Duration {
	elapsed := time.Since(start)
	if c == nil {
		return elapsed
	}
	c.mu.Lock()
	excluded := c.excluded
	if c.open > 0 {
		excluded += time.Since(c.openedAt)
	}
	c.mu.Unlock()
	return max(elapsed-excluded, 0)
}
