// Package appclock measures time GAI spends waiting on application code, such
// as observation sinks, content redactors, and consumers that accept streamed
// tokens. Instrumentation uses it to keep that time out of latencies it
// attributes to a provider.
package appclock

import (
	"context"
	"sync/atomic"
	"time"
)

// Clock accumulates application time measured under one context.
type Clock struct {
	excluded atomic.Int64
}

type contextKey struct{}

// With returns a context that carries a new Clock. Pause calls made with the
// returned context, or a context derived from it, add to that Clock.
func With(ctx context.Context) (context.Context, *Clock) {
	clock := &Clock{}
	return context.WithValue(ctx, contextKey{}, clock), clock
}

// Pause starts measuring application time under ctx. Call the returned
// function when control returns from the application. Without a Clock in ctx
// it measures nothing.
func Pause(ctx context.Context) (resume func()) {
	if ctx == nil {
		return func() {}
	}
	clock, _ := ctx.Value(contextKey{}).(*Clock)
	if clock == nil {
		return func() {}
	}
	started := time.Now()
	return func() { clock.excluded.Add(int64(time.Since(started))) }
}

// Since returns the time elapsed since start minus the application time
// measured so far. It never returns a negative duration. A nil Clock excludes
// nothing.
func (c *Clock) Since(start time.Time) time.Duration {
	elapsed := time.Since(start)
	if c == nil {
		return elapsed
	}
	return max(elapsed-time.Duration(c.excluded.Load()), 0)
}
