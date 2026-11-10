package loop

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"time"

	"github.com/Samuelk0nrad/gai/ai"
)

// ErrAttemptTimeout identifies a deadline created by RetryPolicy, rather than
// a caller or total-run deadline. It wraps context.DeadlineExceeded so callers
// can retain standard context handling while policy can retry it safely.
var ErrAttemptTimeout = attemptTimeoutError{}

type attemptTimeoutError struct{}

func (attemptTimeoutError) Error() string { return "retry attempt timeout" }
func (attemptTimeoutError) Unwrap() error { return context.DeadlineExceeded }

// RetryPolicy controls classified model-generation retries. MaxRetries: 0
// explicitly disables retries.
type RetryPolicy struct {
	MaxRetries     int
	InitialBackoff time.Duration
	MaxBackoff     time.Duration
	Multiplier     float64
	// Jitter is the proportional randomization amount in the inclusive range [0, 1].
	Jitter float64
	// JitterSource supplies a value in [0, 1) when applying jitter. Nil uses
	// math/rand/v2's default source. Out-of-range source values are constrained
	// to that range before use.
	JitterSource func() float64
	// Wait blocks for a retry delay or returns the context error when canceled.
	// Nil waits using a wall-clock timer.
	Wait              func(context.Context, time.Duration) error
	RespectRetryAfter bool
	AttemptTimeout    time.Duration
	TotalTimeout      time.Duration
}

// Validate verifies that RetryPolicy's public limits and durations are valid.
func (p RetryPolicy) Validate() error {
	switch {
	case p.MaxRetries < 0:
		return errors.New("retry policy MaxRetries must be non-negative")
	case p.InitialBackoff < 0:
		return errors.New("retry policy InitialBackoff must be non-negative")
	case p.MaxBackoff < 0:
		return errors.New("retry policy MaxBackoff must be non-negative")
	case p.AttemptTimeout < 0:
		return errors.New("retry policy AttemptTimeout must be non-negative")
	case p.TotalTimeout < 0:
		return errors.New("retry policy TotalTimeout must be non-negative")
	case math.IsNaN(p.Multiplier) || math.IsInf(p.Multiplier, 0):
		return errors.New("retry policy Multiplier must be finite")
	case math.IsNaN(p.Jitter) || p.Jitter < 0 || p.Jitter > 1:
		return errors.New("retry policy Jitter must be within [0, 1]")
	}
	return nil
}

func (p RetryPolicy) ShouldRetry(retries int, err error) bool {
	return p.hasRetryBudget(retries) && p.isRetryable(err)
}

func (p RetryPolicy) hasRetryBudget(retries int) bool {
	return retries < p.MaxRetries
}

func (p RetryPolicy) isRetryable(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) {
		return false
	}
	if errors.Is(err, ErrAttemptTimeout) {
		return true
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var provider *ai.ProviderError
	if !errors.As(err, &provider) {
		return false
	}
	return provider.Kind == ai.ProviderErrorRateLimited || provider.Kind == ai.ProviderErrorTransient
}

func (p RetryPolicy) Backoff(retries int, err error) time.Duration {
	var provider *ai.ProviderError
	if p.RespectRetryAfter && errors.As(err, &provider) && provider.RetryAfter > 0 {
		d := provider.RetryAfter
		if p.MaxBackoff > 0 && d > p.MaxBackoff {
			d = p.MaxBackoff
		}
		return d
	}
	d := p.InitialBackoff
	if d <= 0 {
		return 0
	}
	if p.MaxBackoff > 0 && d > p.MaxBackoff {
		d = p.MaxBackoff
	}
	m := p.Multiplier
	if m <= 0 {
		m = 2
	}
	for range retries {
		d = cappedDuration(float64(d)*m, p.MaxBackoff)
	}
	if p.Jitter > 0 {
		source := p.JitterSource
		if source == nil {
			source = rand.Float64
		}
		jitter := source()
		switch {
		case math.IsNaN(jitter) || jitter < 0:
			jitter = 0
		case jitter >= 1:
			jitter = math.Nextafter(1, 0)
		}
		d = cappedDuration(float64(d)*(1-p.Jitter+jitter*2*p.Jitter), p.MaxBackoff)
	}
	return d
}

func cappedDuration(value float64, maximum time.Duration) time.Duration {
	if value <= 0 || math.IsNaN(value) {
		return 0
	}
	if maximum <= 0 {
		maximum = time.Duration(math.MaxInt64)
	}
	if value >= float64(maximum) {
		return maximum
	}
	return time.Duration(value)
}

func (p RetryPolicy) wait(ctx context.Context, delay time.Duration) error {
	if p.Wait != nil {
		return p.Wait(ctx, delay)
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
