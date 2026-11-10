package loop_test

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/loop"
)

func TestRetryPolicyClassifiesProviderErrors(t *testing.T) {
	policy := loop.RetryPolicy{MaxRetries: 1}
	if !policy.ShouldRetry(0, &ai.ProviderError{Kind: ai.ProviderErrorRateLimited}) {
		t.Fatal("rate limit should retry")
	}
	if policy.ShouldRetry(0, &ai.ProviderError{Kind: ai.ProviderErrorInvalidRequest}) {
		t.Fatal("invalid request must not retry")
	}
	if policy.ShouldRetry(1, &ai.ProviderError{Kind: ai.ProviderErrorTransient}) {
		t.Fatal("exhausted retries must not retry")
	}
}

func TestRetryPolicyAttemptTimeoutRetriesButCallerCancellationDoesNot(t *testing.T) {
	policy := loop.RetryPolicy{MaxRetries: 1, AttemptTimeout: time.Millisecond}
	if !policy.ShouldRetry(0, loop.ErrAttemptTimeout) {
		t.Fatal("policy attempt timeout should retry")
	}
	if policy.ShouldRetry(0, context.Canceled) || policy.ShouldRetry(0, context.DeadlineExceeded) {
		t.Fatal("caller cancellation and deadline must not retry")
	}
	if !errors.Is(loop.ErrAttemptTimeout, context.DeadlineExceeded) {
		t.Fatal("attempt timeout must retain deadline identity")
	}
}

func TestRetryPolicyBackoffNeverExceedsMaximumWithJitter(t *testing.T) {
	policy := loop.RetryPolicy{
		InitialBackoff: 10 * time.Millisecond,
		MaxBackoff:     10 * time.Millisecond,
		Multiplier:     2,
		Jitter:         1,
	}
	for range 64 {
		if backoff := policy.Backoff(1, nil); backoff > policy.MaxBackoff {
			t.Fatalf("backoff %s exceeds maximum %s", backoff, policy.MaxBackoff)
		}
	}
}

func TestRetryPolicyBackoffClampsInitialBackoffToMaximum(t *testing.T) {
	policy := loop.RetryPolicy{
		InitialBackoff: time.Second,
		MaxBackoff:     10 * time.Millisecond,
	}

	if backoff := policy.Backoff(0, nil); backoff != policy.MaxBackoff {
		t.Fatalf("backoff = %s, want maximum %s", backoff, policy.MaxBackoff)
	}
}

func TestRetryPolicyBackoffClampsRetryAfterToMaximum(t *testing.T) {
	policy := loop.RetryPolicy{
		MaxBackoff:        10 * time.Millisecond,
		RespectRetryAfter: true,
	}
	err := &ai.ProviderError{RetryAfter: time.Second}

	if backoff := policy.Backoff(0, err); backoff != policy.MaxBackoff {
		t.Fatalf("backoff = %s, want maximum %s", backoff, policy.MaxBackoff)
	}
}

func TestRetryPolicyBackoffUsesInjectedJitterSource(t *testing.T) {
	policy := loop.RetryPolicy{
		InitialBackoff: 10 * time.Millisecond,
		Jitter:         0.5,
		JitterSource: func() float64 {
			return 0.75
		},
	}

	if backoff := policy.Backoff(0, nil); backoff != 12500*time.Microsecond {
		t.Fatalf("backoff = %s, want 12.5ms", backoff)
	}
}

func TestRetryPolicyBackoffCapsScaledDurationsBeforeConversion(t *testing.T) {
	t.Run("multiplier", func(t *testing.T) {
		policy := loop.RetryPolicy{
			InitialBackoff: time.Second,
			MaxBackoff:     time.Millisecond,
			Multiplier:     math.MaxFloat64,
		}

		if backoff := policy.Backoff(1, nil); backoff != policy.MaxBackoff {
			t.Fatalf("backoff = %s, want maximum %s", backoff, policy.MaxBackoff)
		}
	})

	t.Run("jitter", func(t *testing.T) {
		policy := loop.RetryPolicy{
			InitialBackoff: time.Duration(math.MaxInt64),
			Jitter:         1,
			JitterSource: func() float64 {
				return math.Nextafter(1, 0)
			},
		}

		if backoff := policy.Backoff(0, nil); backoff != time.Duration(math.MaxInt64) {
			t.Fatalf("backoff = %s, want maximum duration", backoff)
		}
	})
}

func TestRetryPolicyValidateRejectsInvalidPublicValues(t *testing.T) {
	tests := []struct {
		name   string
		policy loop.RetryPolicy
	}{
		{name: "negative retries", policy: loop.RetryPolicy{MaxRetries: -1}},
		{name: "negative initial backoff", policy: loop.RetryPolicy{InitialBackoff: -time.Nanosecond}},
		{name: "negative maximum backoff", policy: loop.RetryPolicy{MaxBackoff: -time.Nanosecond}},
		{name: "negative attempt timeout", policy: loop.RetryPolicy{AttemptTimeout: -time.Nanosecond}},
		{name: "negative total timeout", policy: loop.RetryPolicy{TotalTimeout: -time.Nanosecond}},
		{name: "NaN multiplier", policy: loop.RetryPolicy{Multiplier: math.NaN()}},
		{name: "positive infinite multiplier", policy: loop.RetryPolicy{Multiplier: math.Inf(1)}},
		{name: "negative infinite multiplier", policy: loop.RetryPolicy{Multiplier: math.Inf(-1)}},
		{name: "negative jitter", policy: loop.RetryPolicy{Jitter: -0.01}},
		{name: "jitter above one", policy: loop.RetryPolicy{Jitter: 1.01}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.policy.Validate(); err == nil {
				t.Fatal("Validate() succeeded for an invalid retry policy")
			}
		})
	}
}

func TestLoopValidateRejectsInvalidRetryPolicy(t *testing.T) {
	l := loop.New(&scriptedStreamModel{}, nil, testPromptBuilder(), nil)
	l.RetryPolicy = &loop.RetryPolicy{Jitter: 1.01}

	if err := l.Validate(); err == nil {
		t.Fatal("Loop.Validate() succeeded with an invalid retry policy")
	}
}

func TestRetryPolicyBackoffGuardsInjectedJitterSource(t *testing.T) {
	policy := loop.RetryPolicy{
		InitialBackoff: 10 * time.Millisecond,
		Jitter:         1,
		JitterSource: func() float64 {
			return -1
		},
	}

	if backoff := policy.Backoff(0, nil); backoff < 0 {
		t.Fatalf("backoff = %s, must not be negative", backoff)
	}
}
