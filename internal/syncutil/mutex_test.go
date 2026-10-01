package syncutil

import (
	"context"
	"errors"
	"testing"
)

func TestContextMutexHonorsCanceledContext(t *testing.T) {
	var mu ContextMutex
	if err := mu.Lock(context.Background()); err != nil {
		t.Fatalf("first lock: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := mu.Lock(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled lock error = %v, want context.Canceled", err)
	}
	mu.Unlock()
}
