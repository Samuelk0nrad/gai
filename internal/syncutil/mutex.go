package syncutil

import (
	"context"
	"sync"
)

// ContextMutex is a zero-value-ready mutex that honors context cancellation
// while waiting for ownership. It is useful for serialized, optional discovery
// paths where a canceled caller must not wait behind network I/O.
type ContextMutex struct {
	once sync.Once
	ch   chan struct{}
}

func (m *ContextMutex) Lock(ctx context.Context) error {
	m.once.Do(func() {
		m.ch = make(chan struct{}, 1)
		m.ch <- struct{}{}
	})
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-m.ch:
		return nil
	}
}

func (m *ContextMutex) Unlock() {
	m.ch <- struct{}{}
}
