package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/lace-ai/gai/context/history"
)

// memoryStore illustrates the atomic contract. It is process-local and has no
// deletion API or disk durability. A database adapter must enforce the same
// comparison at its own transaction boundary across every writer.
type memoryStore struct {
	mu       sync.Mutex
	sessions map[string]history.HistorySnapshot
	sequence uint64
}

func (s *memoryStore) LoadHistory(ctx context.Context, id string) (history.HistorySnapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return history.HistorySnapshot{}, err
	}
	snapshot := s.sessions[id]
	snapshot.State = snapshot.State.Clone()
	return snapshot, nil
}

func (s *memoryStore) CompareAndSwapHistory(ctx context.Context, id string, expected history.Revision, next *history.HistoryState) (history.Revision, error) {
	if next == nil {
		return "", history.ErrHistoryStateRequired
	}
	// Validate canonical storage content. The caller keeps next stable until return.
	if _, err := json.Marshal(next); err != nil {
		return "", err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return "", err
	}
	actual := s.sessions[id].Revision
	if actual != expected {
		return "", &history.RevisionConflictError{SessionID: id, Expected: expected, Actual: actual}
	}
	if s.sequence == ^uint64(0) {
		return "", fmt.Errorf("revision space exhausted")
	}
	s.sequence++
	revision := history.Revision(fmt.Sprint(s.sequence))
	if s.sessions == nil {
		s.sessions = make(map[string]history.HistorySnapshot)
	}
	s.sessions[id] = history.HistorySnapshot{Revision: revision, State: next.Clone()}
	return revision, nil
}
