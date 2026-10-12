package history

import "context"

// Revision is an opaque, store-generated concurrency token. Compare revisions
// only for equality; it is unrelated to schema versions, turn IDs, and counts.
// The empty revision denotes a session that has never been initialized.
type Revision string

// HistorySnapshot is one coherent persisted version of working history.
// LoadHistory returns detached state, including nested canonical payloads.
// State nil with an empty Revision means never-created history; nil with a
// nonempty Revision is a tombstone retained by stores that support deletion.
// Initialized history, including an empty HistoryState, has a nonempty Revision.
type HistorySnapshot struct {
	Revision Revision
	State    *HistoryState
}

func (s HistorySnapshot) validate() error {
	if s.State != nil && s.Revision == "" {
		return ErrInvalidHistorySnapshot
	}
	return nil
}

// HistoryReader loads stable, detached snapshots. Implementations must copy
// under their synchronization boundary, not after returning a live pointer.
// A caller may mutate a returned state without changing storage or other loads.
type HistoryReader interface {
	LoadHistory(ctx context.Context, sessionID string) (HistorySnapshot, error)
}

// HistoryStore persists working history, not necessarily an archival transcript.
// Calculated token counts are never persisted.
//
// CompareAndSwapHistory atomically compares expected with the current revision
// and replaces the state only on equality. It returns a fresh nonempty revision
// on success, and *RevisionConflictError (without changing state) on mismatch.
// Empty expected is create-only, never an unconditional-write wildcard.
// next must be nonnil; persist an empty HistoryState to clear working history.
// The store must detach next before returning, so later caller mutations cannot
// alter persisted content. Callers keep next stable for the duration of the call.
//
// Every mutation path, including native append/edit/delete APIs, must update the
// same revision atomically. Revisions must never be reused, including after
// deletion/recreation. Deletion is outside this interface; stores offering it
// must retain a tombstone or an equivalent generation identity.
//
// Honor cancellation before committing. A transport error or cancellation during
// commit can have an unknown outcome; only a revision conflict guarantees that
// this CAS did not commit. GAI does not retry writes or expensive model work.
type HistoryStore interface {
	HistoryReader
	CompareAndSwapHistory(ctx context.Context, sessionID string, expected Revision, next *HistoryState) (Revision, error)
}
