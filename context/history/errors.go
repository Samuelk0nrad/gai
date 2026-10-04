package history

import (
	"errors"
	"fmt"
)

var (
	ErrInvalidSummaryAmount    = errors.New("summary amount must be between 0 and 1")
	ErrInvalidSummaryMaxTokens = errors.New("summary max tokens must be non-negative")
	ErrSummarizerRequired      = errors.New("compaction requires a model or summarizer")
	ErrHistoryStateRequired    = errors.New("history state is required")
	ErrSummarizerMissing       = errors.New("summarizer not configured for compaction")
	ErrCompactorNil            = errors.New("compactor is nil")
	ErrInvalidHistoryBudget    = errors.New("history budget must be non-negative")
	ErrInvalidHistorySnapshot  = errors.New("initialized history requires a nonempty revision")
)

// RevisionConflictError reports a definite rejected stale write. It may be
// wrapped; use errors.As to detect it. Revisions are opaque equality tokens.
type RevisionConflictError struct {
	SessionID string
	Expected  Revision
	Actual    Revision
}

func (e *RevisionConflictError) Error() string {
	return fmt.Sprintf("history revision conflict for session %q: expected %q, actual %q", e.SessionID, e.Expected, e.Actual)
}
