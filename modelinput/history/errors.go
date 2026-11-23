package history

import (
	"errors"
	"fmt"
)

var (
	ErrInvalidSummaryAmount     = errors.New("summary amount must be between 0 and 1")
	ErrInvalidSummaryMaxTokens  = errors.New("summary max tokens must be non-negative")
	ErrSummarizerRequired       = errors.New("compaction requires a model or summarizer")
	ErrHistoryStateRequired     = errors.New("history state is required")
	ErrSummarizerMissing        = errors.New("summarizer not configured for compaction")
	ErrCompactorNil             = errors.New("compactor is nil")
	ErrInvalidHistoryBudget     = errors.New("history budget must be non-negative")
	ErrInvalidHistorySnapshot   = errors.New("initialized history requires a nonempty revision")
	ErrHistoryStoreRequired     = errors.New("automatic history compaction requires a writable HistoryStore")
	ErrHistoryPressureRemaining = errors.New("history still exceeds its allocation after compaction")
	// ErrSummaryPrefixRequired means a continuation summarizer has no request
	// to continue. Pass WithPrefix to Compact, or configure another Summarizer.
	ErrSummaryPrefixRequired = errors.New("continuation summary requires a prefix request")
	// ErrSummaryPrefixMismatch means the continued request does not contain
	// the stored summary or a turn to fold, so the model would not see it.
	ErrSummaryPrefixMismatch = errors.New("continuation prefix does not contain the history to summarize")
	// ErrSummaryRequestTooLarge means no summary request, even one folding a
	// single turn, fits the request budget.
	ErrSummaryRequestTooLarge = errors.New("summary request exceeds the request budget")
	// ErrInvalidSummaryPrefix rejects a prefix the summary call cannot extend.
	ErrInvalidSummaryPrefix = errors.New("invalid continuation summary prefix")
	// ErrInvalidSummaryReply rejects a summary reply that is empty, truncated,
	// or calls a tool.
	ErrInvalidSummaryReply          = errors.New("invalid summary reply")
	ErrInvalidSummaryResponseFormat = errors.New("unknown summary response format")
	ErrInvalidExpectedRevision      = errors.New("expected revision must be nonempty")
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
