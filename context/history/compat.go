// Package history preserves the former model-input history import path.
//
// Deprecated: Use github.com/lace-ai/gai/modelinput/history.
// It remains available until its removal in a future breaking release.
package history

import (
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/modelinput/history"
)

type (
	Revision              = history.Revision
	HistorySnapshot       = history.HistorySnapshot
	HistoryReader         = history.HistoryReader
	HistoryStore          = history.HistoryStore
	HistoryState          = history.HistoryState
	HistorySource         = history.HistorySource
	Summary               = history.Summary
	Part                  = history.Part
	CompactorDefinition   = history.CompactorDefinition
	CompactionResult      = history.CompactionResult
	Compactor             = history.Compactor
	RevisionConflictError = history.RevisionConflictError
)

const HistorySchemaVersion = history.HistorySchemaVersion

var (
	ErrInvalidSummaryAmount     = history.ErrInvalidSummaryAmount
	ErrInvalidSummaryMaxTokens  = history.ErrInvalidSummaryMaxTokens
	ErrSummarizerRequired       = history.ErrSummarizerRequired
	ErrHistoryStateRequired     = history.ErrHistoryStateRequired
	ErrSummarizerMissing        = history.ErrSummarizerMissing
	ErrCompactorNil             = history.ErrCompactorNil
	ErrInvalidHistoryBudget     = history.ErrInvalidHistoryBudget
	ErrInvalidHistorySnapshot   = history.ErrInvalidHistorySnapshot
	ErrHistoryStoreRequired     = history.ErrHistoryStoreRequired
	ErrHistoryPressureRemaining = history.ErrHistoryPressureRemaining
)

func NewHistory(sessionID string, reader HistoryReader) *HistorySource {
	return history.NewHistory(sessionID, reader)
}

func NewCompactor(sessionID string, store HistoryStore, def CompactorDefinition) (*Compactor, error) {
	return history.NewCompactor(sessionID, store, def)
}

func NewSummary(id, startTurnID, endTurnID string, startTurnCount, endTurnCount int, content ai.ContentPart) *Summary {
	return history.NewSummary(id, startTurnID, endTurnID, startTurnCount, endTurnCount, content)
}
