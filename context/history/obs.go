package history

import (
	"context"
	"strings"

	"github.com/lace-ai/gai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/internal/observe"
	"go.opentelemetry.io/otel/attribute"
)

const contextTracerName = "github.com/lace-ai/gai/context"

const (
	selectionStageLoaded    = "loaded"
	selectionStageCandidate = "candidate"
)

type historyObserver struct {
	op string

	debug     gai.ObservationSink
	operation *observe.Operation

	sessionID      string
	counterID      string
	tokenBudget    int
	summaryAmount  float32
	selectionStage string

	statePresent    bool
	summaryIncluded bool
	budgetReached   bool

	totalTokens       int
	turnCount         int
	includedTurnCount int
	messageCount      int
	contentCount      int

	summaryTokens         int
	summaryTotalTurnCount int
	summaryTurnCount      int
	summaryRemainingCount int
	summaryExisting       bool
	summarySkipReason     string
}

func newHistoryBuildObserver(ctx context.Context, debug gai.ObservationSink, sessionID string, tokenBudget int) (context.Context, *historyObserver) {
	ctx, operation := observe.Start(ctx, debug, contextTracerName, "context.history", "context.operation", "build", "context:HistorySource",
		attribute.String("context.source", "history"),
		attribute.String("context.session_id", sessionID),
		attribute.Int("context.token_budget", tokenBudget),
	)
	return ctx, &historyObserver{
		op:          "build",
		debug:       debug,
		operation:   operation,
		sessionID:   sessionID,
		tokenBudget: tokenBudget,
	}
}

func newHistoryCompactionObserver(ctx context.Context, debug gai.ObservationSink, sessionID string, tokenBudget int, summaryAmount float32) (context.Context, *historyObserver) {
	ctx, operation := observe.Start(ctx, debug, contextTracerName, "context.history", "context.operation", "compact", "context:Compactor",
		attribute.String("context.session_id", sessionID),
		attribute.Int("context.token_budget", tokenBudget),
		attribute.Float64("context.history.summary_amount", float64(summaryAmount)),
	)
	return ctx, &historyObserver{
		op:            "compact",
		debug:         debug,
		operation:     operation,
		sessionID:     sessionID,
		tokenBudget:   tokenBudget,
		summaryAmount: summaryAmount,
	}
}

func (o *historyObserver) Finish(err error) {
	if o == nil || o.operation == nil {
		return
	}

	switch o.op {
	case "build":
		o.operation.Set(
			attribute.Bool("context.history.state_present", o.statePresent),
			attribute.Bool("context.history.summary_included", o.summaryIncluded),
			attribute.Bool("context.history.budget_reached", o.budgetReached),
			attribute.Int("context.history.total_tokens", o.totalTokens),
			attribute.Int("context.history.turn_count", o.turnCount),
			attribute.Int("context.history.included_turn_count", o.includedTurnCount),
			attribute.Int("context.history.message_count", o.messageCount),
		)
	case "compact":
		attrs := []attribute.KeyValue{
			attribute.Int("context.history.turn_count", o.summaryTotalTurnCount),
			attribute.Bool("context.history.existing_summary", o.summaryExisting),
		}
		if o.summarySkipReason != "" {
			attrs = append(attrs, attribute.String("context.history.summary_skip_reason", o.summarySkipReason))
		}
		if o.summaryTurnCount > 0 {
			attrs = append(attrs,
				attribute.Int("context.history.summarized_turn_count", o.summaryTurnCount),
				attribute.Int("context.history.remaining_turn_count", o.summaryRemainingCount),
			)
		}
		if o.summaryTokens > 0 {
			attrs = append(attrs, attribute.Int("context.history.summary_tokens", o.summaryTokens))
		}
		o.operation.Set(attrs...)
	}

	o.operation.Finish(err)
}

func (o *historyObserver) SetTokenCounterID(counterID string) {
	if o == nil || o.operation == nil {
		return
	}
	o.counterID = counterID
	o.operation.Set(attribute.String("context.counter_id", counterID))
}

func (o *historyObserver) MarkStatePresent() {
	if o == nil {
		return
	}
	o.statePresent = true
}

func (o *historyObserver) MarkBudgetReached() {
	if o == nil {
		return
	}
	o.budgetReached = true
}

func (o *historyObserver) StoreMissing(ctx context.Context) {
	o.emit(ctx, "history_source_store_missing", map[string]any{
		"session_id": o.sessionID,
	}, gaictx.ErrSessionStoreNotFound)
}

func (o *historyObserver) TokenCounterMissing(ctx context.Context) {
	o.emit(ctx, "history_source_counter_missing", map[string]any{
		"session_id": o.sessionID,
	}, gaictx.ErrTokenCounterNotFound)
}

func (o *historyObserver) StateLoadFailed(ctx context.Context, err error) {
	o.emit(ctx, "history_source_state_load_failed", map[string]any{
		"session_id": o.sessionID,
		"counter_id": o.counterID,
	}, err)
}

func (o *historyObserver) StateMissing(ctx context.Context) {
	o.emit(ctx, "history_source_state_missing", map[string]any{
		"session_id": o.sessionID,
		"counter_id": o.counterID,
	}, nil)
}

func (o *historyObserver) SummaryIncluded(ctx context.Context, summary *Summary, tokens int) {
	if o == nil || summary == nil {
		return
	}
	o.summaryIncluded = true
	fields := map[string]any{
		"session_id":          o.sessionID,
		"counter_id":          o.counterID,
		"summary_tokens":      tokens,
		"summary_start_turn":  summary.StartTurnID,
		"summary_end_turn":    summary.EndTurnID,
		"summary_start_count": summary.StartTurnCount,
		"summary_end_count":   summary.EndTurnCount,
	}
	gai.AddObservationContent(ctx, o.debug, fields, "summary_content", gai.ContentKindMemory, summary.Content.Text)
	o.emit(ctx, "history_source_summary_included", fields, nil)
}

func (o *historyObserver) SummaryTokenCountFailed(ctx context.Context, summary *Summary, err error) {
	if o == nil || summary == nil {
		return
	}
	fields := map[string]any{
		"session_id":          o.sessionID,
		"counter_id":          o.counterID,
		"summary_start_turn":  summary.StartTurnID,
		"summary_end_turn":    summary.EndTurnID,
		"summary_start_count": summary.StartTurnCount,
		"summary_end_count":   summary.EndTurnCount,
	}
	gai.AddObservationContent(ctx, o.debug, fields, "summary_content", gai.ContentKindMemory, summary.Content.Text)
	o.emit(ctx, "history_source_summary_token_count_failed", fields, err)
}

func (o *historyObserver) SummaryMissing(ctx context.Context) {
	o.emit(ctx, "history_source_summary_missing", map[string]any{
		"session_id": o.sessionID,
		"counter_id": o.counterID,
	}, nil)
}

func (o *historyObserver) TurnTokenizeFailed(ctx context.Context, turn *gaictx.Turn, err error) {
	o.emit(ctx, "history_source_turn_tokenize_failed", map[string]any{
		"session_id": o.sessionID,
		"counter_id": o.counterID,
		"turn_id":    turn.ID,
		"turn_count": turn.Count,
	}, err)
}

func (o *historyObserver) BudgetReached(ctx context.Context, totalTokens int, turn *gaictx.Turn) {
	o.MarkBudgetReached()
	fields := map[string]any{
		"session_id":   o.sessionID,
		"counter_id":   o.counterID,
		"token_budget": o.tokenBudget,
		"total_tokens": totalTokens,
	}
	if turn != nil {
		fields["last_turn_id"] = turn.ID
		fields["last_turn_cnt"] = turn.Count
	}
	o.emit(ctx, "history_source_token_budget_reached", fields, nil)
}

func (o *historyObserver) BuildFinished(ctx context.Context, part *Part, tokenCount, turnCount, includedTurnCount, messageCount int) {
	if o == nil {
		return
	}
	o.totalTokens = tokenCount
	o.turnCount = turnCount
	o.includedTurnCount = includedTurnCount
	o.messageCount = messageCount
	if part != nil {
		o.contentCount = len(part.Messages)
	}

	fields := map[string]any{
		"session_id":    o.sessionID,
		"counter_id":    o.counterID,
		"token_budget":  o.tokenBudget,
		"total_tokens":  tokenCount,
		"turn_count":    turnCount,
		"message_count": messageCount,
		"content_count": o.contentCount,
	}
	if part != nil && gai.ObservationContentEnabled(ctx, o.debug, gai.ContentKindMemory) {
		if rendered, err := (gaictx.XMLRenderer{}).Render(ctx, []gaictx.Part{part}); err == nil {
			gai.AddObservationContent(ctx, o.debug, fields, "history_content", gai.ContentKindMemory, rendered)
		}
	}
	o.emit(ctx, "history_source_build_finished", fields, nil)
}

func (o *historyObserver) SummaryGenerated(ctx context.Context, summary *Summary, tokens, summarizedTurnCount, remainingTurnCount int, previousSummaryFound bool) {
	if o == nil || summary == nil {
		return
	}
	o.summaryTotalTurnCount = summarizedTurnCount + remainingTurnCount
	o.summaryTurnCount = summarizedTurnCount
	o.summaryRemainingCount = remainingTurnCount
	o.summaryTokens = tokens
	o.summaryExisting = previousSummaryFound

	fields := map[string]any{
		"session_id":             o.sessionID,
		"counter_id":             o.counterID,
		"token_budget":           o.tokenBudget,
		"summary_tokens":         o.summaryTokens,
		"summarized_turn_count":  summarizedTurnCount,
		"remaining_turn_count":   remainingTurnCount,
		"summary_start_turn":     summary.StartTurnID,
		"summary_end_turn":       summary.EndTurnID,
		"summary_start_count":    summary.StartTurnCount,
		"summary_end_count":      summary.EndTurnCount,
		"previous_summary_found": previousSummaryFound,
	}
	gai.AddObservationContent(ctx, o.debug, fields, "summary_content", gai.ContentKindMemory, summary.Content.Text)
	o.emit(ctx, "history_source_summary_generated", fields, nil)
}

func (o *historyObserver) ObserveState(turnCount int, existingSummary bool) {
	if o == nil {
		return
	}
	o.summaryTotalTurnCount = turnCount
	o.summaryExisting = existingSummary
}

func (o *historyObserver) SummarySkippedNoTurns(ctx context.Context) {
	o.summarySkipReason = "no_turns"
	o.emit(ctx, "history_source_summary_skipped", map[string]any{
		"session_id":   o.sessionID,
		"token_budget": o.tokenBudget,
		"reason":       "no_turns",
	}, nil)
}

func (o *historyObserver) SummarySkippedAmountZero(ctx context.Context) {
	o.summarySkipReason = "amount_zero"
	o.emit(ctx, "history_source_summary_skipped", map[string]any{
		"session_id":     o.sessionID,
		"token_budget":   o.tokenBudget,
		"summary_amount": float64(o.summaryAmount),
		"reason":         "amount_zero",
	}, nil)
}

func (o *historyObserver) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if o == nil || o.operation == nil {
		return
	}
	if o.op == "compact" {
		name = strings.Replace(name, "history_source_", "history_compactor_", 1)
		if o.selectionStage != "" {
			fields["selection_stage"] = o.selectionStage
		}
	}
	o.operation.Emit(ctx, name, fields, err)
}

// CompactionSelection identifies whether diagnostics describe loaded history
// or an uncommitted candidate, without changing ordinary build observations.
func (o *historyObserver) CompactionSelection(ctx context.Context, state *HistoryState, result selection, err error, stage string) {
	previousStage := o.selectionStage
	o.selectionStage = stage
	defer func() { o.selectionStage = previousStage }()
	o.Selection(ctx, state, result, err)
}

// Selection reports the pure selector's observations without coupling selection
// to telemetry or allowing diagnostic callbacks to mutate the selected state.
func (o *historyObserver) Selection(ctx context.Context, state *HistoryState, result selection, err error) {
	if state == nil {
		return
	}
	if result.summaryFailed {
		o.SummaryTokenCountFailed(ctx, state.Summary, err)
	}
	if result.failedTurn != nil {
		o.TurnTokenizeFailed(ctx, result.failedTurn, err)
	}
	if state.Summary == nil {
		o.SummaryMissing(ctx)
	}
	if result.summaryIncluded {
		o.SummaryIncluded(ctx, state.Summary, result.summaryTokens)
	}
	if result.budgetReached {
		o.BudgetReached(ctx, result.tokens, result.budgetTurn)
	}
}

func (o *historyObserver) CompactionConflict(ctx context.Context, conflict *RevisionConflictError) {
	o.emit(ctx, "history_compactor_conflict", map[string]any{"session_id": conflict.SessionID}, conflict)
}

func (o *historyObserver) CompactionFinished(ctx context.Context, result CompactionResult, err error) {
	o.operation.Set(attribute.Bool("context.history.state_saved", result.Changed), attribute.Bool("context.history.pressure_remaining", result.PressureRemaining))
	o.emit(ctx, "history_compactor_finished", map[string]any{"session_id": o.sessionID, "changed": result.Changed, "pressure_remaining": result.PressureRemaining}, err)
}
