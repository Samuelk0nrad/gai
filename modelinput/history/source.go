package history

import (
	"context"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/modelinput"
)

// HistorySource renders persisted conversation history as prompt context.
type HistorySource struct {
	historyStateStore HistoryReader
	sessionID         string

	debug   gai.ObservationSink
	counter ai.TokenCounter
}

// NewHistory creates a read-only history source. Its ordinary source functions
// never run a summarizer or persist state. Use NewCompactor for explicit
// compaction or agent.Definition.AutoCompactHistory for preparation at run start.
func NewHistory(sessionID string, reader HistoryReader) *HistorySource {
	return &HistorySource{historyStateStore: reader, sessionID: sessionID}
}

func (p *HistorySource) Name() string {
	return "history"
}

func (s *HistorySource) SetTokenCounter(counter ai.TokenCounter) {
	s.counter = counter
}

func (s *HistorySource) ObservationSink(debug gai.ObservationSink, conv modelinput.Conversation) {
	s.debug = debug
}

// Function returns a semantic history part selected for tokenBudget.
func (s *HistorySource) Function(ctx context.Context, tokenBudget int) (modelinput.Part, error) {
	part, _, err := s.build(ctx, tokenBudget)
	return part, err
}

// FunctionWithTokens returns the selected part and the token total calculated
// during this invocation with the configured counter. Builder can consume that
// total without recounting the part; the part itself remains stateless.
func (s *HistorySource) FunctionWithTokens(ctx context.Context, tokenBudget int) (modelinput.Part, int, error) {
	return s.build(ctx, tokenBudget)
}

// FunctionWithBudget selects canonical history whose build-local count already
// includes emitted message framing. History bypasses the arbitrary-part renderer,
// so its existing projection is identical to the builder's project callback.
func (s *HistorySource) FunctionWithBudget(ctx context.Context, tokenBudget int, _ func(context.Context, modelinput.Part) (int, error)) (modelinput.Part, int, error) {
	return s.build(ctx, tokenBudget)
}

// build selects and counts one snapshot. Its token total is local to this call.
func (s *HistorySource) build(ctx context.Context, tokenBudget int) (result modelinput.Part, tokens int, err error) {
	ctx, obs := newHistoryBuildObserver(ctx, s.debug, s.sessionID, tokenBudget)
	defer func() {
		obs.Finish(err)
	}()

	if s.historyStateStore == nil {
		obs.StoreMissing(ctx)
		return nil, 0, modelinput.ErrSessionStoreNotFound
	}
	if s.counter == nil {
		obs.TokenCounterMissing(ctx)
		return nil, 0, modelinput.ErrTokenCounterNotFound
	}
	counterID := s.counter.ID()
	obs.SetTokenCounterID(counterID)
	snapshot, err := s.historyStateStore.LoadHistory(ctx, s.sessionID)
	if err != nil {
		obs.StateLoadFailed(ctx, err)
		return nil, 0, err
	}
	if err = snapshot.validate(); err != nil {
		obs.StateLoadFailed(ctx, err)
		return nil, 0, err
	}
	state := snapshot.State
	if state == nil {
		obs.StateMissing(ctx)
	} else {
		obs.MarkStatePresent()
	}
	selected, err := selectHistory(ctx, state, tokenBudget, s.counter)
	obs.Selection(ctx, state, selected, err)
	if err != nil {
		return nil, 0, err
	}
	obs.BuildFinished(ctx, &selected.part, selected.tokens, selected.turnsVisited, selected.turnsIncluded, selected.messages)
	return &selected.part, selected.tokens, nil
}
