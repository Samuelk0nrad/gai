package main

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/history"
)

// sessionLocks serializes complete runs, while allowing unrelated sessions to
// proceed. References include both the active owner and waiters, so removing an
// idle gate cannot split one live session into two independent locks.
type sessionLocks struct {
	mu    sync.Mutex
	gates map[string]*sessionGate
}
type sessionGate struct {
	token chan struct{}
	refs  int
}

func (s *sessionLocks) acquire(ctx context.Context, id string) (func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	if s.gates == nil {
		s.gates = make(map[string]*sessionGate)
	}
	gate := s.gates[id]
	if gate == nil {
		gate = &sessionGate{token: make(chan struct{}, 1)}
		gate.token <- struct{}{}
		s.gates[id] = gate
	}
	gate.refs++
	s.mu.Unlock()
	drop := func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		gate.refs--
		if gate.refs == 0 {
			delete(s.gates, id)
		}
	}
	select {
	case <-ctx.Done():
		drop()
		return nil, ctx.Err()
	case <-gate.token:
		release := func() { gate.token <- struct{}{}; drop() }
		if err := ctx.Err(); err != nil {
			release()
			return nil, err
		}
		return release, nil
	}
}

// fixedHistory pins prompt selection to the exact snapshot whose revision will
// be used for the final CAS. A source that reloaded storage during prompt setup
// could silently use a different version than the eventual write precondition.
type fixedHistory struct{ snapshot history.HistorySnapshot }

func (r fixedHistory) LoadHistory(ctx context.Context, _ string) (history.HistorySnapshot, error) {
	if err := ctx.Err(); err != nil {
		return history.HistorySnapshot{}, err
	}
	return history.HistorySnapshot{Revision: r.snapshot.Revision, State: r.snapshot.State.Clone()}, nil
}

var errHistoryPressure = errors.New("history still exceeds its allocation after compaction")

type chatService struct {
	store         history.HistoryStore
	model         ai.Model
	compaction    *history.CompactorDefinition
	requestWindow int
	locks         sessionLocks
}

// Run owns the session through compaction, generation/tools, and persistence.
// A nonnil error means the application must not claim persistence succeeded.
// The completed workflow result is still returned on a commit error: external
// effects/output already produced by that run must not be replayed automatically.
func (s *chatService) Run(ctx context.Context, sessionID, text string) (agent.WorkflowResult, error) {
	var result agent.WorkflowResult
	release, err := s.locks.acquire(ctx, sessionID)
	if err != nil {
		return result, err
	}
	defer release()

	// This example has no system instructions, tools, or structured options.
	// Count its fixed request using the same estimator as the actual run.
	const outputReserve = 64
	counter := ai.TextTokenEstimator{}
	fixed, err := ai.EstimateRequestTokens(ctx, ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, text)}}, counter)
	if err != nil {
		return result, err
	}
	if s.requestWindow <= 0 || fixed.InputTokens > s.requestWindow-outputReserve {
		return result, fmt.Errorf("%w: current input and output reserve exceed window", ai.ErrRequestBudgetExceeded)
	}
	historyBudget := s.requestWindow - outputReserve - fixed.InputTokens
	if s.compaction != nil {
		definition := *s.compaction
		definition.TokenCounter = counter
		compactor, err := history.NewCompactor(sessionID, s.store, definition)
		if err != nil {
			return result, err
		}
		compacted, err := compactor.Compact(ctx, historyBudget)
		if err != nil {
			return result, fmt.Errorf("compact history: %w", err)
		}
		// Explicit application policy: stop if one compaction was insufficient.
		// Do not loop, repeat paid work, or hide loss of older context here.
		if compacted.PressureRemaining {
			return result, errHistoryPressure
		}
	}
	snapshot, err := s.store.LoadHistory(ctx, sessionID)
	if err != nil {
		return result, err
	}
	a := agent.New(agent.Definition{
		Model:        s.model,
		TokenCounter: counter,
		Limits:       agent.Limits{MaxTokens: outputReserve},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{
				TokenBudget:        s.requestWindow,
				OutputTokenReserve: outputReserve,
				TokenCounter:       counter,
				ContextSources:     []gaictx.ContextSource{history.NewHistory(sessionID, fixedHistory{snapshot})},
			}), nil
		},
	})
	workflow, err := a.NewRun(ctx, agent.RunInput{Prompt: gaictx.PromptInput{User: ai.TextParts(text)}})
	if err != nil {
		return result, err
	}
	result, err = workflow.Run(ctx)
	if err != nil {
		return result, err
	}

	next := snapshot.State.Clone()
	if next == nil {
		next = &history.HistoryState{SchemaVersion: history.HistorySchemaVersion}
	}
	count := 0
	if next.Summary != nil {
		count = next.Summary.EndTurnCount
	}
	for _, turn := range next.Turns {
		if turn.Count > count {
			count = turn.Count
		}
	}
	if count == int(^uint(0)>>1) {
		return result, fmt.Errorf("turn count exhausted")
	}
	turn := completedTurn(sessionID, count+1, result.Primary.Messages)
	next.Turns = append(next.Turns, turn)
	// Use the prompt's revision, even if another writer bypassed our local lock.
	// Never reload here merely to obtain a revision that will accept stale output.
	_, err = s.store.CompareAndSwapHistory(ctx, sessionID, snapshot.Revision, next)
	if err != nil {
		return result, fmt.Errorf("persist accepted turn: %w", err)
	}
	return result, nil
}

func completedTurn(sessionID string, count int, messages []ai.Message) gaictx.Turn {
	turn := gaictx.Turn{ID: fmt.Sprintf("%s/turn-%d", sessionID, count), Count: count}
	// Primary.Messages already includes the accepted user message. Do not add
	// input text separately or persist AttemptedTokens/AttemptedText.
	for i, message := range messages {
		stored := gaictx.StoredMessage{
			SchemaVersion: gaictx.MessageSchemaVersion,
			ID:            fmt.Sprintf("%s/message-%d", turn.ID, i),
			SessionID:     sessionID, TurnID: turn.ID, Message: message.Clone(),
		}
		if i == 0 && message.Role == ai.RoleUser {
			turn.UserMessage = &stored
		} else {
			turn.Messages = append(turn.Messages, stored)
		}
	}
	return turn
}
