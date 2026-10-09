package main

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/context/history"
)

func testService(model ai.Model) *chatService {
	return &chatService{store: &memoryStore{}, model: model, requestWindow: 2048}
}

type runOutcome struct {
	result agent.WorkflowResult
	err    error
}

func startRun(ctx context.Context, service *chatService, session, text string) <-chan runOutcome {
	done := make(chan runOutcome, 1)
	go func() { result, err := service.Run(ctx, session, text); done <- runOutcome{result, err} }()
	return done
}

// Done announces that the waiter reached a cancellation-aware blocking point
// while the first run still owns the session.
type waitingContext struct {
	context.Context
	waiting chan struct{}
	once    sync.Once
}

func (c *waitingContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.waiting) })
	return c.Context.Done()
}

func TestSessionOwnershipCoversGenerationAndNextRunSeesAcceptedTurn(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	releaseFirst := func() { releaseOnce.Do(func() { close(release) }) }
	defer releaseFirst()
	var calls atomic.Int32
	var secondRequest ai.AIRequest
	model := textModel(func(ctx context.Context, req ai.AIRequest) (string, error) {
		if calls.Add(1) == 1 {
			close(started)
			select {
			case <-release:
				return "first accepted answer", nil
			case <-ctx.Done():
				return "", ctx.Err()
			}
		}
		secondRequest = req.Copy()
		return "second answer", nil
	})
	service := testService(model)
	first := startRun(t.Context(), service, "same", "first question")
	<-started
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	waiting := &waitingContext{Context: ctx, waiting: make(chan struct{})}
	blocked := startRun(waiting, service, "same", "canceled question")
	<-waiting.waiting
	cancel()
	if out := <-blocked; !errors.Is(out.err, context.Canceled) {
		t.Fatalf("waiting error = %v", out.err)
	}
	successfulWaiter := &waitingContext{Context: t.Context(), waiting: make(chan struct{})}
	second := startRun(successfulWaiter, service, "same", "second question")
	<-successfulWaiter.waiting
	if calls.Load() != 1 {
		t.Fatal("second session run generated before the first committed")
	}
	releaseFirst()
	if out := <-first; out.err != nil {
		t.Fatal(out.err)
	}
	if out := <-second; out.err != nil {
		t.Fatal(out.err)
	}
	var users, assistants []string
	for _, message := range secondRequest.Messages {
		switch message.Role {
		case ai.RoleUser:
			users = append(users, message.Text())
		case ai.RoleAssistant:
			assistants = append(assistants, message.Text())
		}
	}
	if !reflect.DeepEqual(users, []string{"first question", "second question"}) || !reflect.DeepEqual(assistants, []string{"first accepted answer"}) {
		t.Fatalf("unexpected ordered context: %+v", secondRequest.Messages)
	}
	stored, err := service.store.LoadHistory(t.Context(), "same")
	if err != nil || stored.State == nil || len(stored.State.Turns) != 2 || calls.Load() != 2 {
		t.Fatal("accepted turns not persisted once")
	}
	for i, want := range []struct{ user, assistant string }{{"first question", "first accepted answer"}, {"second question", "second answer"}} {
		turn := stored.State.Turns[i]
		if turn.Count != i+1 || turn.UserMessage == nil || turn.UserMessage.Message.Text() != want.user || len(turn.Messages) != 1 || turn.Messages[0].Message.Text() != want.assistant {
			t.Fatalf("stored turn %d = %+v", i+1, turn)
		}
	}
	if len(service.locks.gates) != 0 {
		t.Fatal("idle/canceled session gate leaked")
	}
}

func TestDifferentSessionsGenerateConcurrently(t *testing.T) {
	ready, release := make(chan struct{}, 2), make(chan struct{})
	model := textModel(func(ctx context.Context, _ ai.AIRequest) (string, error) {
		ready <- struct{}{}
		select {
		case <-release:
			return "answer", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})
	service := testService(model)
	a, b := startRun(t.Context(), service, "a", "a"), startRun(t.Context(), service, "b", "b")
	<-ready
	<-ready
	close(release)
	for _, done := range []<-chan runOutcome{a, b} {
		if out := <-done; out.err != nil {
			t.Fatal(out.err)
		}
	}
	for _, id := range []string{"a", "b"} {
		saved, err := service.store.LoadHistory(t.Context(), id)
		if err != nil || saved.State == nil || len(saved.State.Turns) != 1 {
			t.Fatalf("session %s not persisted: %v", id, err)
		}
	}
}

func TestStaleAnswerIsReturnedWithConflictWithoutReplaying(t *testing.T) {
	var calls atomic.Int32
	store := &memoryStore{}
	service := &chatService{store: store, requestWindow: 2048}
	service.model = textModel(func(ctx context.Context, _ ai.AIRequest) (string, error) {
		calls.Add(1)
		other := &history.HistoryState{}
		other.Turns = append(other.Turns, completedTurn("same", 1, []ai.Message{ai.TextMessage(ai.RoleUser, "external writer")}))
		if _, err := store.CompareAndSwapHistory(ctx, "same", "", other); err != nil {
			return "", err
		}
		return "answer with already-performed effects", nil
	})
	result, err := service.Run(t.Context(), "same", "original question")
	var conflict *history.RevisionConflictError
	if !errors.As(err, &conflict) || result.Text != "answer with already-performed effects" || calls.Load() != 1 {
		t.Fatalf("conflict/output = %v %+v calls=%d", err, result, calls.Load())
	}
	saved, _ := store.LoadHistory(t.Context(), "same")
	if len(saved.State.Turns) != 1 || saved.State.Turns[0].UserMessage.Message.Text() != "external writer" {
		t.Fatal("stale answer overwritten or appended using a fresh revision")
	}
}

// A deliberately cheap counter would hide pressure if the service allowed its
// compactor to use a different counter from the prompt and request budget guard.
type cheapCounter struct{ ai.TextTokenEstimator }

func (cheapCounter) ID() string { return "test/cheap" }
func (cheapCounter) CountTokens(ctx context.Context, _ string) (int, error) {
	return 0, ctx.Err()
}

func TestExplicitCompactionBeforeRunAndPressurePolicy(t *testing.T) {
	for _, remains := range []bool{false, true} {
		t.Run(map[bool]string{false: "fits", true: "pressure remains"}[remains], func(t *testing.T) {
			var summaries, answers atomic.Int32
			store := &memoryStore{}
			state := &history.HistoryState{}
			state.Turns = append(state.Turns, completedTurn("s", 7, []ai.Message{ai.TextMessage(ai.RoleUser, strings.Repeat("old context ", 200))}))
			if _, err := store.CompareAndSwapHistory(t.Context(), "s", "", state); err != nil {
				t.Fatal(err)
			}
			service := &chatService{store: store, requestWindow: 256,
				model: textModel(func(_ context.Context, req ai.AIRequest) (string, error) {
					answers.Add(1)
					if len(req.Messages) == 0 || req.Messages[0].Text() != "Conversation summary:\nshort summary" {
						t.Errorf("summary missing from request: %+v", req.Messages)
					}
					return "new answer", nil
				}),
				compaction: &history.CompactorDefinition{Amount: 1, TokenCounter: cheapCounter{}, Model: textModel(func(context.Context, ai.AIRequest) (string, error) {
					summaries.Add(1)
					if remains {
						return strings.Repeat("large summary ", 200), nil
					}
					return "short summary", nil
				})},
			}
			_, err := service.Run(t.Context(), "s", "new question")
			if remains {
				if !errors.Is(err, errHistoryPressure) || answers.Load() != 0 {
					t.Fatal("remaining pressure did not stop answer")
				}
			} else if err != nil {
				t.Fatal(err)
			}
			saved, _ := store.LoadHistory(t.Context(), "s")
			if saved.State.Summary == nil || saved.State.Summary.EndTurnCount != 7 || summaries.Load() != 1 {
				t.Fatal("compaction missing or repeated")
			}
			if !remains && (len(saved.State.Turns) != 1 || saved.State.Turns[0].Count != 8) {
				t.Fatal("turn numbering restarted after compaction")
			}
		})
	}
}

func TestGenerationFailureDoesNotPersistAndReleasesSession(t *testing.T) {
	failure := errors.New("generation failed")
	var calls atomic.Int32
	service := testService(textModel(func(context.Context, ai.AIRequest) (string, error) {
		if calls.Add(1) == 1 {
			return "", failure
		}
		return "accepted", nil
	}))
	if _, err := service.Run(t.Context(), "s", "first"); !errors.Is(err, failure) {
		t.Fatal(err)
	}
	first, _ := service.store.LoadHistory(t.Context(), "s")
	if first.State != nil || first.Revision != "" {
		t.Fatal("failed attempt was persisted")
	}
	if _, err := service.Run(t.Context(), "s", "second"); err != nil {
		t.Fatal(err)
	}
}

func TestMemoryStoreIsolationAndCanonicalAcceptedTurn(t *testing.T) {
	store := &memoryStore{}
	messages := []ai.Message{ai.TextMessage(ai.RoleUser, "user"), {Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "call", Type: "function", Name: "tool", Args: []byte(`{"x":1}`)}}}}, {Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "call", Name: "tool", Parts: ai.TextParts("result")}}}}, ai.TextMessage(ai.RoleAssistant, "answer")}
	turn := completedTurn("s", 1, messages)
	state := &history.HistoryState{}
	state.Turns = append(state.Turns, turn)
	rev, err := store.CompareAndSwapHistory(t.Context(), "s", "", state)
	if err != nil {
		t.Fatal(err)
	}
	messages[1].Parts[0].ToolCall.Args[0] = '!'
	state.Turns[0].Messages[1].Message.Parts[0].ToolResult.Parts[0].Text = "changed"
	loaded, _ := store.LoadHistory(t.Context(), "s")
	loaded.State.Turns[0].UserMessage.Message.Parts[0].Text = "changed"
	again, _ := store.LoadHistory(t.Context(), "s")
	got := again.State.Turns[0]
	if again.Revision != rev || got.UserMessage.Message.Text() != "user" || len(got.Messages) != 3 || string(got.Messages[0].Message.Parts[0].ToolCall.Args) != `{"x":1}` || got.Messages[1].Message.Parts[0].ToolResult.Parts[0].Text != "result" {
		t.Fatal("accepted canonical payloads were duplicated/lost/aliased")
	}
	if _, err := store.CompareAndSwapHistory(t.Context(), "s", "", state); err == nil {
		t.Fatal("unconditional overwrite")
	}
}
