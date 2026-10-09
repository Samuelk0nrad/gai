package history_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/history"
)

type historyModel struct {
	calls   atomic.Int32
	respond func(context.Context, ai.AIRequest) (string, error)
}

func (m *historyModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)
	m.calls.Add(1)
	go func() {
		defer close(out)
		text, err := m.respond(ctx, req)
		if err != nil {
			ai.SendToken(ctx, out, ai.Token{Err: err})
			return
		}
		part := ai.ContentPart{Kind: ai.ContentText, Text: text}
		ai.SendToken(ctx, out, ai.Token{Part: &part})
	}()
	return out
}
func summaryModel() *historyModel {
	return &historyModel{respond: func(context.Context, ai.AIRequest) (string, error) { return "short summary", nil }}
}
func oldHistory() *history.HistoryState {
	return &history.HistoryState{Turns: []gaictx.Turn{{ID: "old", Count: 1, UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, strings.Repeat("old text ", 100))}}}}
}
func newCompactor(t *testing.T, store history.HistoryStore, model ai.Model) *history.Compactor {
	t.Helper()
	c, err := history.NewCompactor("session", store, history.CompactorDefinition{Model: model, Amount: 1})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

type compactionObservations struct {
	mu     sync.Mutex
	events []gai.Observation
}

func (o *compactionObservations) Emit(_ context.Context, event gai.Observation) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.events = append(o.events, event)
}

func (o *compactionObservations) named(name string) []gai.Observation {
	o.mu.Lock()
	defer o.mu.Unlock()
	var events []gai.Observation
	for _, event := range o.events {
		if event.Name == name {
			events = append(events, event)
		}
	}
	return events
}

func requireCompactionObservation(t *testing.T, observations *compactionObservations, name string, wantErr error, fields map[string]any) {
	t.Helper()
	events := observations.named(name)
	if len(events) != 1 {
		t.Fatalf("%s observations = %d, want 1", name, len(events))
	}
	event := events[0]
	if event.Source != "context:Compactor" || event.Err != nil {
		t.Fatalf("%s source/raw error = %s/%v", name, event.Source, event.Err)
	}
	if wantErr != nil && (event.Fields["outcome"] != "error" || event.Fields["error_type"] != fmt.Sprintf("%T", wantErr)) {
		t.Fatalf("%s error metadata = %+v", name, event.Fields)
	}
	for key, want := range fields {
		if got := event.Fields[key]; got != want {
			t.Fatalf("%s field %s = %v, want %v", name, key, got, want)
		}
	}
}

func TestCASConcurrentWritersHaveExactlyOneWinner(t *testing.T) {
	for _, initialized := range []bool{false, true} {
		t.Run(fmt.Sprint(initialized), func(t *testing.T) {
			store := &sharedHistoryStore{}
			if initialized {
				store.state = oldHistory()
			}
			ready := make(chan struct{}, 2)
			release := make(chan struct{})
			results := make(chan error, 2)
			for i := range 2 {
				go func() {
					loaded, err := store.LoadHistory(t.Context(), "session")
					if err != nil {
						results <- err
						return
					}
					ready <- struct{}{}
					<-release
					_, err = store.CompareAndSwapHistory(t.Context(), "session", loaded.Revision, &history.HistoryState{Turns: []gaictx.Turn{{ID: fmt.Sprint(i)}}})
					results <- err
				}()
			}
			<-ready
			<-ready
			close(release)
			successes, conflicts := 0, 0
			for range 2 {
				err := <-results
				var conflict *history.RevisionConflictError
				if err == nil {
					successes++
				} else if errors.As(err, &conflict) {
					conflicts++
					if conflict.Expected == conflict.Actual || conflict.SessionID != "session" {
						t.Fatalf("bad conflict: %+v", conflict)
					}
				} else {
					t.Fatal(err)
				}
			}
			if successes != 1 || conflicts != 1 || len(store.saved) != 1 {
				t.Fatalf("successes=%d conflicts=%d saves=%d", successes, conflicts, len(store.saved))
			}
		})
	}
}

func TestStoreSnapshotsAndAcceptedWritesAreDetached(t *testing.T) {
	store := &sharedHistoryStore{}
	input := oldHistory()
	rev, err := store.CompareAndSwapHistory(t.Context(), "session", "", input)
	if err != nil || rev == "" {
		t.Fatalf("create: %q %v", rev, err)
	}
	input.Turns[0].UserMessage.Message.Parts[0].Text = "changed input"
	first, _ := store.LoadHistory(t.Context(), "session")
	first.State.Turns[0].UserMessage.Message.Parts[0].Text = "changed snapshot"
	second, _ := store.LoadHistory(t.Context(), "session")
	if second.Revision != rev || second.State.Turns[0].UserMessage.Message.Text() != oldHistory().Turns[0].UserMessage.Message.Text() {
		t.Fatal("storage aliases caller memory")
	}
	if _, err := store.CompareAndSwapHistory(t.Context(), "session", "", input); err == nil {
		t.Fatal("empty revision became unconditional overwrite")
	}
	if _, err := store.CompareAndSwapHistory(t.Context(), "session", rev, nil); !errors.Is(err, history.ErrHistoryStateRequired) {
		t.Fatal("nil state accepted")
	}
	cleared, err := store.CompareAndSwapHistory(t.Context(), "session", rev, &history.HistoryState{})
	if err != nil || cleared == rev || cleared == "" {
		t.Fatalf("clear reused revision: %q %v", cleared, err)
	}
	if _, err := store.CompareAndSwapHistory(t.Context(), "session", rev, input); err == nil {
		t.Fatal("stale state resurrected cleared history")
	}
}

func TestCompactorNoopAndReadOnlyBuild(t *testing.T) {
	for _, test := range []struct {
		name     string
		state    *history.HistoryState
		budget   int
		pressure bool
	}{
		{"missing", nil, 1, false},
		{"empty", &history.HistoryState{}, 1, false},
		{"fits", oldHistory(), 10000, false},
		{"summary-only pressure", &history.HistoryState{Summary: history.NewSummary("s", "old", "old", 1, 1, ai.ContentPart{Kind: ai.ContentText, Text: "large summary"})}, 1, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			store := &sharedHistoryStore{state: test.state}
			model := summaryModel()
			initial, _ := store.LoadHistory(t.Context(), "session")
			result, err := newCompactor(t, store, model).Compact(t.Context(), test.budget)
			if err != nil || result.Changed || result.Revision != initial.Revision || result.PressureRemaining != test.pressure {
				t.Fatalf("no-op = %+v %v", result, err)
			}
			source := history.NewHistory("session", store)
			source.SetTokenCounter(ai.TextTokenEstimator{})
			if _, err := source.Function(t.Context(), 1); err != nil {
				t.Fatal(err)
			}
			if store.casCalls != 0 || model.calls.Load() != 0 {
				t.Fatal("no-op/read-only build wrote or generated")
			}
		})
	}
}

func TestCompactorReportsCommittedRevisionAndRemainingPressure(t *testing.T) {
	for _, budget := range []int{0, 100} {
		store := &sharedHistoryStore{state: oldHistory()}
		initial, _ := store.LoadHistory(t.Context(), "session")
		observations := &compactionObservations{}
		c, err := history.NewCompactor("session", store, history.CompactorDefinition{Model: summaryModel(), Amount: 1, ObservationSink: observations})
		if err != nil {
			t.Fatal(err)
		}
		result, err := c.Compact(t.Context(), budget)
		if err != nil || !result.Changed || result.Revision == initial.Revision || result.PressureRemaining != (budget == 0) {
			t.Fatalf("compact = %+v %v", result, err)
		}
		stored, _ := store.LoadHistory(t.Context(), "session")
		if stored.Revision != result.Revision || stored.State.Summary == nil || len(stored.State.Turns) != 0 || store.casCalls != 1 {
			t.Fatal("commit/result mismatch")
		}
		pressure := observations.named("history_compactor_token_budget_reached")
		wantPressureEvents := 1
		if result.PressureRemaining {
			wantPressureEvents = 2
		}
		if len(pressure) != wantPressureEvents || pressure[0].Fields["last_turn_id"] != "old" {
			t.Fatalf("input/candidate pressure observations = %+v, want %d", pressure, wantPressureEvents)
		}
		if !result.PressureRemaining {
			requireCompactionObservation(t, observations, "history_compactor_summary_included", nil, map[string]any{"summary_start_turn": "old", "summary_end_turn": "old"})
		}
	}
}

func TestCompactionRejectsConcurrentAppend(t *testing.T) {
	store := &sharedHistoryStore{state: oldHistory()}
	started, release := make(chan struct{}), make(chan struct{})
	model := &historyModel{respond: func(ctx context.Context, _ ai.AIRequest) (string, error) {
		close(started)
		select {
		case <-release:
			return "short summary", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}}
	errorsOut := make(chan error, 1)
	observations := &compactionObservations{}
	c, err := history.NewCompactor("session", store, history.CompactorDefinition{Model: model, Amount: 1, ObservationSink: observations})
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		result, err := c.Compact(t.Context(), 20)
		if result != (history.CompactionResult{}) {
			errorsOut <- fmt.Errorf("failed compaction returned committed result: %+v", result)
			return
		}
		errorsOut <- err
	}()
	<-started
	loaded, _ := store.LoadHistory(t.Context(), "session")
	loaded.State.Turns = append(loaded.State.Turns, gaictx.Turn{ID: "new", Count: 2})
	winner, err := store.CompareAndSwapHistory(t.Context(), "session", loaded.Revision, loaded.State)
	if err != nil {
		t.Fatal(err)
	}
	close(release)
	err = <-errorsOut
	var conflict *history.RevisionConflictError
	if !errors.As(err, &conflict) || conflict.Actual != winner {
		t.Fatalf("conflict = %v", err)
	}
	stored, _ := store.LoadHistory(t.Context(), "session")
	if stored.State.Summary != nil || len(stored.State.Turns) != 2 || stored.State.Turns[1].ID != "new" || model.calls.Load() != 1 {
		t.Fatal("append lost or compaction retried")
	}
	requireCompactionObservation(t, observations, "history_compactor_conflict", err, nil)
}

func TestConcurrentCompactorsDoNotRetry(t *testing.T) {
	store := &sharedHistoryStore{state: oldHistory()}
	ready, release := make(chan struct{}, 2), make(chan struct{})
	model := &historyModel{respond: func(ctx context.Context, _ ai.AIRequest) (string, error) {
		ready <- struct{}{}
		select {
		case <-release:
			return "summary", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}}
	c := newCompactor(t, store, model)
	results := make(chan error, 2)
	for range 2 {
		go func() { _, err := c.Compact(t.Context(), 20); results <- err }()
	}
	<-ready
	<-ready
	close(release)
	success, conflicts := 0, 0
	for range 2 {
		err := <-results
		var conflict *history.RevisionConflictError
		if err == nil {
			success++
		} else if errors.As(err, &conflict) {
			conflicts++
		} else {
			t.Fatal(err)
		}
	}
	if success != 1 || conflicts != 1 || model.calls.Load() != 2 || len(store.saved) != 1 {
		t.Fatal("concurrent compactors overwrote or retried")
	}
}

type compactionCounter struct {
	ai.TextTokenEstimator
	count func(context.Context, string) (int, error)
}

func (c compactionCounter) CountTokens(ctx context.Context, text string) (int, error) {
	return c.count(ctx, text)
}

type failingStore struct {
	history.HistoryStore
	loadErr, saveErr error
}

func (s failingStore) LoadHistory(ctx context.Context, id string) (history.HistorySnapshot, error) {
	if s.loadErr != nil {
		return history.HistorySnapshot{}, s.loadErr
	}
	return s.HistoryStore.LoadHistory(ctx, id)
}
func (s failingStore) CompareAndSwapHistory(ctx context.Context, id string, expected history.Revision, next *history.HistoryState) (history.Revision, error) {
	if s.saveErr != nil {
		return "", s.saveErr
	}
	return s.HistoryStore.CompareAndSwapHistory(ctx, id, expected, next)
}

func TestCompactionFailuresDoNotPersistCandidates(t *testing.T) {
	failure := errors.New("injected failure")
	for _, stage := range []string{"load", "invalid snapshot", "model", "count input", "count input summary", "count candidate", "cancel before commit", "save"} {
		t.Run(stage, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			store := &sharedHistoryStore{state: oldHistory()}
			model := summaryModel()
			wrapper := failingStore{HistoryStore: store}
			observations := &compactionObservations{}
			def := history.CompactorDefinition{Amount: 1, Model: model, ObservationSink: observations}
			want := failure
			wantEvent := ""
			wantFields := map[string]any{"session_id": "session", "counter_id": (ai.TextTokenEstimator{}).ID()}
			switch stage {
			case "load":
				wrapper.loadErr = failure
				wantEvent = "history_compactor_state_load_failed"
			case "invalid snapshot":
				wrapper.HistoryStore = invalidSnapshotStore{store}
				want = history.ErrInvalidHistorySnapshot
				wantEvent = "history_compactor_state_load_failed"
			case "model":
				model.respond = func(context.Context, ai.AIRequest) (string, error) { return "", failure }
			case "save":
				wrapper.saveErr = failure
			default:
				switch stage {
				case "count input":
					wantEvent = "history_compactor_turn_tokenize_failed"
					wantFields["turn_id"], wantFields["turn_count"] = "old", 1
				case "count input summary", "count candidate":
					wantEvent = "history_compactor_summary_token_count_failed"
					wantFields["summary_start_turn"], wantFields["summary_end_turn"] = "old", "old"
					if stage == "count input summary" {
						store.state.Summary = history.NewSummary("summary", "old", "old", 1, 1, ai.ContentPart{Kind: ai.ContentText, Text: "existing summary"})
					}
				}
				if stage == "cancel before commit" {
					want = context.Canceled
				}
				def.TokenCounter = compactionCounter{count: func(ctx context.Context, text string) (int, error) {
					if stage == "count input" || stage == "count input summary" {
						return 0, failure
					}
					if strings.Contains(text, "Conversation summary:") {
						if stage == "count candidate" {
							return 0, failure
						}
						cancel()
						return 1, nil
					}
					return (ai.TextTokenEstimator{}).CountTokens(ctx, text)
				}}
			}
			before := store.state.Clone()
			c, err := history.NewCompactor("session", wrapper, def)
			if err != nil {
				t.Fatal(err)
			}
			result, err := c.Compact(ctx, 20)
			if !errors.Is(err, want) || result != (history.CompactionResult{}) {
				t.Fatalf("result=%+v error=%v want=%v", result, err, want)
			}
			if !reflect.DeepEqual(store.state, before) || len(store.saved) != 0 {
				t.Fatal("failed compaction persisted candidate")
			}
			requireCompactionObservation(t, observations, "history_compactor_finished", err, nil)
			if wantEvent != "" {
				requireCompactionObservation(t, observations, wantEvent, err, wantFields)
			}
		})
	}
}

func TestCompactionRejectsUnsupportedContentWithoutModelOrWrite(t *testing.T) {
	for _, kind := range []string{"media", "opaque message", "opaque summary"} {
		t.Run(kind, func(t *testing.T) {
			state := oldHistory()
			ext := ai.Extension{Namespace: "provider", Type: "state", Data: []byte(`"opaque"`), Required: true}
			switch kind {
			case "media":
				state.Turns[0].UserMessage.Message.Parts = append(state.Turns[0].UserMessage.Message.Parts, ai.ContentPart{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", URI: "https://example.test/image.png"}})
			case "opaque message":
				state.Turns[0].UserMessage.Message.Extensions = []ai.Extension{ext}
			case "opaque summary":
				state.Summary = history.NewSummary("summary", "t0", "t0", 0, 0, ai.ContentPart{Kind: ai.ContentText, Text: "earlier", Extensions: []ai.Extension{ext}})
			}
			before := state.Clone()
			store, model := &sharedHistoryStore{state: state}, summaryModel()
			_, err := newCompactor(t, store, model).Compact(t.Context(), 1)
			if !errors.Is(err, ai.ErrUnsupportedCapability) {
				t.Fatalf("unsupported content error = %v", err)
			}
			if model.calls.Load() != 0 || store.casCalls != 0 || !reflect.DeepEqual(state, before) {
				t.Fatal("unsupported content was changed/sent")
			}
		})
	}
}

func TestCompactorConfigurationAndCancellation(t *testing.T) {
	store := &sharedHistoryStore{}
	for _, def := range []history.CompactorDefinition{{}, {Model: summaryModel(), Amount: -1}, {Model: summaryModel(), Amount: 2}, {Model: summaryModel(), Amount: float32(math.NaN())}, {Model: summaryModel(), SummaryMaxTokens: -1}} {
		if _, err := history.NewCompactor("s", store, def); err == nil {
			t.Fatalf("invalid definition accepted: %+v", def)
		}
	}
	c := newCompactor(t, store, summaryModel())
	if _, err := c.Compact(t.Context(), -1); !errors.Is(err, history.ErrInvalidHistoryBudget) {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := c.Compact(ctx, 10); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := (*history.Compactor)(nil).Compact(t.Context(), 10); !errors.Is(err, history.ErrCompactorNil) {
		t.Fatal(err)
	}
}

type snapshotReader struct{ snapshot history.HistorySnapshot }

func (s snapshotReader) LoadHistory(context.Context, string) (history.HistorySnapshot, error) {
	return history.HistorySnapshot{Revision: s.snapshot.Revision, State: s.snapshot.State.Clone()}, nil
}

type invalidSnapshotStore struct{ history.HistoryStore }

func (invalidSnapshotStore) LoadHistory(context.Context, string) (history.HistorySnapshot, error) {
	return history.HistorySnapshot{State: oldHistory()}, nil
}

func TestReaderOnlySourceAndInvalidSnapshot(t *testing.T) {
	for _, invalid := range []bool{false, true} {
		snapshot := history.HistorySnapshot{Revision: "stable", State: oldHistory()}
		if invalid {
			snapshot.Revision = ""
		}
		source := history.NewHistory("session", snapshotReader{snapshot: snapshot})
		source.SetTokenCounter(ai.TextTokenEstimator{})
		_, err := source.Function(t.Context(), 1)
		if invalid && !errors.Is(err, history.ErrInvalidHistorySnapshot) {
			t.Fatalf("invalid snapshot error = %v", err)
		}
		if !invalid && err != nil {
			t.Fatal(err)
		}
	}
	store := &sharedHistoryStore{}
	_, err := newCompactor(t, invalidSnapshotStore{store}, summaryModel()).Compact(t.Context(), 1)
	if !errors.Is(err, history.ErrInvalidHistorySnapshot) || store.casCalls != 0 {
		t.Fatal("compacted unversioned snapshot")
	}
}

func TestTombstoneRetainsRevisionIdentity(t *testing.T) {
	store := &sharedHistoryStore{revision: 2}
	result, err := newCompactor(t, store, summaryModel()).Compact(t.Context(), 1)
	if err != nil || result.Changed || result.Revision != "2" || store.casCalls != 0 {
		t.Fatalf("tombstone compaction = %+v %v", result, err)
	}
	for _, stale := range []history.Revision{"", "1"} {
		if _, err := store.CompareAndSwapHistory(t.Context(), "session", stale, oldHistory()); err == nil {
			t.Fatal("stale write recreated deleted history")
		}
	}
	rev, err := store.CompareAndSwapHistory(t.Context(), "session", result.Revision, oldHistory())
	if err != nil || rev == result.Revision || rev == "" {
		t.Fatalf("recreate = %q %v", rev, err)
	}
}
