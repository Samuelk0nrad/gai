package agent_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/history"
	"github.com/lace-ai/gai/loop"
)

type autoHistoryStore struct {
	mu                 sync.Mutex
	snapshot           history.HistorySnapshot
	loads, writes      int
	loadErr, commitErr error
}

func (s *autoHistoryStore) LoadHistory(ctx context.Context, _ string) (history.HistorySnapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.loads++
	if err := ctx.Err(); err != nil {
		return history.HistorySnapshot{}, err
	}
	return history.HistorySnapshot{Revision: s.snapshot.Revision, State: s.snapshot.State.Clone()}, s.loadErr
}

func (s *autoHistoryStore) CompareAndSwapHistory(ctx context.Context, id string, expected history.Revision, next *history.HistoryState) (history.Revision, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return "", err
	}
	s.writes++
	if s.commitErr != nil {
		return "", s.commitErr
	}
	if expected != s.snapshot.Revision {
		return "", &history.RevisionConflictError{SessionID: id, Expected: expected, Actual: s.snapshot.Revision}
	}
	s.snapshot = history.HistorySnapshot{Revision: history.Revision(fmt.Sprint(s.writes + 1)), State: next.Clone()}
	return s.snapshot.Revision, nil
}

func autoHistory(text string) *autoHistoryStore {
	return &autoHistoryStore{snapshot: history.HistorySnapshot{Revision: "1", State: &history.HistoryState{
		Turns: []gaictx.Turn{{ID: "t1", Count: 1, UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, text)}}},
	}}}
}

func autoText(text string) []ai.Token {
	return []ai.Token{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: text}}}
}

func autoHistoryDefinition(model ai.Model, store history.HistoryReader) agent.Definition {
	return agent.Definition{Model: model, AutoCompactHistory: true, TokenCounter: ai.TextTokenEstimator{},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{TokenBudget: 256, ContextSources: []gaictx.ContextSource{history.NewHistory("s", store)}}), nil
		},
	}
}

func TestAutoCompactHistoryOnlyUnderPressureAndOnlyAtRunStart(t *testing.T) {
	for _, pressured := range []bool{false, true} {
		t.Run(fmt.Sprint(pressured), func(t *testing.T) {
			text := "short prior question"
			scripts := [][]ai.Token{autoText("answer")}
			if pressured {
				text = strings.Repeat("old context ", 500)
				scripts = [][]ai.Token{autoText("short summary"), autoText("answer")}
			}
			store := autoHistory(text)
			model := &scriptedWorkflowModel{scripts: scripts}
			workflow, err := agent.New(autoHistoryDefinition(model, store)).NewRun(t.Context(), textRunInput("question"))
			if err != nil {
				t.Fatal(err)
			}
			if len(model.Requests()) != 0 || store.loads != 0 || store.writes != 0 {
				t.Fatal("NewRun performed compaction")
			}
			result, err := workflow.Run(t.Context())
			if err != nil || result.Text != "answer" {
				t.Fatalf("run: %v, %+v", err, result)
			}
			wantWrites := 0
			if pressured {
				wantWrites = 1
			}
			if store.writes != wantWrites || len(model.Requests()) != wantWrites+1 {
				t.Fatal("unexpected compaction/generation count")
			}
			request := model.Requests()[wantWrites]
			if pressured && !strings.Contains(requestText(request), "Conversation summary:\nshort summary") {
				t.Fatal("answer did not use committed summary")
			}
			if !pressured && !strings.Contains(requestText(request), text) {
				t.Fatal("fitting history lost")
			}
		})
	}
}

func TestAutoCompactHistoryFailureStopsBeforeAnswer(t *testing.T) {
	failure := errors.New("failed summary")
	conflict := &history.RevisionConflictError{SessionID: "s", Expected: "1", Actual: "2"}
	for _, tc := range []struct {
		name                     string
		loadErr, commitErr, want error
		summary                  []ai.Token
		writes, calls            int
	}{
		{"load", failure, nil, failure, nil, 0, 0},
		{"summary", nil, nil, failure, []ai.Token{{Err: failure}}, 0, 1},
		{"conflict", nil, conflict, conflict, autoText("short"), 1, 1},
		{"remaining pressure", nil, nil, history.ErrHistoryPressureRemaining, autoText(strings.Repeat("huge summary ", 500)), 1, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store := autoHistory(strings.Repeat("old context ", 500))
			store.loadErr, store.commitErr = tc.loadErr, tc.commitErr
			model := &scriptedWorkflowModel{scripts: [][]ai.Token{tc.summary, autoText("must not answer")}}
			workflow, err := agent.New(autoHistoryDefinition(model, store)).NewRun(t.Context(), textRunInput("question"))
			if err != nil {
				t.Fatal(err)
			}
			result, err := workflow.Run(t.Context())
			if !errors.Is(err, tc.want) || result.Text != "" || store.writes != tc.writes || len(model.Requests()) != tc.calls {
				t.Fatalf("run: err=%v result=%+v writes=%d calls=%d", err, result, store.writes, len(model.Requests()))
			}
			if tc.want == conflict {
				var got *history.RevisionConflictError
				if !errors.As(err, &got) || store.snapshot.Revision != "1" {
					t.Fatal("conflict lost or storage overwritten")
				}
			}
		})
	}
}

func TestAutoCompactHistoryOptInAndDisabledBudgets(t *testing.T) {
	for _, tc := range []struct {
		name           string
		enabled        bool
		override       *bool
		disabledBudget bool
	}{
		{name: "default off"},
		{name: "override off", enabled: true, override: new(bool)},
		{name: "disabled window", enabled: true, disabledBudget: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store := autoHistory(strings.Repeat("history ", 500))
			model := &scriptedWorkflowModel{scripts: [][]ai.Token{autoText("answer")}}
			def := autoHistoryDefinition(model, store)
			def.AutoCompactHistory = tc.enabled
			input := textRunInput("question")
			input.Execution = &agent.ExecutionOverrides{AutoCompactHistory: tc.override}
			if tc.disabledBudget {
				input.Execution.RequestBudget = agent.Optional[*ai.RequestBudgetConfig]{Set: true, Value: &ai.RequestBudgetConfig{}}
			}
			workflow, err := agent.New(def).NewRun(t.Context(), input)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = workflow.Run(t.Context()); err != nil {
				t.Fatal(err)
			}
			if store.writes != 0 || store.loads != 1 || len(model.Requests()) != 1 {
				t.Fatal("disabled compaction performed work")
			}
		})
	}
}

func TestAutoCompactHistoryRejectsUnsupportedBuilderAndReader(t *testing.T) {
	model := &scriptedWorkflowModel{}
	def := autoHistoryDefinition(model, localCountHistoryStore{})
	workflow, err := agent.New(def).NewRun(t.Context(), textRunInput("question"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = workflow.Run(t.Context()); !errors.Is(err, history.ErrHistoryStoreRequired) {
		t.Fatal(err)
	}
	def.Prompt = func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
		return &inputOnlyPromptBuilder{}, nil
	}
	def.TokenCounter = nil
	if _, err = agent.New(def).NewRun(t.Context(), textRunInput("question")); !errors.Is(err, agent.ErrHistoryCompactionNotConfigurable) {
		t.Fatal(err)
	}
	if len(model.Requests()) != 0 {
		t.Fatal("model called despite unsupported configuration")
	}
}

func TestAutomaticCompactionDoesNotChangeOrdinaryBuilds(t *testing.T) {
	store := autoHistory(strings.Repeat("history ", 500))
	builder := gaictx.New(gaictx.Definition{TokenBudget: 30, ContextSources: []gaictx.ContextSource{history.NewHistory("s", store)}})
	for range 2 {
		if _, err := builder.BuildContext(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	if store.writes != 0 || store.loads != 2 {
		t.Fatal("ordinary build mutated storage")
	}
}

type compactionProbe struct {
	prepared, selected []int
	model              ai.Model
	counter            ai.TokenCounter
}

func (*compactionProbe) Name() string { return "history" }
func (p *compactionProbe) Function(_ context.Context, budget int) (gaictx.Part, error) {
	p.selected = append(p.selected, budget)
	return nil, nil
}
func (p *compactionProbe) CompactHistory(_ context.Context, budget int, model ai.Model, counter ai.TokenCounter) error {
	p.prepared = append(p.prepared, budget)
	p.model, p.counter = model, counter
	return nil
}

type compactionPrefix struct{ calls int }

func (*compactionPrefix) Name() string { return "retrieval" }
func (p *compactionPrefix) Function(context.Context, int) (gaictx.Part, error) {
	p.calls++
	return gaictx.NewTextPart("retrieved context"), nil
}

type scaledCompactionCounter struct{ ai.TextTokenEstimator }

func (scaledCompactionCounter) ID() string { return "test/double-estimate" }
func (scaledCompactionCounter) CountTokens(ctx context.Context, text string) (int, error) {
	count, err := (ai.TextTokenEstimator{}).CountTokens(ctx, text)
	return 2 * count, err
}

func TestAutoCompactHistoryUsesEffectiveRenderedAllocation(t *testing.T) {
	for _, native := range []bool{false, true} {
		for _, renderer := range []gaictx.Renderer{&gaictx.XMLRenderer{}, &gaictx.SimpleRenderer{}} {
			t.Run(fmt.Sprintf("native=%v/%T", native, renderer), func(t *testing.T) {
				probe, prefix := &compactionProbe{}, &compactionPrefix{}
				recorder := &scriptedWorkflowModel{scripts: [][]ai.Token{autoText("answer")}}
				var model ai.Model = recorder
				if native {
					model = nativeToolWorkflowModel{recorder}
				}
				counter := scaledCompactionCounter{}
				def := agent.Definition{Model: &scriptedWorkflowModel{}, Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
					return gaictx.New(gaictx.Definition{TokenBudget: 10, Renderer: renderer,
						SystemInstructions: []gaictx.Part{gaictx.NewTextPart("system instructions")},
						ContextSources:     []gaictx.ContextSource{prefix, probe}}), nil
				}}
				enabled, output := true, 40
				budget := &ai.RequestBudgetConfig{Limit: 9000, InputLimit: 8000, OutputReserve: 20, SafetyMargin: 7}
				input := textRunInput("question")
				input.Prompt.Context = []gaictx.Part{gaictx.NewTextPart("fixed context")}
				input.Execution = &agent.ExecutionOverrides{AutoCompactHistory: &enabled, Model: model, Tools: []loop.Tool{loop.NewEchoTool()},
					Limits:         agent.LimitsOverrides{MaxTokens: &output},
					TokenCounter:   agent.Optional[ai.TokenCounter]{Set: true, Value: counter},
					RequestBudget:  agent.Optional[*ai.RequestBudgetConfig]{Set: true, Value: budget},
					ResponseFormat: &ai.ResponseFormat{Type: ai.ResponseFormatJSONObject},
				}
				workflow, err := agent.New(def).NewRun(t.Context(), input)
				if err != nil {
					t.Fatal(err)
				}
				// NewRun must snapshot the override rather than retaining its pointer.
				enabled = false
				if _, err = workflow.Run(t.Context()); err != nil {
					t.Fatal(err)
				}
				request := recorder.Requests()[0]
				count, err := ai.EstimateRequestTokens(t.Context(), request, counter)
				if err != nil {
					t.Fatal(err)
				}
				want := min(budget.Limit-output, budget.InputLimit) - budget.SafetyMargin - count.InputTokens
				if len(probe.prepared) != 1 || probe.prepared[0] != want || len(probe.selected) != 1 || probe.selected[0] != want || prefix.calls != 1 || probe.model != model || probe.counter.ID() != counter.ID() {
					t.Fatalf("allocation=%v selection=%v want=%d prefixCalls=%d", probe.prepared, probe.selected, want, prefix.calls)
				}
			})
		}
	}
}

func TestAutoCompactHistoryZeroAllocationAndUnfixableInput(t *testing.T) {
	for _, window := range []int{7, 8} {
		t.Run(fmt.Sprint(window), func(t *testing.T) {
			probe := &compactionProbe{}
			model := &scriptedWorkflowModel{scripts: [][]ai.Token{autoText("answer")}}
			a := agent.New(agent.Definition{Model: model, AutoCompactHistory: true, TokenCounter: ai.TextTokenEstimator{},
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
					return gaictx.New(gaictx.Definition{TokenBudget: window, ContextSources: []gaictx.ContextSource{probe}}), nil
				}})
			workflow, err := a.NewRun(t.Context(), textRunInput("q"))
			if err != nil {
				t.Fatal(err)
			}
			_, err = workflow.Run(t.Context())
			if window == 7 {
				if !errors.Is(err, ai.ErrRequestBudgetExceeded) || len(probe.prepared) != 0 || len(model.Requests()) != 0 {
					t.Fatalf("unfixable input: %v %+v", err, probe)
				}
			} else if err != nil || len(probe.prepared) != 1 || probe.prepared[0] != 0 {
				t.Fatalf("zero allocation: %v %+v", err, probe)
			}
		})
	}
}

func TestAutoCompactHistoryStopsAfterOversizedPrecedingSource(t *testing.T) {
	store := autoHistory(strings.Repeat("history ", 500))
	model, prefix := &scriptedWorkflowModel{}, &compactionPrefix{}
	builder := gaictx.New(gaictx.Definition{TokenBudget: 8, ContextSources: []gaictx.ContextSource{prefix, history.NewHistory("s", store)}})
	def := autoHistoryDefinition(model, store)
	def.Prompt = func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return builder, nil }
	workflow, err := agent.New(def).NewRun(t.Context(), textRunInput("q"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = workflow.Run(t.Context()); !errors.Is(err, ai.ErrRequestBudgetExceeded) {
		t.Fatal(err)
	}
	if prefix.calls != 1 || store.loads != 0 || store.writes != 0 || len(model.Requests()) != 0 {
		t.Fatal("oversized earlier source did not stop before compaction")
	}
	// The explicit preparation guard must not change ordinary read-only builds.
	if _, err = builder.BuildContext(t.Context()); err != nil {
		t.Fatal(err)
	}
	if prefix.calls != 2 || store.loads != 1 || store.writes != 0 {
		t.Fatal("ordinary selection changed")
	}
}

func TestAutoCompactHistoryIsNotRepeatedByToolsOrRetries(t *testing.T) {
	store := autoHistory(strings.Repeat("history ", 2000))
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{
		autoText("summary"),
		{{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "c1", Type: "function", Name: "echo", Args: []byte(`{"text":"hello"}`)}}}},
		{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("retry")}}},
		autoText("answer"),
	}}
	def := autoHistoryDefinition(nativeToolWorkflowModel{model}, store)
	def.Tools = []loop.Tool{loop.NewEchoTool()}
	def.RequestBudget = &ai.RequestBudgetConfig{Limit: 1000}
	def.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}
	workflow, err := agent.New(def).NewRun(t.Context(), textRunInput("question"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := workflow.Run(t.Context())
	if err != nil || result.Text != "answer" || store.writes != 1 || len(model.Requests()) != 4 {
		t.Fatalf("run: %v result=%+v writes=%d calls=%d", err, result, store.writes, len(model.Requests()))
	}
}

func TestAutoCompactHistoryHonorsCanceledRun(t *testing.T) {
	store := autoHistory(strings.Repeat("history ", 500))
	model := &scriptedWorkflowModel{}
	workflow, err := agent.New(autoHistoryDefinition(model, store)).NewRun(t.Context(), textRunInput("question"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err = workflow.Run(ctx); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if store.writes != 0 || len(model.Requests()) != 0 {
		t.Fatal("canceled run performed compaction")
	}
}
