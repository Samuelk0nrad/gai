package agent_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/Samuelk0nrad/gai/agent"
	"github.com/Samuelk0nrad/gai/agent/summary"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/loop"
	"github.com/Samuelk0nrad/gai/modelinput"
	"github.com/Samuelk0nrad/gai/modelinput/history"
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
		Turns: []modelinput.Turn{{ID: "t1", Count: 1, UserMessage: &modelinput.StoredMessage{Message: ai.TextMessage(ai.RoleUser, text)}}},
	}}}
}

func autoText(text string) []ai.Token {
	return []ai.Token{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: text}}}
}

// autoHistoryDefinition uses the separate text summarizer, so these tests
// exercise orchestration without the continuation request budget.
func autoHistoryDefinition(model ai.Model, store history.HistoryReader) agent.Definition {
	return agent.Definition{Model: model, HistoryCompaction: &agent.HistoryCompactionPolicy{Summarizer: summary.New(model)}, TokenCounter: ai.TextTokenEstimator{},
		Prompt: func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
			return modelinput.New(modelinput.Definition{TokenBudget: 256, ContextSources: []modelinput.ContextSource{history.NewHistory("s", store)}}), nil
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

func TestAutoCompactHistoryFailureStopsOrContinues(t *testing.T) {
	failure := errors.New("failed summary")
	conflict := &history.RevisionConflictError{SessionID: "s", Expected: "1", Actual: "2"}
	for _, mode := range []agent.HistoryCompactionFailure{agent.HistoryCompactionStop, agent.HistoryCompactionContinue} {
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
			t.Run(fmt.Sprint(mode, "/", tc.name), func(t *testing.T) {
				store := autoHistory(strings.Repeat("old context ", 500))
				store.loadErr, store.commitErr = tc.loadErr, tc.commitErr
				model := &scriptedWorkflowModel{scripts: [][]ai.Token{tc.summary, autoText("answer")}}
				if tc.loadErr != nil {
					// The failed load leaves no summary call.
					model.scripts = [][]ai.Token{autoText("answer")}
				}
				def := autoHistoryDefinition(model, store)
				def.HistoryCompaction.OnFailure = mode
				workflow, err := agent.New(def).NewRun(t.Context(), textRunInput("question"))
				if err != nil {
					t.Fatal(err)
				}
				result, err := workflow.Run(t.Context())
				if mode == agent.HistoryCompactionStop {
					if !errors.Is(err, tc.want) || result.Text != "" || store.writes != tc.writes || len(model.Requests()) != tc.calls {
						t.Fatalf("run: err=%v result=%+v writes=%d calls=%d", err, result, store.writes, len(model.Requests()))
					}
				} else if tc.loadErr != nil {
					// Selection reloads after a continued compaction failure,
					// so a broken store still ends the run there.
					if !errors.Is(err, failure) || len(model.Requests()) != 0 {
						t.Fatalf("run: err=%v calls=%d", err, len(model.Requests()))
					}
				} else if err != nil || result.Text != "answer" || store.writes != tc.writes || len(model.Requests()) != tc.calls+1 {
					t.Fatalf("continue: err=%v result=%+v writes=%d calls=%d", err, result, store.writes, len(model.Requests()))
				}
				reports := result.Primary.HistoryCompaction
				if len(reports) != 1 || !errors.Is(reports[0].Err, tc.want) || reports[0].Continued != (mode == agent.HistoryCompactionContinue) || reports[0].Source != "history" {
					t.Fatalf("reports = %+v", reports)
				}
				if tc.want == conflict {
					var got *history.RevisionConflictError
					if !errors.As(reports[0].Err, &got) || store.snapshot.Revision != "1" {
						t.Fatal("conflict lost or storage overwritten")
					}
				}
			})
		}
	}
}

func TestHistoryCompactionEarlyFailuresAndCancellation(t *testing.T) {
	failure := errors.New("summary failed")
	early := history.CompactionResult{Early: true}
	for _, tc := range []struct {
		name      string
		mode      agent.HistoryCompactionFailure
		result    history.CompactionResult
		err       error
		continued bool
	}{
		// History that still fits never ends the run, even under Stop.
		{"early failure under stop", agent.HistoryCompactionStop, early, failure, true},
		{"failure over the allocation under stop", agent.HistoryCompactionStop, history.CompactionResult{}, failure, false},
		// A deadline the summarizer set for itself is a summary failure.
		{"summarizer deadline under continue", agent.HistoryCompactionContinue, history.CompactionResult{}, context.DeadlineExceeded, true},
		// Cancellation ends the run even while the run's context is live.
		{"summarizer cancellation under continue", agent.HistoryCompactionContinue, history.CompactionResult{}, context.Canceled, false},
		{"early cancellation", agent.HistoryCompactionStop, early, context.Canceled, false},
		{"early configuration error", agent.HistoryCompactionStop, early, history.ErrSummaryPrefixRequired, false},
		// Another writer changed the history, which may no longer fit.
		{"early conflict under stop", agent.HistoryCompactionStop, early, &history.RevisionConflictError{SessionID: "s", Expected: "1", Actual: "2"}, false},
		{"early conflict under continue", agent.HistoryCompactionContinue, early, &history.RevisionConflictError{SessionID: "s", Expected: "1", Actual: "2"}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			probe := &compactionProbe{result: tc.result, err: tc.err}
			model := &scriptedWorkflowModel{scripts: [][]ai.Token{autoText("answer")}}
			a := agent.New(agent.Definition{Model: model, TokenCounter: ai.TextTokenEstimator{},
				HistoryCompaction: &agent.HistoryCompactionPolicy{OnFailure: tc.mode, Threshold: .6},
				Prompt: func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
					return modelinput.New(modelinput.Definition{TokenBudget: 256, ContextSources: []modelinput.ContextSource{probe}}), nil
				}})
			workflow, err := a.NewRun(t.Context(), textRunInput("q"))
			if err != nil {
				t.Fatal(err)
			}
			result, err := workflow.Run(t.Context())
			if probe.threshold != .6 {
				t.Fatalf("threshold = %v", probe.threshold)
			}
			if tc.continued {
				if err != nil || result.Text != "answer" {
					t.Fatalf("run: %v %+v", err, result)
				}
			} else if !errors.Is(err, tc.err) || len(model.Requests()) != 0 {
				t.Fatalf("run: err=%v calls=%d", err, len(model.Requests()))
			}
			reports := result.Primary.HistoryCompaction
			if len(reports) != 1 || reports[0].Continued != tc.continued || !errors.Is(reports[0].Err, tc.err) || reports[0].Result.Early != tc.result.Early {
				t.Fatalf("reports = %+v", reports)
			}
		})
	}
}

func TestHistoryCompactionTimeoutFailsTheRunWithoutCanceling(t *testing.T) {
	probe := &compactionProbe{err: fmt.Errorf("summary call: %w", context.DeadlineExceeded)}
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{autoText("answer")}}
	a := agent.New(agent.Definition{Model: model, TokenCounter: ai.TextTokenEstimator{}, HistoryCompaction: &agent.HistoryCompactionPolicy{},
		Prompt: func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
			return modelinput.New(modelinput.Definition{TokenBudget: 256, ContextSources: []modelinput.ContextSource{probe}}), nil
		}})
	workflow, err := a.NewRun(t.Context(), textRunInput("q"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := workflow.Run(t.Context())
	// The summarizer's own deadline is a failure, not a cancellation.
	if !errors.Is(err, agent.ErrHistoryCompactionTimeout) || errors.Is(err, context.DeadlineExceeded) || result.Canceled || result.Primary.Canceled {
		t.Fatalf("err=%v canceled=%v", err, result.Canceled)
	}
	if reports := result.Primary.HistoryCompaction; len(reports) != 1 || !errors.Is(reports[0].Err, context.DeadlineExceeded) {
		t.Fatalf("reports = %+v", reports)
	}
}

func TestAutoCompactHistoryConfigurationErrorsStopUnderContinue(t *testing.T) {
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{autoText("answer")}}
	def := autoHistoryDefinition(model, localCountHistoryStore{})
	def.HistoryCompaction.OnFailure = agent.HistoryCompactionContinue
	workflow, err := agent.New(def).NewRun(t.Context(), textRunInput("question"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = workflow.Run(t.Context()); !errors.Is(err, history.ErrHistoryStoreRequired) || len(model.Requests()) != 0 {
		t.Fatalf("err=%v calls=%d", err, len(model.Requests()))
	}
}

func TestHistoryCompactionPolicyValidation(t *testing.T) {
	nan := float32(math.NaN())
	for _, policy := range []agent.HistoryCompactionPolicy{
		{OnFailure: 9}, {Amount: -1}, {Amount: 2}, {Amount: nan}, {SummaryMaxTokens: -1}, {ResponseFormat: 9},
		{Threshold: -.1}, {Threshold: 1.5}, {Threshold: nan},
		{Summarizer: (*history.ContinuationSummarizer)(nil)},
	} {
		def := autoHistoryDefinition(&scriptedWorkflowModel{}, autoHistory("x"))
		def.HistoryCompaction = &policy
		if _, err := agent.New(def).NewRun(t.Context(), textRunInput("q")); !errors.Is(err, agent.ErrInvalidExecutionConfig) {
			t.Fatalf("policy %+v: %v", policy, err)
		}
	}
}

func TestAutoCompactHistoryOptInAndDisabledBudgets(t *testing.T) {
	for _, tc := range []struct {
		name           string
		enabled        bool
		override       bool
		disabledBudget bool
	}{
		{name: "default off"},
		{name: "override off", enabled: true, override: true},
		{name: "disabled window", enabled: true, disabledBudget: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store := autoHistory(strings.Repeat("history ", 500))
			model := &scriptedWorkflowModel{scripts: [][]ai.Token{autoText("answer")}}
			def := autoHistoryDefinition(model, store)
			if !tc.enabled {
				def.HistoryCompaction = nil
			}
			input := textRunInput("question")
			input.Execution = &agent.ExecutionOverrides{HistoryCompaction: agent.Optional[*agent.HistoryCompactionPolicy]{Set: tc.override}}
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
	def.Prompt = func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
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
	builder := modelinput.New(modelinput.Definition{TokenBudget: 30, ContextSources: []modelinput.ContextSource{history.NewHistory("s", store)}})
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
	prefix             ai.AIRequest
	budget             ai.RequestBudgetConfig
	threshold          float32
	// result and err replace the default unchanged result when set.
	result history.CompactionResult
	err    error
}

func (*compactionProbe) Name() string { return "history" }
func (p *compactionProbe) Function(_ context.Context, budget int) (modelinput.Part, error) {
	p.selected = append(p.selected, budget)
	return nil, nil
}
func (p *compactionProbe) CompactHistory(ctx context.Context, req history.AutoCompaction) (history.CompactionResult, error) {
	p.prepared = append(p.prepared, req.HistoryBudget)
	p.model, p.counter, p.budget, p.threshold = req.Model, req.Counter, req.RequestBudget, req.Threshold
	prefix, err := req.Prefix(ctx)
	if err != nil {
		return history.CompactionResult{}, err
	}
	p.prefix = prefix
	if p.err != nil || p.result != (history.CompactionResult{}) {
		return p.result, p.err
	}
	return history.CompactionResult{Revision: "r"}, nil
}

type compactionPrefix struct{ calls int }

func (*compactionPrefix) Name() string { return "retrieval" }
func (p *compactionPrefix) Function(context.Context, int) (modelinput.Part, error) {
	p.calls++
	return modelinput.NewTextPart("retrieved context"), nil
}

type scaledCompactionCounter struct{ ai.TextTokenEstimator }

func (scaledCompactionCounter) ID() string { return "test/double-estimate" }
func (scaledCompactionCounter) CountTokens(ctx context.Context, text string) (int, error) {
	count, err := (ai.TextTokenEstimator{}).CountTokens(ctx, text)
	return 2 * count, err
}

func TestAutoCompactHistoryUsesEffectiveRenderedAllocation(t *testing.T) {
	for _, native := range []bool{false, true} {
		for _, renderer := range []modelinput.Renderer{&modelinput.XMLRenderer{}, &modelinput.SimpleRenderer{}} {
			t.Run(fmt.Sprintf("native=%v/%T", native, renderer), func(t *testing.T) {
				probe, prefix := &compactionProbe{}, &compactionPrefix{}
				recorder := &scriptedWorkflowModel{scripts: [][]ai.Token{autoText("answer")}}
				var model ai.Model = recorder
				if native {
					model = nativeToolWorkflowModel{recorder}
				}
				counter := scaledCompactionCounter{}
				def := agent.Definition{Model: &scriptedWorkflowModel{}, Prompt: func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
					return modelinput.New(modelinput.Definition{TokenBudget: 10, Renderer: renderer,
						SystemInstructions: []modelinput.Part{modelinput.NewTextPart("system instructions")},
						ContextSources:     []modelinput.ContextSource{prefix, probe}}), nil
				}}
				enabled, output := &agent.HistoryCompactionPolicy{}, 40
				budget := &ai.RequestBudgetConfig{Limit: 9000, InputLimit: 8000, OutputReserve: 20, SafetyMargin: 7}
				input := textRunInput("question")
				input.Prompt.Context = []modelinput.Part{modelinput.NewTextPart("fixed context")}
				input.Execution = &agent.ExecutionOverrides{HistoryCompaction: agent.Optional[*agent.HistoryCompactionPolicy]{Set: true, Value: enabled}, Model: model, Tools: []loop.Tool{loop.NewEchoTool()},
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
				enabled.OnFailure = 9
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
				// The summary prefix is the answer request up to the history
				// source, with the same settings.
				wantPrefix := request.Copy()
				wantPrefix.Messages = wantPrefix.Messages[:len(probe.prefix.Messages)]
				if !reflect.DeepEqual(probe.prefix, wantPrefix) || len(probe.prefix.Messages) == 0 || !strings.Contains(requestText(probe.prefix), "retrieved context") || probe.budget != *budget {
					t.Fatalf("prefix = %+v\nwant %+v", probe.prefix, wantPrefix)
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
			a := agent.New(agent.Definition{Model: model, HistoryCompaction: &agent.HistoryCompactionPolicy{}, TokenCounter: ai.TextTokenEstimator{},
				Prompt: func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
					return modelinput.New(modelinput.Definition{TokenBudget: window, ContextSources: []modelinput.ContextSource{probe}}), nil
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
	builder := modelinput.New(modelinput.Definition{TokenBudget: 8, ContextSources: []modelinput.ContextSource{prefix, history.NewHistory("s", store)}})
	def := autoHistoryDefinition(model, store)
	def.Prompt = func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) { return builder, nil }
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

func conversationHistory(turns int) *autoHistoryStore {
	state := &history.HistoryState{SchemaVersion: history.HistorySchemaVersion}
	for i := 1; i <= turns; i++ {
		user := modelinput.StoredMessage{Message: ai.TextMessage(ai.RoleUser, fmt.Sprintf("Question %d about the deployment plan and its rollout order.", i))}
		state.Turns = append(state.Turns, modelinput.Turn{ID: fmt.Sprint("t", i), Count: i, UserMessage: &user, Messages: []modelinput.StoredMessage{
			{Message: ai.TextMessage(ai.RoleAssistant, fmt.Sprintf("Answer %d: %s", i, strings.Repeat("detail ", 30)))},
		}})
	}
	return &autoHistoryStore{snapshot: history.HistorySnapshot{Revision: "1", State: state}}
}

func TestHistoryCompactionContinuesTheRunPromptByDefault(t *testing.T) {
	store := conversationHistory(3)
	recorder := &scriptedWorkflowModel{scripts: [][]ai.Token{autoText("short summary"), autoText("answer")}}
	model := nativeToolWorkflowModel{recorder}
	// The summary limit must fit the small test window.
	def := agent.Definition{Model: model, HistoryCompaction: &agent.HistoryCompactionPolicy{SummaryMaxTokens: 100}, TokenCounter: ai.TextTokenEstimator{},
		Tools: []loop.Tool{loop.NewEchoTool()},
		Prompt: func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
			return modelinput.New(modelinput.Definition{TokenBudget: 1000,
				SystemInstructions: []modelinput.Part{modelinput.NewTextPart("system instructions")},
				ContextSources:     []modelinput.ContextSource{history.NewHistory("s", store)}}), nil
		},
	}
	// A long question leaves too little room for the stored history, while
	// the summary request, which does not contain it, still fits.
	workflow, err := agent.New(def).NewRun(t.Context(), textRunInput(strings.Repeat("word ", 640)))
	if err != nil {
		t.Fatal(err)
	}
	result, err := workflow.Run(t.Context())
	if err != nil || result.Text != "answer" {
		t.Fatalf("run: %v %+v", err, result)
	}
	requests := recorder.Requests()
	if len(requests) != 2 {
		t.Fatalf("requests = %d", len(requests))
	}
	summaryRequest, answer := requests[0], requests[1]
	// System prompt, three stored turns and one instruction.
	if len(summaryRequest.Messages) != 8 || summaryRequest.Messages[0].Role != ai.RoleSystem {
		t.Fatalf("summary request = %+v", summaryRequest.Messages)
	}
	if !reflect.DeepEqual(summaryRequest.Messages[0], answer.Messages[0]) || !reflect.DeepEqual(summaryRequest.Tools, answer.Tools) || len(answer.Tools) == 0 ||
		!reflect.DeepEqual(summaryRequest.ToolChoice, answer.ToolChoice) || !reflect.DeepEqual(summaryRequest.Reasoning, answer.Reasoning) {
		t.Fatal("summary request settings differ from the run's")
	}
	instruction := summaryRequest.Messages[7].Text()
	if !strings.Contains(instruction, `"Question 3 about the deployment plan`) {
		t.Fatalf("instruction = %q", instruction)
	}
	if !strings.Contains(requestText(answer), "Conversation summary:\nshort summary") {
		t.Fatal("answer did not use the committed summary")
	}
	reports := result.Primary.HistoryCompaction
	if len(reports) != 1 || !reports[0].Result.Changed || reports[0].Result.Revision != "2" || reports[0].Result.SummarizedTurns != 2 || reports[0].Err != nil {
		t.Fatalf("reports = %+v", reports)
	}
}

func TestTranscriptContinuesIntoEndOfRunCompaction(t *testing.T) {
	store := conversationHistory(3)
	recorder := &scriptedWorkflowModel{scripts: [][]ai.Token{
		{{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "c1", Type: "function", Name: "echo", Args: []byte(`{"text":"hello"}`)}}}},
		autoText("final answer"),
		autoText("end of run summary"),
	}}
	model := nativeToolWorkflowModel{recorder}
	def := agent.Definition{Model: model, TokenCounter: ai.TextTokenEstimator{}, Tools: []loop.Tool{loop.NewEchoTool()},
		Prompt: func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
			return modelinput.New(modelinput.Definition{TokenBudget: 4000,
				SystemInstructions: []modelinput.Part{modelinput.NewTextPart("system instructions")},
				ContextSources:     []modelinput.ContextSource{history.NewHistory("s", store)}}), nil
		},
	}
	workflow, err := agent.New(def).NewRun(t.Context(), textRunInput("Question 4: what changed?"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := workflow.Run(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	transcript := result.Primary.Transcript
	last := recorder.Requests()[1]
	if transcript == nil || len(transcript.Messages) != len(last.Messages)+1 ||
		!reflect.DeepEqual(transcript.Messages[:len(last.Messages)], last.Messages) ||
		transcript.Messages[len(last.Messages)].Text() != "final answer" || !reflect.DeepEqual(transcript.Tools, last.Tools) {
		t.Fatalf("transcript = %+v", transcript)
	}

	// The application persists the accepted turn, then compacts from the
	// run's transcript against the revision it committed.
	next := store.snapshot.State.Clone()
	turn := modelinput.Turn{ID: "t4", Count: 4}
	for i, message := range result.Primary.Messages {
		stored := modelinput.StoredMessage{Message: message}
		if i == 0 {
			turn.UserMessage = &stored
			continue
		}
		turn.Messages = append(turn.Messages, stored)
	}
	next.Turns = append(next.Turns, turn)
	revision, err := store.CompareAndSwapHistory(t.Context(), "s", "1", next)
	if err != nil {
		t.Fatal(err)
	}
	compactor, err := history.NewCompactor("s", store, history.CompactorDefinition{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	compacted, err := compactor.Compact(t.Context(), 50, history.WithPrefix(transcript), history.WithExpectedRevision(revision))
	if err != nil || !compacted.Changed {
		t.Fatalf("compact: %+v %v", compacted, err)
	}
	summaryRequest := recorder.Requests()[2]
	if len(summaryRequest.Messages) != len(transcript.Messages)+1 || !reflect.DeepEqual(summaryRequest.Messages[:len(transcript.Messages)], transcript.Messages) ||
		!reflect.DeepEqual(summaryRequest.Tools, transcript.Tools) {
		t.Fatal("end-of-run summary does not continue the transcript")
	}
	if !strings.Contains(summaryRequest.Messages[len(summaryRequest.Messages)-1].Text(), `"Question 3 about the deployment plan`) {
		t.Fatalf("instruction = %q", summaryRequest.Messages[len(summaryRequest.Messages)-1].Text())
	}
	if got := store.snapshot.State; got.Summary == nil || got.Summary.Content.Text != "end of run summary" || len(got.Turns) != 2 || got.Turns[0].ID != "t3" || got.Turns[1].ID != "t4" {
		t.Fatalf("stored = %+v", got)
	}
}

func TestTranscriptWithoutRequestBudget(t *testing.T) {
	recorder := &scriptedWorkflowModel{scripts: [][]ai.Token{
		{{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "c1", Type: "function", Name: "echo", Args: []byte(`{"text":"hello"}`)}}}},
		autoText("final answer"),
	}}
	def := agent.Definition{Model: nativeToolWorkflowModel{recorder}, Tools: []loop.Tool{loop.NewEchoTool()},
		Prompt: func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
			return modelinput.New(modelinput.Definition{SystemInstructions: []modelinput.Part{modelinput.NewTextPart("system instructions")}}), nil
		},
	}
	workflow, err := agent.New(def).NewRun(t.Context(), textRunInput("question"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := workflow.Run(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	last := recorder.Requests()[1]
	transcript := result.Primary.Transcript
	if transcript == nil || len(transcript.Messages) != len(last.Messages)+1 || !reflect.DeepEqual(transcript.Messages[:len(last.Messages)], last.Messages) ||
		transcript.Messages[0].Role != ai.RoleSystem || transcript.Messages[len(last.Messages)].Text() != "final answer" {
		t.Fatalf("transcript = %+v", transcript)
	}
}
