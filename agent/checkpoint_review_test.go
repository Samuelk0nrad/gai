package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/modelinput"
)

func TestDurableJSONOutputSurvivesRestoration(t *testing.T) {
	const raw = `{"answer":42}`
	for _, boundary := range []string{"pending", "completed primary", "completed middleware"} {
		t.Run(boundary, func(t *testing.T) {
			model := &durableModel{responses: [][]ai.Token{{{Part: &ai.ContentPart{Kind: ai.ContentJSON, JSON: json.RawMessage(raw)}}}}}
			def := durableDefinition(model)
			if boundary == "completed middleware" {
				def = durableDefinition(&durableModel{responses: [][]ai.Token{response("main")}})
				def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(agent.New(durableDefinition(model)), agent.AgentMiddlewareConfig{ID: "json", Output: agent.ReplaceOutput})}
			}
			store := &checkpointStore{}
			w := newDurableRun(t, agent.New(def), store)
			store.hook = func(cp agent.Checkpoint) error {
				if boundary == "pending" && pending(cp) != nil {
					return w.RequestPause()
				}
				if boundary != "pending" && cp.Status == agent.StatusSucceeded {
					return errStore
				}
				return nil
			}
			_, err := w.Run(t.Context())
			if boundary == "pending" && err != nil || boundary != "pending" && !errors.Is(err, errStore) {
				t.Fatal(err)
			}
			store.hook = nil
			w, err = agent.New(def).Resume(t.Context(), store.load(), agent.ResumeInput{}, store.durability())
			if err != nil {
				t.Fatal(err)
			}
			events := collectAgentEvents(w.RunEvents(t.Context()))
			result, err := w.Wait()
			if err != nil || result.Text != raw || model.count() != 1 || len(result.Output) != 1 || result.Output[0].Text != raw {
				t.Fatal(err, result, model.count())
			}
			wantLive := ""
			if boundary == "pending" {
				wantLive = raw
			}
			if outputText(events) != wantLive {
				t.Fatalf("live output = %q, want %q", outputText(events), wantLive)
			}
		})
	}
}

func TestDurablePropagatedMiddlewareSetupFailureCanRecover(t *testing.T) {
	for _, boundary := range []string{"map input", "child prompt"} {
		t.Run(boundary, func(t *testing.T) {
			var recovered atomic.Bool
			var maps, predicates atomic.Int32
			primary := &durableModel{responses: [][]ai.Token{response("main")}}
			child := &durableModel{responses: [][]ai.Token{response("child")}}
			childDef := durableDefinition(child)
			childDef.Prompt = func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
				if boundary == "child prompt" && !recovered.Load() {
					return nil, errStore
				}
				return modelinput.New(modelinput.Definition{SystemInstructions: []modelinput.Part{modelinput.NewTextPart("child instructions")}}), nil
			}
			def := durableDefinition(primary)
			def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(agent.New(childDef), agent.AgentMiddlewareConfig{
				ID: "child", Output: agent.AppendOutput,
				ShouldRun: func(agent.WorkflowResult) bool { predicates.Add(1); return true },
				MapInput: func(context.Context, agent.WorkflowResult) (agent.RunInput, error) {
					maps.Add(1)
					if boundary == "map input" && !recovered.Load() {
						return agent.RunInput{}, errStore
					}
					return agent.RunInput{}, nil
				},
			})}
			store := &checkpointStore{}
			_, err := newDurableRun(t, agent.New(def), store).Run(t.Context())
			if !errors.Is(err, errStore) || child.count() != 0 {
				t.Fatal(err, child.count())
			}
			cp := store.load()
			if cp.Stages[1].Complete || cp.ActiveStageID != "child" {
				t.Fatal("setup failure closed the stage", cp)
			}
			recovered.Store(true)
			w, err := agent.New(def).Resume(t.Context(), cp, agent.ResumeInput{}, store.durability())
			if err != nil {
				t.Fatal(err)
			}
			result := runDurable(t, w)
			if result.Text != "mainchild" || primary.count() != 1 || child.count() != 1 || maps.Load() != 2 || predicates.Load() != 1 || len(result.Errors) != 0 || len(result.Checkpoint.Stages[1].Errors) != 0 {
				t.Fatal(result, primary.count(), child.count(), maps.Load(), predicates.Load())
			}
		})
	}
}

func TestDurableResumeRestorationFailureDoesNotClaim(t *testing.T) {
	for _, invalid := range []string{"missing continuation", "invalid model"} {
		t.Run(invalid, func(t *testing.T) {
			def := durableDefinition(&durableModel{})
			store := &checkpointStore{}
			w := newDurableRun(t, agent.New(def), store)
			if err := w.RequestPause(); err != nil {
				t.Fatal(err)
			}
			runDurable(t, w)
			cp := store.load()
			if invalid == "missing continuation" {
				cp.Status = agent.StatusFailed
				cp.Stages[0].Continuation = nil
				store.saved = cp.Clone()
			} else {
				def.Model = nil
			}
			if _, err := agent.New(def).Resume(t.Context(), cp, agent.ResumeInput{}, store.durability()); err == nil {
				t.Fatal("invalid restore succeeded")
			}
			if !reflect.DeepEqual(cp, store.load()) {
				t.Fatal("failed restore mutated storage")
			}
		})
	}
}

func TestDurableResumeValidatesSavedChildBeforeClaim(t *testing.T) {
	child := durableDefinition(&durableModel{})
	def := durableDefinition(&durableModel{responses: [][]ai.Token{response("main")}})
	def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(agent.New(child), agent.AgentMiddlewareConfig{ID: "child"})}
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	store.hook = func(cp agent.Checkpoint) error {
		if cp.ActiveStageID == "child" && cp.Stages[1].Continuation != nil {
			return w.RequestPause()
		}
		return nil
	}
	runDurable(t, w)
	cp := store.load()
	store.hook = nil
	child.Model = nil
	def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(agent.New(child), agent.AgentMiddlewareConfig{ID: "child"})}
	if _, err := agent.New(def).Resume(t.Context(), cp, agent.ResumeInput{}, store.durability()); !errors.Is(err, loop.ErrModelNotConfigured) {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(cp, store.load()) {
		t.Fatal("invalid child claimed checkpoint")
	}
}

type cancellationModel struct{ ready chan struct{} }

func (*cancellationModel) Descriptor() ai.ModelDescriptor { return ai.ModelDescriptor{} }
func (m *cancellationModel) GenerateStream(ctx context.Context, _ ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token)
	go func() {
		defer close(out)
		ai.SendToken(ctx, out, ai.Token{Completion: &ai.Completion{UsageReported: true, Usage: ai.Usage{InputTokens: 17, OutputTokens: 3}}})
		close(m.ready)
		<-ctx.Done()
	}()
	return out
}

func TestDurableCancellationPersistsKnownUsageAndTerminalStatus(t *testing.T) {
	for _, failSettlement := range []bool{false, true} {
		t.Run(map[bool]string{false: "saved", true: "save failed"}[failSettlement], func(t *testing.T) {
			type contextKey struct{}
			ctx, cancel := context.WithCancel(context.WithValue(t.Context(), contextKey{}, "trace"))
			defer cancel()
			model := &cancellationModel{ready: make(chan struct{})}
			store := &checkpointStore{}
			d := store.durability()
			d.Commit = func(commitCtx context.Context, revision uint64, next agent.Checkpoint) error {
				if commitCtx.Value(contextKey{}) != "trace" {
					t.Error("checkpoint lost context values")
				}
				if err := commitCtx.Err(); err != nil {
					return err
				}
				if failSettlement && next.Stages[0].Continuation != nil && len(next.Stages[0].Continuation.Usage) == 1 && next.Stages[0].Continuation.Usage[0].Reported {
					return errStore
				}
				return store.commit(commitCtx, revision, next)
			}
			w, err := agent.New(durableDefinition(model)).NewRun(ctx, agent.RunInput{Durability: &d})
			if err != nil {
				t.Fatal(err)
			}
			// Abandon the public stream after cancellation; Wait must still settle.
			_ = w.RunEvents(ctx)
			<-model.ready
			cancel()
			result, err := w.Wait()
			if failSettlement {
				if !errors.Is(err, errStore) || !errors.Is(err, loop.ErrCheckpointCommit) || result.Status != agent.StatusFailed {
					t.Fatal(err, result)
				}
				return
			}
			if !errors.Is(err, context.Canceled) || result.Status != agent.StatusCanceled || result.BilledUsage.InputTokens != 17 {
				t.Fatal(err, result)
			}
			cp := store.load()
			if cp.Status != agent.StatusCanceled || cp.Stages[0].Continuation.Usage[0].Usage.InputTokens != 17 {
				t.Fatal(cp)
			}
			if _, err := agent.New(durableDefinition(model)).Resume(t.Context(), cp, agent.ResumeInput{}, d); !errors.Is(err, agent.ErrWorkflowFinished) {
				t.Fatal(err)
			}
		})
	}
}

func TestDurablePauseBeforeHandlerPreservesPhysicalAttempt(t *testing.T) {
	var invocation loop.Invocation
	tool := durableTool(t, "write", func(ctx context.Context, _ ai.ToolCall) (string, error) {
		invocation, _ = loop.InvocationFromContext(ctx)
		return "saved", nil
	})
	def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("write"), response("done")}}, tool)
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	store.hook = func(cp agent.Checkpoint) error {
		if p := pending(cp); p != nil && p.Invocations[0].Disposition == loop.InvocationDispatched {
			return w.RequestPause()
		}
		return nil
	}
	runDurable(t, w)
	cp := store.load()
	inv := pending(cp).Invocations[0]
	if inv.Disposition != loop.InvocationPlanned || inv.Attempt != 0 || invocation.InvocationID != "" {
		t.Fatal("unstarted dispatch consumed attempt", inv, invocation)
	}
	store.hook = nil
	w, err := agent.New(def).Resume(t.Context(), cp, agent.ResumeInput{}, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	runDurable(t, w)
	if invocation.Attempt != 1 || invocation.InvocationID != inv.ID {
		t.Fatal(invocation)
	}
}

func TestDurableCancellationSavesStartedParallelToolResults(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	started := make(chan string, 2)
	tool := durableTool(t, "write", func(ctx context.Context, call ai.ToolCall) (string, error) {
		started <- call.ID
		<-ctx.Done()
		return "processed:" + call.ID, nil
	})
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(durableDefinition(&durableModel{responses: [][]ai.Token{calls("write", "write")}}, tool)), store)
	_ = w.RunEvents(ctx)
	<-started
	<-started
	cancel()
	result, err := w.Wait()
	if !errors.Is(err, context.Canceled) || result.Status != agent.StatusCanceled {
		t.Fatal(err, result)
	}
	cp := store.load()
	if cp.Status != agent.StatusCanceled || pending(cp) == nil {
		t.Fatal(cp)
	}
	for _, inv := range pending(cp).Invocations {
		if inv.Disposition != loop.InvocationCompleted || inv.Execution.State != loop.ToolSucceeded || inv.Result == nil || inv.Result.Text() != "processed:"+inv.Call.ID {
			t.Fatal("known parallel outcome lost", inv)
		}
	}
}

func TestDurableLoopReportsToolSettlementFailureDuringCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	started := make(chan struct{})
	tool := durableTool(t, "write", func(ctx context.Context, _ ai.ToolCall) (string, error) {
		close(started)
		<-ctx.Done()
		return "processed", nil
	})
	l := loop.New(&durableModel{responses: [][]ai.Token{calls("write")}}, []loop.Tool{tool}, modelinput.New(modelinput.Definition{SystemInstructions: []modelinput.Part{modelinput.NewTextPart("system")}}), nil)
	l.ToolTransport = loop.ToolTransportNative
	l.Continuation = &loop.ContinuationControl{RunID: "run", ExecutionID: "execution", StageID: "primary", Commit: func(commitCtx context.Context, c loop.Continuation) error {
		if err := commitCtx.Err(); err != nil {
			t.Error("settlement context canceled", err)
			return err
		}
		if c.Pending != nil && c.Pending.Invocations[0].Disposition == loop.InvocationCompleted {
			return errStore
		}
		return nil
	}}
	events := l.Run(ctx)
	<-started
	cancel()
	var terminal loop.Event
	for event := range events {
		if event.Type == loop.EventError || event.Type == loop.EventCanceled || event.Type == loop.EventDone {
			terminal = event
		}
	}
	if terminal.Type != loop.EventError || !errors.Is(terminal.Err, loop.ErrCheckpointCommit) || !errors.Is(terminal.Err, errStore) {
		t.Fatal("tool settlement failure hidden by cancellation", terminal)
	}
}
