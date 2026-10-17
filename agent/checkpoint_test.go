package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/modelinput"
)

type durableModel struct {
	mu        sync.Mutex
	requests  []ai.AIRequest
	responses [][]ai.Token
	onRequest func(context.Context, int)
}

func (*durableModel) Descriptor() ai.ModelDescriptor {
	return ai.ModelDescriptor{NativeTools: ai.FeatureSupportSupported}
}
func (m *durableModel) GenerateStream(ctx context.Context, request ai.AIRequest) <-chan ai.Token {
	m.mu.Lock()
	index := len(m.requests)
	m.requests = append(m.requests, request.Copy())
	var tokens []ai.Token
	if index < len(m.responses) {
		tokens = m.responses[index]
	} else {
		tokens = []ai.Token{{Err: errors.New("unexpected model invocation")}}
	}
	m.mu.Unlock()
	out := make(chan ai.Token, len(tokens))
	go func() {
		defer close(out)
		if m.onRequest != nil {
			m.onRequest(ctx, index)
		}
		for _, token := range tokens {
			if !ai.SendToken(ctx, out, token) {
				return
			}
		}
	}()
	return out
}
func (m *durableModel) count() int { m.mu.Lock(); defer m.mu.Unlock(); return len(m.requests) }
func response(text string) []ai.Token {
	return []ai.Token{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: text}}}
}
func calls(names ...string) []ai.Token {
	var tokens []ai.Token
	for i, name := range names {
		call := ai.ToolCall{Type: "function", ID: string(rune('a' + i)), Name: name, Args: json.RawMessage(`{}`)}
		tokens = append(tokens, ai.Token{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &call}})
	}
	return tokens
}

type checkpointStore struct {
	mu    sync.Mutex
	saved agent.Checkpoint
	hook  func(agent.Checkpoint) error
}

var errStore = errors.New("storage unavailable")
var errStale = errors.New("stale revision")

func (s *checkpointStore) commit(ctx context.Context, revision uint64, next agent.Checkpoint) error {
	if s.hook != nil {
		if err := s.hook(next); err != nil {
			return err
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if revision != s.saved.Revision {
		return errStale
	}
	encoded, err := json.Marshal(next)
	if err != nil {
		return err
	}
	var restored agent.Checkpoint
	if err = json.Unmarshal(encoded, &restored); err != nil {
		return err
	}
	s.saved = restored
	return nil
}
func (s *checkpointStore) load() agent.Checkpoint {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.saved.Clone()
}
func (s *checkpointStore) durability() agent.Durability {
	return agent.Durability{DefinitionRevision: "test-v1", Commit: s.commit}
}
func durableDefinition(model ai.Model, tools ...loop.Tool) agent.Definition {
	return agent.Definition{Name: "main", Model: model, Tools: tools, Prompt: func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
		return modelinput.New(modelinput.Definition{SystemInstructions: []modelinput.Part{modelinput.NewTextPart("system")}}), nil
	}}
}
func durableTool(t *testing.T, name string, fn loop.ToolFunc) loop.Tool {
	t.Helper()
	tool, err := loop.NewTool(name, "test", ai.ToolParameters{}, fn)
	if err != nil {
		t.Fatal(err)
	}
	return tool
}
func newDurableRun(t *testing.T, a *agent.Agent, store *checkpointStore) *agent.Workflow {
	t.Helper()
	d := store.durability()
	w, err := a.NewRun(context.Background(), agent.RunInput{ID: "logical-run", Prompt: modelinput.PromptInput{User: ai.TextParts("original")}, Durability: &d})
	if err != nil {
		t.Fatal(err)
	}
	return w
}
func runDurable(t *testing.T, w *agent.Workflow) agent.WorkflowResult {
	t.Helper()
	result, err := w.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return result
}
func pending(cp agent.Checkpoint) *loop.Generation {
	for _, s := range cp.Stages {
		if s.ID == cp.ActiveStageID && s.Continuation != nil {
			return s.Continuation.Pending
		}
	}
	return nil
}

func TestDurablePausePlanAndResumeWithoutRegeneration(t *testing.T) {
	var effects atomic.Int32
	var identities []loop.Invocation
	tool := durableTool(t, "write", func(ctx context.Context, _ ai.ToolCall) (string, error) {
		inv, ok := loop.InvocationFromContext(ctx)
		if !ok {
			t.Error("missing invocation identity")
		}
		identities = append(identities, inv)
		effects.Add(1)
		return "saved", nil
	})
	firstTokens := append(calls("write"), ai.Token{Completion: &ai.Completion{UsageReported: true, Usage: ai.Usage{InputTokens: 17, OutputTokens: 3}}})
	firstModel := &durableModel{responses: [][]ai.Token{firstTokens}}
	var factoryCalls atomic.Int32
	def := durableDefinition(firstModel, tool)
	def.Prompt = func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
		factoryCalls.Add(1)
		return modelinput.New(modelinput.Definition{SystemInstructions: []modelinput.Part{modelinput.NewTextPart("frozen instructions")}}), nil
	}
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	store.hook = func(cp agent.Checkpoint) error {
		if pending(cp) != nil {
			return w.RequestPause()
		}
		return nil
	}
	result := runDurable(t, w)
	if result.Status != agent.StatusPaused || !result.Complete || result.Checkpoint == nil || effects.Load() != 0 {
		t.Fatalf("pause=%+v effects=%d", result, effects.Load())
	}
	if result.BilledUsage.InputTokens != 17 || result.BilledUsage.OutputTokens != 3 {
		t.Fatal("paused usage", result.BilledUsage)
	}
	if _, err := w.Run(context.Background()); !errors.Is(err, agent.ErrWorkflowAlreadyRun) {
		t.Fatal(err)
	}
	checkpoint := store.load()
	if err := checkpoint.Validate(); err != nil {
		t.Fatal(err)
	}
	id := pending(checkpoint).Invocations[0].ID
	nextModel := &durableModel{responses: [][]ai.Token{response("done")}}
	def.Model = nextModel
	def.Prompt = func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
		t.Error("prompt rebuilt on resume")
		return nil, errors.New("must not run")
	}
	store.hook = nil
	resumed, err := agent.New(def).Resume(context.Background(), checkpoint, agent.ResumeInput{}, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	claim := store.load()
	events := collectAgentEvents(resumed.RunEvents(context.Background()))
	final, err := resumed.Wait()
	if err != nil {
		t.Fatal(err)
	}
	if final.Status != agent.StatusSucceeded || final.Text != "done" || effects.Load() != 1 || firstModel.count() != 1 || nextModel.count() != 1 {
		t.Fatalf("result=%+v effects=%d models=%d/%d", final, effects.Load(), firstModel.count(), nextModel.count())
	}
	if final.BilledUsage.InputTokens != 17 || final.BilledUsage.OutputTokens != 3 {
		t.Fatal("resumed usage", final.BilledUsage)
	}
	if identities[0].InvocationID != id || identities[0].RunID != "logical-run" || identities[0].ExecutionID == checkpoint.ExecutionID {
		t.Fatalf("identity=%+v", identities)
	}
	if events[0].Type != agent.EventResumed {
		t.Fatalf("first event=%+v", events[0])
	}
	if !reflect.DeepEqual(events[0].Checkpoint, &claim) || events[0].CheckpointRevision != claim.Revision {
		t.Fatal("resumed snapshot advanced beyond the claim", events[0].Checkpoint)
	}
	for i, event := range events {
		if event.Sequence != uint64(i+1) || event.ExecutionID == checkpoint.ExecutionID || event.RunID != "logical-run" {
			t.Fatalf("event identity=%+v", event)
		}
	}
	if factoryCalls.Load() != 1 {
		t.Fatal("factory calls", factoryCalls.Load())
	}
	messages := nextModel.requests[0].Messages
	var inputCount int
	for _, m := range messages {
		if m.Text() == "original" {
			inputCount++
		}
	}
	if inputCount != 1 {
		t.Fatalf("duplicated input: %+v", messages)
	}
}

func TestDurableCommitFailuresFenceSideEffects(t *testing.T) {
	for _, boundary := range []string{"model", "plan", "dispatch", "result"} {
		t.Run(boundary, func(t *testing.T) {
			var effects atomic.Int32
			tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { effects.Add(1); return "effect", nil })
			model := &durableModel{responses: [][]ai.Token{calls("write"), response("done")}}
			def := durableDefinition(model, tool)
			store := &checkpointStore{}
			store.hook = func(cp agent.Checkpoint) error {
				p := pending(cp)
				s := cp.Stages[0].Continuation
				if s == nil {
					return nil
				}
				if boundary == "model" && s.Phase == loop.PhaseGenerating {
					return errStore
				}
				if p != nil {
					inv := p.Invocations[0]
					if boundary == "plan" && inv.Disposition == loop.InvocationPlanned || boundary == "dispatch" && inv.Disposition == loop.InvocationDispatched || boundary == "result" && inv.Disposition == loop.InvocationCompleted {
						return errStore
					}
				}
				return nil
			}
			result, err := newDurableRun(t, agent.New(def), store).Run(context.Background())
			if !errors.Is(err, loop.ErrCheckpointCommit) || result.Status != agent.StatusFailed {
				t.Fatalf("result=%+v err=%v", result, err)
			}
			expected := int32(0)
			if boundary == "result" {
				expected = 1
			}
			if effects.Load() != expected {
				t.Fatalf("effects=%d", effects.Load())
			}
			if boundary == "model" && model.count() != 0 {
				t.Fatal("model dispatched before commit")
			}
			if boundary == "result" {
				store.hook = nil
				_, err := agent.New(def).Resume(context.Background(), store.load(), agent.ResumeInput{}, store.durability())
				var recovery *loop.RecoveryRequiredError
				if !errors.As(err, &recovery) || len(recovery.Invocations) != 1 || effects.Load() != 1 {
					t.Fatalf("recovery=%v effects=%d", err, effects.Load())
				}
			}
		})
	}
}

func TestDurableRecoveryReusesCompletedSiblingAndVerifiedResult(t *testing.T) {
	var effects atomic.Int32
	tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { effects.Add(1); return "processed", nil })
	model := &durableModel{responses: [][]ai.Token{calls("write", "write")}}
	def := durableDefinition(model, tool)
	def.ToolExecution.MaxConcurrent = 1
	store := &checkpointStore{}
	store.hook = func(cp agent.Checkpoint) error {
		p := pending(cp)
		if p != nil && p.Invocations[1].Disposition == loop.InvocationCompleted {
			return errStore
		}
		return nil
	}
	_, err := newDurableRun(t, agent.New(def), store).Run(context.Background())
	if !errors.Is(err, loop.ErrCheckpointCommit) {
		t.Fatal(err)
	}
	cp := store.load()
	if pending(cp).Invocations[0].Disposition != loop.InvocationCompleted || pending(cp).Invocations[1].Disposition != loop.InvocationDispatched {
		t.Fatal("missing individual progress")
	}
	store.hook = nil
	def.Model = &durableModel{responses: [][]ai.Token{response("done")}}
	inv := pending(cp).Invocations[1]
	input := agent.ResumeInput{CommandID: "reconcile", ToolResolutions: []loop.ToolRecoveryResolution{{InvocationID: inv.ID, Action: loop.UseVerifiedResult, Result: &ai.ToolResult{ToolCallID: inv.Call.ID, Name: inv.Call.Name, Parts: ai.TextParts("verified")}, Execution: loop.ToolExecution{State: loop.ToolSucceeded, Output: loop.ToolOutputAccepted}}}}
	w, err := agent.New(def).Resume(context.Background(), cp, input, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	result := runDurable(t, w)
	if effects.Load() != 2 || result.Text != "done" {
		t.Fatalf("effects=%d result=%+v", effects.Load(), result)
	}
	var outputs []string
	for _, m := range result.Primary.Messages {
		for _, r := range m.ToolResults() {
			outputs = append(outputs, r.Text())
		}
	}
	if !reflect.DeepEqual(outputs, []string{"processed", "verified"}) {
		t.Fatal(outputs)
	}
}

func TestDurableSteeringCancelsAndDeduplicates(t *testing.T) {
	var effects atomic.Int32
	tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { effects.Add(1); return "effect", nil })
	def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("write", "write")}}, tool)
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	store.hook = func(cp agent.Checkpoint) error {
		if pending(cp) != nil {
			return w.RequestPause()
		}
		return nil
	}
	runDurable(t, w)
	cp := store.load()
	store.hook = nil
	input := agent.ResumeInput{CommandID: "steer", Prompt: modelinput.PromptInput{User: ai.TextParts("only today's stock")}, PendingTools: agent.CancelUnstartedTools}
	next := &durableModel{responses: [][]ai.Token{response("new plan")}}
	def.Model = next
	resumed, err := agent.New(def).Resume(context.Background(), cp, input, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	// Simulate worker loss just after accepting the command, before Run.
	accepted := store.load()
	_ = resumed
	resumed, err = agent.New(def).Resume(context.Background(), accepted, input, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	conflict := input
	conflict.Prompt.User = ai.TextParts("different")
	if _, err := agent.New(def).Resume(context.Background(), store.load(), conflict, store.durability()); !errors.Is(err, agent.ErrCommandConflict) {
		t.Fatal(err)
	}
	result := runDurable(t, resumed)
	if effects.Load() != 0 || result.Text != "new plan" {
		t.Fatal(result, effects.Load())
	}
	var aggregateSteering int
	for _, m := range result.Primary.Messages {
		if m.Text() == "only today's stock" {
			aggregateSteering++
		}
	}
	if aggregateSteering != 1 {
		t.Fatal("aggregate steering", aggregateSteering)
	}
	messages := next.requests[0].Messages
	var toolResults, steering int
	for _, m := range messages {
		toolResults += len(m.ToolResults())
		if m.Text() == "only today's stock" {
			steering++
			if toolResults != 2 {
				t.Fatal("steering before tool results")
			}
		}
	}
	if steering != 1 {
		t.Fatalf("steering count=%d %+v", steering, messages)
	}
}

func TestDurableDeferredApprovalAndCurrentPolicy(t *testing.T) {
	for _, revoke := range []bool{false, true} {
		t.Run(map[bool]string{false: "approve", true: "revoked"}[revoke], func(t *testing.T) {
			var effects atomic.Int32
			tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { effects.Add(1); return "done", nil })
			def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("write")}}, tool)
			def.ToolPolicy = loop.ToolPolicyFunc(func(context.Context, loop.ToolPolicyInput) (loop.ToolDecision, error) {
				return loop.ToolDecision{Action: loop.ToolRequireApproval}, nil
			})
			store := &checkpointStore{}
			durability := store.durability()
			durability.DeferredApprovals = true
			w, err := agent.New(def).NewRun(context.Background(), agent.RunInput{Durability: &durability})
			if err != nil {
				t.Fatal(err)
			}
			result := runDurable(t, w)
			if result.Status != agent.StatusPaused || effects.Load() != 0 {
				t.Fatal(result)
			}
			cp := store.load()
			request := pending(cp).Invocations[0].Approval
			if request == nil {
				t.Fatal("no persisted approval")
			}
			def.Model = &durableModel{responses: [][]ai.Token{response("finished")}}
			if revoke {
				def.ToolPolicy = loop.ToolPolicyFunc(func(context.Context, loop.ToolPolicyInput) (loop.ToolDecision, error) {
					return loop.ToolDecision{Action: loop.ToolDeny}, nil
				})
			}
			resumed, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{CommandID: "approve", Approvals: []loop.ToolApprovalDecision{{RequestID: request.ID, Approved: true}}}, durability)
			if err != nil {
				t.Fatal(err)
			}
			runDurable(t, resumed)
			want := int32(1)
			if revoke {
				want = 0
			}
			if effects.Load() != want {
				t.Fatal(effects.Load())
			}
		})
	}
}

func TestDurablePauseStopsQueuedToolsAndRetainsDeadline(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var w *agent.Workflow
		var effects atomic.Int32
		tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) {
			effects.Add(1)
			if err := w.RequestPause(); err != nil {
				t.Error(err)
			}
			return "one", nil
		})
		def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("write", "write")}}, tool)
		def.ToolExecution.MaxConcurrent = 1
		def.RetryPolicy = &loop.RetryPolicy{TotalTimeout: time.Minute}
		store := &checkpointStore{}
		w = newDurableRun(t, agent.New(def), store)
		result := runDurable(t, w)
		if result.Status != agent.StatusPaused || effects.Load() != 1 {
			t.Fatal(result, effects.Load())
		}
		cp := store.load()
		if pending(cp).Invocations[0].Disposition != loop.InvocationCompleted || pending(cp).Invocations[1].Disposition != loop.InvocationPlanned {
			t.Fatal("incorrect pause ledger")
		}
		time.Sleep(2 * time.Minute)
		resumed, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{}, store.durability())
		if err != nil {
			t.Fatal(err)
		}
		result, err = resumed.Run(context.Background())
		if !errors.Is(err, context.DeadlineExceeded) || effects.Load() != 1 {
			t.Fatalf("deadline result=%+v err=%v effects=%d", result, err, effects.Load())
		}
	})
}

func TestDurableCompatibilityAndStaleOwnership(t *testing.T) {
	model := &durableModel{responses: [][]ai.Token{response("done")}}
	def := durableDefinition(model)
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	if err := w.RequestPause(); err != nil {
		t.Fatal(err)
	}
	runDurable(t, w)
	cp := store.load()
	different := store.durability()
	different.DefinitionRevision = "v2"
	if _, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{}, different); !errors.Is(err, agent.ErrDefinitionRevision) {
		t.Fatal(err)
	}
	winner, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{}, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{}, store.durability()); !errors.Is(err, errStale) {
		t.Fatal(err)
	}
	runDurable(t, winner)
	if model.count() != 1 {
		t.Fatal(model.count())
	}
}

func TestDurableRejectsCustomMiddlewareBeforePrompt(t *testing.T) {
	def := durableDefinition(&durableModel{})
	def.Prompt = func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
		t.Error("prompt called")
		return nil, nil
	}
	def.Middleware = []agent.Middleware{agent.MiddlewareFunc(func(context.Context, *agent.MiddlewareContext, <-chan agent.Event) <-chan agent.Event { return nil })}
	store := &checkpointStore{}
	d := store.durability()
	if _, err := agent.New(def).NewRun(context.Background(), agent.RunInput{Durability: &d}); !errors.Is(err, agent.ErrUnsupportedDurableMiddleware) {
		t.Fatal(err)
	}
}

func TestDurableResumeRejectsNestedMiddlewareBeforeClaim(t *testing.T) {
	childDef := durableDefinition(&durableModel{responses: [][]ai.Token{response("child")}})
	def := durableDefinition(&durableModel{responses: [][]ai.Token{response("main")}})
	def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(agent.New(childDef), agent.AgentMiddlewareConfig{ID: "child"})}
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	if err := w.RequestPause(); err != nil {
		t.Fatal(err)
	}
	runDurable(t, w)
	cp := store.load()
	childDef.Middleware = []agent.Middleware{agent.NewAgentMiddleware(agent.New(durableDefinition(&durableModel{})), agent.AgentMiddlewareConfig{ID: "nested"})}
	def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(agent.New(childDef), agent.AgentMiddlewareConfig{ID: "child"})}
	if _, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{}, store.durability()); !errors.Is(err, agent.ErrMiddlewareAgentNested) {
		t.Fatal(err)
	}
	if store.load().Revision != cp.Revision {
		t.Fatal("invalid definition claimed checkpoint")
	}
}

func TestDurableMiddlewareResumesChildAndReusesCompletedStages(t *testing.T) {
	for _, policy := range []agent.OutputPolicy{agent.PreserveOutput, agent.AppendOutput, agent.ReplaceOutput} {
		t.Run(string(rune('0'+policy)), func(t *testing.T) {
			var primaryCalls, mapCalls, predicateCalls atomic.Int32
			primaryModel := &durableModel{responses: [][]ai.Token{response("main")}, onRequest: func(context.Context, int) { primaryCalls.Add(1) }}
			childModel := &durableModel{responses: [][]ai.Token{response("child")}}
			child := agent.New(durableDefinition(childModel))
			def := durableDefinition(primaryModel)
			def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(child, agent.AgentMiddlewareConfig{ID: "stable-child", Name: "child", Output: policy, ShouldRun: func(agent.WorkflowResult) bool { predicateCalls.Add(1); return true }, MapInput: func(context.Context, agent.WorkflowResult) (agent.RunInput, error) {
				mapCalls.Add(1)
				return agent.RunInput{}, nil
			}})}
			store := &checkpointStore{}
			w := newDurableRun(t, agent.New(def), store)
			store.hook = func(cp agent.Checkpoint) error {
				if cp.ActiveStageID == "stable-child" && cp.Stages[1].Continuation != nil {
					return w.RequestPause()
				}
				return nil
			}
			result := runDurable(t, w)
			if result.Status != agent.StatusPaused || childModel.count() != 0 {
				t.Fatal(result)
			}
			cp := store.load()
			store.hook = nil
			resumed, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{}, store.durability())
			if err != nil {
				t.Fatal(err)
			}
			// Fail after the child result/output decision is persisted, before terminal workflow status.
			store.hook = func(cp agent.Checkpoint) error {
				if cp.Status == agent.StatusSucceeded {
					return errStore
				}
				return nil
			}
			result, err = resumed.Run(context.Background())
			if !errors.Is(err, errStore) {
				t.Fatal(err)
			}
			completed := store.load()
			store.hook = nil
			resumed, err = agent.New(def).Resume(context.Background(), completed, agent.ResumeInput{}, store.durability())
			if err != nil {
				t.Fatal(err)
			}
			events := collectAgentEvents(resumed.RunEvents(context.Background()))
			result, err = resumed.Wait()
			if err != nil {
				t.Fatal(err)
			}
			want := "main"
			if policy == agent.AppendOutput {
				want = "mainchild"
			}
			if policy == agent.ReplaceOutput {
				want = "child"
			}
			if result.Text != want || primaryCalls.Load() != 1 || childModel.count() != 1 || mapCalls.Load() != 1 || predicateCalls.Load() != 1 {
				t.Fatalf("text=%q want=%q counts=%d/%d/%d/%d", result.Text, want, primaryCalls.Load(), childModel.count(), mapCalls.Load(), predicateCalls.Load())
			}
			if strings.TrimSpace(outputText(events)) != "" {
				t.Fatal("completed output redelivered", outputText(events))
			}
		})
	}
}

func TestDurableTerminalCancelSteersInsteadOfExecuting(t *testing.T) {
	var effects atomic.Int32
	tool := durableTool(t, "finish", func(context.Context, ai.ToolCall) (string, error) { effects.Add(1); return "side effect", nil })
	tool, err := loop.WithToolOptions(tool, loop.ToolOptions{Terminal: true})
	if err != nil {
		t.Fatal(err)
	}
	def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("finish")}}, tool)
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	store.hook = func(cp agent.Checkpoint) error {
		if pending(cp) != nil {
			return w.RequestPause()
		}
		return nil
	}
	runDurable(t, w)
	store.hook = nil
	def.Model = &durableModel{responses: [][]ai.Token{response("steered")}}
	resumed, err := agent.New(def).Resume(context.Background(), store.load(), agent.ResumeInput{CommandID: "cancel-terminal", PendingTools: agent.CancelUnstartedTools, Prompt: modelinput.PromptInput{User: ai.TextParts("different plan")}}, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	result := runDurable(t, resumed)
	if result.Text != "steered" || effects.Load() != 0 {
		t.Fatal(result, effects.Load())
	}
}

func TestDurableProcessorFailureKeepsEffectAndRedactsCheckpoint(t *testing.T) {
	var effects atomic.Int32
	tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { effects.Add(1); return "SECRET-raw-result", nil })
	def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("write")}}, tool)
	def.ToolResultProcessor = loop.ToolResultProcessorFunc(func(context.Context, loop.ToolPolicyInput, loop.ToolResult) (loop.ToolResult, error) {
		return loop.ToolResult{}, errors.New("cannot process SECRET-raw-result")
	})
	store := &checkpointStore{}
	_, err := newDurableRun(t, agent.New(def), store).Run(context.Background())
	if !errors.Is(err, loop.ErrToolResultProcess) {
		t.Fatal(err)
	}
	cp := store.load()
	encoded, _ := json.Marshal(cp)
	if strings.Contains(string(encoded), "SECRET") {
		t.Fatal("raw result leaked to checkpoint")
	}
	inv := pending(cp).Invocations[0]
	if inv.Execution.State != loop.ToolSucceeded || !inv.ProcessingFailed {
		t.Fatal(inv)
	}
	if _, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{}, store.durability()); !errors.Is(err, loop.ErrRecoveryRequired) {
		t.Fatal(err)
	}
	// Invalid verified content must be rejected without replacing the saved state.
	invalid := &ai.ToolResult{ToolCallID: inv.Call.ID, Name: inv.Call.Name, Parts: []ai.ContentPart{{Kind: ai.ContentMedia, Media: &ai.MediaPart{}}}}
	resolution := loop.ToolRecoveryResolution{InvocationID: inv.ID, Action: loop.UseVerifiedResult, Result: invalid, Execution: loop.ToolExecution{State: loop.ToolSucceeded, Output: loop.ToolOutputAccepted}}
	if _, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{CommandID: "invalid", ToolResolutions: []loop.ToolRecoveryResolution{resolution}}, store.durability()); err == nil {
		t.Fatal("accepted invalid canonical result")
	}
	if store.load().Revision != cp.Revision {
		t.Fatal("invalid resolution committed")
	}
	// A verified canonical result retains media and extension content.
	resolution.Result = &ai.ToolResult{ToolCallID: inv.Call.ID, Name: inv.Call.Name, Parts: []ai.ContentPart{{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", Data: []byte("image")}, Extensions: []ai.Extension{{Namespace: "test", Type: "opaque", Data: json.RawMessage(`"state"`)}}}}}
	next := &durableModel{responses: [][]ai.Token{response("reconciled")}}
	def.Model = next
	resumed, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{CommandID: "verified", ToolResolutions: []loop.ToolRecoveryResolution{resolution}}, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	runDurable(t, resumed)
	var got []ai.ToolResult
	for _, m := range next.requests[0].Messages {
		got = append(got, m.ToolResults()...)
	}
	if len(got) != 1 || !reflect.DeepEqual(got[0], *resolution.Result) || effects.Load() != 1 {
		t.Fatalf("result=%+v effects=%d", got, effects.Load())
	}
}

func TestDurableParallelSiblingAndExplicitReplayIdentity(t *testing.T) {
	var effects atomic.Int32
	var identities sync.Map
	release := make(chan struct{})
	var once sync.Once
	tool := durableTool(t, "write", func(ctx context.Context, call ai.ToolCall) (string, error) {
		inv, _ := loop.InvocationFromContext(ctx)
		identities.Store(call.ID, inv)
		effects.Add(1)
		if call.ID == "b" {
			<-release
		}
		return call.ID, nil
	})
	def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("write", "write")}}, tool)
	store := &checkpointStore{}
	store.hook = func(cp agent.Checkpoint) error {
		p := pending(cp)
		if p == nil {
			return nil
		}
		if p.Invocations[0].Disposition == loop.InvocationCompleted {
			once.Do(func() { close(release) })
		}
		if p.Invocations[1].Disposition == loop.InvocationCompleted {
			return errStore
		}
		return nil
	}
	_, err := newDurableRun(t, agent.New(def), store).Run(context.Background())
	if !errors.Is(err, errStore) {
		t.Fatal(err)
	}
	cp := store.load()
	inv := pending(cp).Invocations[1]
	if pending(cp).Invocations[0].Disposition != loop.InvocationCompleted {
		t.Fatal("lost sibling")
	}
	store.hook = nil
	def.Model = &durableModel{responses: [][]ai.Token{response("done")}}
	resolution := loop.ToolRecoveryResolution{InvocationID: inv.ID, Action: loop.ReplayInvocation}
	if _, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{CommandID: "unsafe", ToolResolutions: []loop.ToolRecoveryResolution{resolution}}, store.durability()); err == nil {
		t.Fatal("replay without explicit safety assertion")
	}
	resolution.SafeReplay = true
	resumed, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{CommandID: "safe", ToolResolutions: []loop.ToolRecoveryResolution{resolution}}, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	runDurable(t, resumed)
	raw, _ := identities.Load("b")
	replayed := raw.(loop.Invocation)
	if effects.Load() != 3 || replayed.InvocationID != inv.ID || replayed.Attempt != 2 {
		t.Fatalf("effects=%d replay=%+v", effects.Load(), replayed)
	}
}

func TestDurableInFlightModelFinishesBeforePause(t *testing.T) {
	var w *agent.Workflow
	var effects atomic.Int32
	tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { effects.Add(1); return "done", nil })
	model := &durableModel{responses: [][]ai.Token{calls("write")}, onRequest: func(context.Context, int) {
		if err := w.RequestPause(); err != nil {
			t.Error(err)
		}
	}}
	store := &checkpointStore{}
	w = newDurableRun(t, agent.New(durableDefinition(model, tool)), store)
	result := runDurable(t, w)
	if result.Status != agent.StatusPaused || pending(store.load()) == nil || effects.Load() != 0 {
		t.Fatal(result, effects.Load())
	}
}

func TestDurableDiscardBudgetAndKnownFailedUsage(t *testing.T) {
	tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { return "done", nil })
	t.Run("discard consumes iteration", func(t *testing.T) {
		model := &durableModel{responses: [][]ai.Token{response("no required tool")}}
		def := durableDefinition(model, tool)
		def.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"write"}}
		def.Limits.MaxLoopIterations = 1
		store := &checkpointStore{}
		w := newDurableRun(t, agent.New(def), store)
		store.hook = func(cp agent.Checkpoint) error {
			c := cp.Stages[0].Continuation
			if c != nil && c.NextIteration == 2 {
				return w.RequestPause()
			}
			return nil
		}
		// The iteration limit can win before the pause boundary; either way it must
		// remain consumed after restoration.
		_, _ = w.Run(context.Background())
		store.hook = nil
		cp := store.load()
		if cp.Stages[0].Continuation.NextIteration != 2 {
			t.Fatal(cp)
		}
		next := &durableModel{responses: [][]ai.Token{calls("write")}}
		def.Model = next
		resumed, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{}, store.durability())
		if err != nil {
			t.Fatal(err)
		}
		_, err = resumed.Run(context.Background())
		if !errors.Is(err, loop.ErrMaxIterations) || next.count() != 0 {
			t.Fatal(err, next.count())
		}
	})
	t.Run("terminal provider usage survives", func(t *testing.T) {
		tokens := []ai.Token{{Completion: &ai.Completion{UsageReported: true, Usage: ai.Usage{InputTokens: 17, OutputTokens: 3}}}, {Err: errors.New("provider stopped")}}
		def := durableDefinition(&durableModel{responses: [][]ai.Token{tokens}})
		def.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}
		store := &checkpointStore{}
		_, err := newDurableRun(t, agent.New(def), store).Run(context.Background())
		if err == nil {
			t.Fatal("expected model error")
		}
		cp := store.load()
		if len(cp.Stages[0].Continuation.Usage) != 1 || cp.Stages[0].Continuation.Usage[0].Usage.InputTokens != 17 {
			t.Fatal("known usage lost")
		}
		def.Model = &durableModel{responses: [][]ai.Token{response("recovered")}}
		resumed, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{}, store.durability())
		if err != nil {
			t.Fatal(err)
		}
		result := runDurable(t, resumed)
		if result.BilledUsage.InputTokens != 17 {
			t.Fatal(result.BilledUsage)
		}
	})
}

func TestDurableMiddlewareRecordErrorAndRestoredInput(t *testing.T) {
	var maps atomic.Int32
	child := agent.New(durableDefinition(&durableModel{responses: [][]ai.Token{response("post")}}))
	def := durableDefinition(&durableModel{responses: [][]ai.Token{response("main")}})
	def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(child, agent.AgentMiddlewareConfig{ID: "failing", Name: "failing", ErrorPolicy: agent.RecordError, MapInput: func(_ context.Context, result agent.WorkflowResult) (agent.RunInput, error) {
		maps.Add(1)
		if len(result.Input.Prompt.Context) != 1 || result.Input.Prompt.Context[0].Name() != "tenant" {
			t.Error("original context missing")
		}
		if result.Input.Execution == nil || *result.Input.Execution.Limits.MaxTokens != 31 {
			t.Error("original overrides missing")
		}
		return agent.RunInput{}, errors.New("mapping failure")
	}})}
	store := &checkpointStore{}
	d := store.durability()
	part, _ := modelinput.NewNamedPart("tenant", "ACME")
	max := 31
	w, err := agent.New(def).NewRun(context.Background(), agent.RunInput{Durability: &d, Prompt: modelinput.PromptInput{User: ai.TextParts("question"), Context: []modelinput.Part{part}}, Execution: &agent.ExecutionOverrides{Limits: agent.LimitsOverrides{MaxTokens: &max}}})
	if err != nil {
		t.Fatal(err)
	}
	if err := w.RequestPause(); err != nil {
		t.Fatal(err)
	}
	runDurable(t, w)
	resumed, err := agent.New(def).Resume(context.Background(), store.load(), agent.ResumeInput{}, d)
	if err != nil {
		t.Fatal(err)
	}
	store.hook = func(cp agent.Checkpoint) error {
		if cp.Status == agent.StatusSucceeded {
			return errStore
		}
		return nil
	}
	_, err = resumed.Run(context.Background())
	if !errors.Is(err, errStore) {
		t.Fatal(err)
	}
	store.hook = nil
	resumed, err = agent.New(def).Resume(context.Background(), store.load(), agent.ResumeInput{}, d)
	if err != nil {
		t.Fatal(err)
	}
	result := runDurable(t, resumed)
	if result.Status != agent.StatusSucceeded || len(result.Stages) != 1 || len(result.Stages[0].Result.Errors) != 1 || maps.Load() != 1 {
		t.Fatal(result, maps.Load())
	}
}

func TestDurableSnapshotsAreDetached(t *testing.T) {
	model := &durableModel{responses: [][]ai.Token{response("done")}}
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(durableDefinition(model)), store)
	store.hook = func(cp agent.Checkpoint) error {
		cp.RunID = "mutated"
		if cp.Stages[0].Continuation != nil {
			cp.Stages[0].Continuation.Prompt.Messages[0].Parts[0].Text = "mutated"
		}
		return nil
	}
	// The callback owns its copy, so even mutating it cannot alter live state.
	result := runDurable(t, w)
	if result.Input.ID != "logical-run" || result.Checkpoint.RunID != "logical-run" {
		t.Fatal(result)
	}
	result.Checkpoint.Stages[0].Continuation.Prompt.Messages[0].Parts[0].Text = "caller mutation"
	if w.Result().Checkpoint.Stages[0].Continuation.Prompt.Messages[0].Parts[0].Text == "caller mutation" {
		t.Fatal("Result shares checkpoint")
	}
}

func TestDurableRequestPreflightFailurePreservesPreviousUsage(t *testing.T) {
	tool := durableTool(t, "read", func(context.Context, ai.ToolCall) (string, error) {
		return strings.Repeat("large output ", 3000), nil
	})
	tokens := append(calls("read"), ai.Token{Completion: &ai.Completion{UsageReported: true, Usage: ai.Usage{InputTokens: 17, OutputTokens: 3}}})
	model := &durableModel{responses: [][]ai.Token{tokens}}
	def := durableDefinition(model, tool)
	def.RequestBudget = &ai.RequestBudgetConfig{InputLimit: 2000}
	store := &checkpointStore{}
	result, err := newDurableRun(t, agent.New(def), store).Run(context.Background())
	if !errors.Is(err, ai.ErrRequestBudgetExceeded) || model.count() != 1 {
		t.Fatal(err, model.count())
	}
	usage := store.load().Stages[0].Continuation.Usage
	if len(usage) != 1 || !usage[0].Reported || usage[0].Usage.InputTokens != 17 || result.BilledUsage.InputTokens != 17 {
		t.Fatal(usage, result.BilledUsage)
	}
}

func TestDurableCommandRedeliveryAfterStageAdvances(t *testing.T) {
	for _, explicitStage := range []bool{false, true} {
		t.Run(map[bool]string{false: "implicit stage", true: "explicit stage"}[explicitStage], func(t *testing.T) {
			primary := &durableModel{responses: [][]ai.Token{response("main")}}
			child := &durableModel{responses: [][]ai.Token{response("child")}}
			def := durableDefinition(primary)
			def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(agent.New(durableDefinition(child)), agent.AgentMiddlewareConfig{ID: "child", Output: agent.AppendOutput})}
			store := &checkpointStore{}
			w := newDurableRun(t, agent.New(def), store)
			if err := w.RequestPause(); err != nil {
				t.Fatal(err)
			}
			runDurable(t, w)
			command := agent.ResumeInput{CommandID: "command", Prompt: modelinput.PromptInput{User: ai.TextParts("steering")}}
			if explicitStage {
				command.StageID = agent.PrimaryStageID
			}
			var err error
			w, err = agent.New(def).Resume(context.Background(), store.load(), command, store.durability())
			if err != nil {
				t.Fatal(err)
			}
			store.hook = func(cp agent.Checkpoint) error {
				if cp.ActiveStageID == "child" && cp.Stages[1].Continuation != nil {
					return w.RequestPause()
				}
				return nil
			}
			runDurable(t, w)
			cp := store.load()
			if cp.ActiveStageID != "child" || !cp.Stages[0].Complete {
				t.Fatal(cp)
			}
			store.hook = nil
			w, err = agent.New(def).Resume(context.Background(), cp, command, store.durability())
			if err != nil {
				t.Fatal(err)
			}
			result := runDurable(t, w)
			if result.Text != "mainchild" || primary.count() != 1 || child.count() != 1 || len(result.Checkpoint.Commands) != 1 {
				t.Fatal(result, primary.count(), child.count())
			}
		})
	}
}

func TestDurableMiddlewareRecoveryBeforeDownstream(t *testing.T) {
	for _, policy := range []agent.ErrorPolicy{agent.PropagateError, agent.RecordError} {
		t.Run(map[agent.ErrorPolicy]string{agent.PropagateError: "propagate", agent.RecordError: "record"}[policy], func(t *testing.T) {
			var effects atomic.Int32
			tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { effects.Add(1); return "raw", nil })
			childModel := &durableModel{responses: [][]ai.Token{calls("write"), response("reconciled")}}
			child := durableDefinition(childModel, tool)
			child.ToolResultProcessor = loop.ToolResultProcessorFunc(func(context.Context, loop.ToolPolicyInput, loop.ToolResult) (loop.ToolResult, error) {
				return loop.ToolResult{}, errors.New("processing failed")
			})
			last := &durableModel{responses: [][]ai.Token{response("last")}}
			def := durableDefinition(&durableModel{responses: [][]ai.Token{response("main")}})
			def.Middleware = []agent.Middleware{
				agent.NewAgentMiddleware(agent.New(child), agent.AgentMiddlewareConfig{ID: "child", Output: agent.AppendOutput, ErrorPolicy: policy}),
				agent.NewAgentMiddleware(agent.New(durableDefinition(last)), agent.AgentMiddlewareConfig{ID: "last", Output: agent.AppendOutput}),
			}
			store := &checkpointStore{}
			_, err := newDurableRun(t, agent.New(def), store).Run(context.Background())
			if !errors.Is(err, loop.ErrToolResultProcess) || last.count() != 0 {
				t.Fatal(err, last.count())
			}
			cp := store.load()
			if cp.ActiveStageID != "child" || cp.Stages[1].Complete {
				t.Fatal(cp)
			}
			inv := pending(cp).Invocations[0]
			resolution := loop.ToolRecoveryResolution{InvocationID: inv.ID, Action: loop.UseVerifiedResult, Result: &ai.ToolResult{ToolCallID: inv.Call.ID, Name: inv.Call.Name, Parts: ai.TextParts("verified")}, Execution: loop.ToolExecution{State: loop.ToolSucceeded, Output: loop.ToolOutputAccepted}}
			w, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{CommandID: "recover", ToolResolutions: []loop.ToolRecoveryResolution{resolution}}, store.durability())
			if err != nil {
				t.Fatal(err)
			}
			result := runDurable(t, w)
			if effects.Load() != 1 || last.count() != 1 || result.Text != "mainreconciledlast" {
				t.Fatal(effects.Load(), last.count(), result)
			}
		})
	}
}

func TestDurableMetadataHasSameLosslessTypesBeforeAndAfterResume(t *testing.T) {
	var observed []map[string]any
	for _, pause := range []bool{false, true} {
		def := durableDefinition(&durableModel{responses: [][]ai.Token{response("main")}})
		child := agent.New(durableDefinition(&durableModel{responses: [][]ai.Token{response("child")}}))
		def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(child, agent.AgentMiddlewareConfig{ID: "child", MapInput: func(_ context.Context, result agent.WorkflowResult) (agent.RunInput, error) {
			observed = append(observed, result.Input.Meta)
			return agent.RunInput{}, nil
		}})}
		store := &checkpointStore{}
		d := store.durability()
		w, err := agent.New(def).NewRun(context.Background(), agent.RunInput{Durability: &d, Meta: map[string]any{"n": int64(9007199254740993), "nested": struct{ Value int }{3}}})
		if err != nil {
			t.Fatal(err)
		}
		if pause {
			if err := w.RequestPause(); err != nil {
				t.Fatal(err)
			}
			runDurable(t, w)
			w, err = agent.New(def).Resume(context.Background(), store.load(), agent.ResumeInput{}, d)
			if err != nil {
				t.Fatal(err)
			}
		}
		runDurable(t, w)
	}
	if len(observed) != 2 || !reflect.DeepEqual(observed[0], observed[1]) || observed[0]["n"] != json.Number("9007199254740993") {
		t.Fatal(observed)
	}
}

func TestDurablePauseInterruptsRetryBackoff(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		entered := make(chan struct{})
		model := &durableModel{responses: [][]ai.Token{{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient}}}}}
		def := durableDefinition(model)
		def.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1, InitialBackoff: time.Hour, Wait: func(ctx context.Context, _ time.Duration) error {
			close(entered)
			<-ctx.Done()
			return ctx.Err()
		}}
		store := &checkpointStore{}
		w := newDurableRun(t, agent.New(def), store)
		done := make(chan struct{})
		go func() { defer close(done); runDurable(t, w) }()
		<-entered
		if err := w.RequestPause(); err != nil {
			t.Fatal(err)
		}
		<-done
		if w.Result().Status != agent.StatusPaused || model.count() != 1 || store.load().Stages[0].Continuation.RetryCount != 1 {
			t.Fatal(w.Result(), model.count())
		}
	})
}

func TestDurableRetryThenPendingOutputRetainsAttemptIdentity(t *testing.T) {
	tool := durableTool(t, "read", func(context.Context, ai.ToolCall) (string, error) { return "read", nil })
	tokens := append(response("planning"), calls("read")...)
	model := &durableModel{responses: [][]ai.Token{{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient}}}, tokens, response("done")}}
	def := durableDefinition(model, tool)
	def.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	store.hook = func(cp agent.Checkpoint) error {
		if pending(cp) != nil {
			return w.RequestPause()
		}
		return nil
	}
	paused := runDurable(t, w)
	if paused.Text != "" {
		t.Fatal("pending output must remain provisional", paused.Text)
	}
	cp := store.load()
	generation := pending(cp)
	if generation.Attempt != 2 {
		t.Fatal(generation)
	}
	store.hook = nil
	w, err := agent.New(def).Resume(context.Background(), cp, agent.ResumeInput{}, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	events := collectAgentEvents(w.RunEvents(context.Background()))
	result, err := w.Wait()
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "planningdone" || outputText(events) != "planningdone" || model.count() != 3 {
		t.Fatal(result.Text, outputText(events), model.count())
	}
	for _, event := range events {
		if event.Type == agent.EventOutput && event.Output.Text == "planning" && (event.GenerationID != generation.ID || event.AttemptID != 2) {
			t.Fatal(event)
		}
	}
}
