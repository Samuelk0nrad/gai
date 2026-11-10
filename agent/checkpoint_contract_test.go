package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/Samuelk0nrad/gai/agent"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/loop"
	"github.com/Samuelk0nrad/gai/modelinput"
)

func TestDurableMiddlewareWaitsForUnfinishedUpstream(t *testing.T) {
	var decided atomic.Int32
	child := agent.New(durableDefinition(&durableModel{responses: [][]ai.Token{response("post")}}))
	def := durableDefinition(&durableModel{responses: [][]ai.Token{{{Err: errors.New("provider down")}}}})
	sink := &agentObservationSink{}
	def.ObservationSink = sink
	def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(child, agent.AgentMiddlewareConfig{ID: "post", Name: "post", Output: agent.AppendOutput, ShouldRun: func(agent.WorkflowResult) bool {
		decided.Add(1)
		return true
	}})}
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	events := collectAgentEvents(w.RunEvents(t.Context()))
	result, err := w.Wait()
	if err == nil || result.Status != agent.StatusFailed || decided.Load() != 0 {
		t.Fatal("failed primary must not reach ShouldRun", err, result.Status, decided.Load())
	}
	var started bool
	var finish agent.Event
	for _, event := range events {
		if event.Source.Name != "post" {
			continue
		}
		switch event.Type {
		case agent.EventStageStart:
			started = true
		case agent.EventStageFinish:
			finish = event
		}
	}
	if !started || finish.StageOutcome != agent.StageSkipped || finish.StageReason != "upstream_unfinished" || finish.StageID != "post" {
		t.Fatalf("stage events: started=%v finish=%+v", started, finish)
	}
	if event, ok := sink.event("agent_middleware_skipped"); !ok || event.Fields["reason"] != "upstream_unfinished" {
		t.Fatal("missing middleware observation", sink.names())
	}
	cp := store.load()
	if cp.Stages[1].DecisionMade {
		t.Fatal("decision saved while upstream was unfinished")
	}

	// After recovery the stage is decided against the completed primary.
	def.Model = &durableModel{responses: [][]ai.Token{response("main")}}
	resumed, err := agent.New(def).Resume(t.Context(), cp, agent.ResumeInput{}, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	result = runDurable(t, resumed)
	if result.Text != "mainpost" || decided.Load() != 1 {
		t.Fatal(result.Text, decided.Load())
	}
	if _, ok := sink.event("agent_middleware_finished"); !ok {
		t.Fatal("missing middleware finish observation", sink.names())
	}
}

func TestDurableResumeRejectsReplayCombinedWithCancelUnstarted(t *testing.T) {
	tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { return "processed", nil })
	def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("write")}}, tool)
	store := &checkpointStore{}
	store.hook = func(cp agent.Checkpoint) error {
		if p := pending(cp); p != nil && p.Invocations[0].Disposition == loop.InvocationCompleted {
			return errStore
		}
		return nil
	}
	if _, err := newDurableRun(t, agent.New(def), store).Run(t.Context()); !errors.Is(err, loop.ErrCheckpointCommit) {
		t.Fatal(err)
	}
	store.hook = nil
	cp := store.load()
	inv := pending(cp).Invocations[0]
	if inv.Disposition != loop.InvocationDispatched {
		t.Fatal("expected an unknown outcome", inv.Disposition)
	}
	input := agent.ResumeInput{CommandID: "replay", PendingTools: agent.CancelUnstartedTools, ToolResolutions: []loop.ToolRecoveryResolution{{InvocationID: inv.ID, Action: loop.ReplayInvocation, SafeReplay: true}}}
	if _, err := agent.New(def).Resume(t.Context(), cp, input, store.durability()); !errors.Is(err, agent.ErrCheckpoint) {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(cp, store.load()) {
		t.Fatal("rejected command claimed the checkpoint")
	}
}

func TestDurableCommandsStoreDigestNotPayload(t *testing.T) {
	def := durableDefinition(&durableModel{responses: [][]ai.Token{response("first"), response("second")}})
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	if err := w.RequestPause(); err != nil {
		t.Fatal(err)
	}
	runDurable(t, w)
	input := agent.ResumeInput{CommandID: "steer", Prompt: modelinput.PromptInput{User: ai.TextParts("a long private steering message")}}
	resumed, err := agent.New(def).Resume(t.Context(), store.load(), input, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	if err := resumed.RequestPause(); err != nil {
		t.Fatal(err)
	}
	runDurable(t, resumed)
	cp := store.load()
	if len(cp.Commands) != 1 || cp.Commands[0].ID != "steer" || len(cp.Commands[0].Digest) != 64 {
		t.Fatal(cp.Commands)
	}
	encoded, _ := json.Marshal(cp.Commands)
	if strings.Contains(string(encoded), "private steering") {
		t.Fatal("command payload duplicated in Commands", string(encoded))
	}
	changed := input
	changed.Prompt = modelinput.PromptInput{User: ai.TextParts("different")}
	if _, err := agent.New(def).Resume(t.Context(), cp, changed, store.durability()); !errors.Is(err, agent.ErrCommandConflict) {
		t.Fatal(err)
	}
	if _, err := agent.New(def).Resume(t.Context(), cp, input, store.durability()); err != nil {
		t.Fatal("exact redelivery rejected", err)
	}
}

func TestCheckpointCloneIsDeep(t *testing.T) {
	def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("write"), response("done")}}, durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { return "ok", nil }))
	store := &checkpointStore{}
	d := store.durability()
	w, err := agent.New(def).NewRun(t.Context(), agent.RunInput{Durability: &d, Meta: map[string]any{"nested": map[string]any{"list": []any{"a"}}}})
	if err != nil {
		t.Fatal(err)
	}
	result := runDurable(t, w)
	original := *result.Checkpoint
	before, _ := json.Marshal(original)
	clone := original.Clone()
	stage := &clone.Stages[0]
	stage.Input.Meta["nested"].(map[string]any)["list"].([]any)[0] = "changed"
	stage.Errors = append(stage.Errors, "changed")
	g := &stage.Continuation.Generations[0]
	g.Messages[0].Parts[0].Text = "changed"
	g.Invocations[0].Call.Args[0] = ' '
	g.Invocations[0].Result.Parts[0].Text = "changed"
	stage.Continuation.Prompt.Messages[0].Parts[0].Text = "changed"
	stage.Settings.MaxIterations++
	after, _ := json.Marshal(original)
	if string(before) != string(after) {
		t.Fatal("clone shares mutable state with the original")
	}

	// Results expose detached metadata too.
	result.Input.Meta["nested"].(map[string]any)["list"].([]any)[0] = "changed"
	if got := w.Result().Checkpoint.Stages[0].Input.Meta["nested"].(map[string]any)["list"].([]any)[0]; got != "a" {
		t.Fatal("result metadata aliases checkpoint state", got)
	}
}

func TestCheckpointValidateRejectsInvalidStructure(t *testing.T) {
	def := durableDefinition(&durableModel{responses: [][]ai.Token{response("done")}})
	store := &checkpointStore{}
	runDurable(t, newDurableRun(t, agent.New(def), store))
	valid := store.load()
	if err := valid.Validate(); err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(*agent.Checkpoint){
		"version":              func(c *agent.Checkpoint) { c.Version = 2 },
		"run id":               func(c *agent.Checkpoint) { c.RunID = "" },
		"execution id":         func(c *agent.Checkpoint) { c.ExecutionID = "" },
		"definition":           func(c *agent.Checkpoint) { c.DefinitionRevision = "" },
		"revision":             func(c *agent.Checkpoint) { c.Revision = 0 },
		"status":               func(c *agent.Checkpoint) { c.Status = "unknown" },
		"no stages":            func(c *agent.Checkpoint) { c.Stages = nil },
		"primary first":        func(c *agent.Checkpoint) { c.Stages[0].ID = "other"; c.ActiveStageID = "other" },
		"active stage":         func(c *agent.Checkpoint) { c.ActiveStageID = "missing" },
		"duplicate stage":      func(c *agent.Checkpoint) { c.Stages = append(c.Stages, c.Stages[0]) },
		"output policy":        func(c *agent.Checkpoint) { c.Stages[0].Output = agent.ReplaceOutput + 1 },
		"skipped incomplete":   func(c *agent.Checkpoint) { c.Stages[0].Skipped = true; c.Stages[0].Complete = false },
		"complete canceled":    func(c *agent.Checkpoint) { c.Stages[0].Canceled = true },
		"succeeded unfinished": func(c *agent.Checkpoint) { c.Stages[0].Complete = false },
		"complete without continuation": func(c *agent.Checkpoint) {
			c.Stages[0].Continuation = nil
		},
		"continuation": func(c *agent.Checkpoint) { c.Stages[0].Continuation.Version = 0 },
		"elapsed":      func(c *agent.Checkpoint) { c.Stages[0].Continuation.Elapsed = -1 },
		"command id":   func(c *agent.Checkpoint) { c.Commands = []agent.ContinuationCommand{{StageID: "primary", Digest: "x"}} },
		"command digest": func(c *agent.Checkpoint) {
			c.Commands = []agent.ContinuationCommand{{ID: "a", StageID: "primary"}}
		},
		"command stage": func(c *agent.Checkpoint) {
			c.Commands = []agent.ContinuationCommand{{ID: "a", StageID: "missing", Digest: "x"}}
		},
		"duplicate command": func(c *agent.Checkpoint) {
			c.Commands = []agent.ContinuationCommand{{ID: "a", StageID: "primary", Digest: "x"}, {ID: "a", StageID: "primary", Digest: "y"}}
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			c := valid.Clone()
			mutate(&c)
			if err := c.Validate(); !errors.Is(err, agent.ErrCheckpoint) {
				t.Fatal(err)
			}
		})
	}
}

func TestDurableSteeringBeforeFirstGenerationKeepsConversationOrder(t *testing.T) {
	model := &durableModel{responses: [][]ai.Token{response("done")}}
	def := durableDefinition(model)
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	if err := w.RequestPause(); err != nil {
		t.Fatal(err)
	}
	runDurable(t, w)
	resumed, err := agent.New(def).Resume(t.Context(), store.load(), agent.ResumeInput{CommandID: "steer", Prompt: modelinput.PromptInput{User: ai.TextParts("followup")}}, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	result := runDurable(t, resumed)
	var conversation, request []string
	for _, m := range result.Primary.Messages {
		conversation = append(conversation, string(m.Role)+":"+m.Text())
	}
	for _, m := range model.requests[0].Messages {
		if m.Role == ai.RoleUser {
			request = append(request, m.Text())
		}
	}
	if !reflect.DeepEqual(conversation, []string{"user:original", "user:followup", "assistant:done"}) {
		t.Fatal("conversation order", conversation)
	}
	if !reflect.DeepEqual(request, []string{"original", "followup"}) {
		t.Fatal("request order", request)
	}
}

func TestDurableResumeCopiesVerifiedResult(t *testing.T) {
	tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { return "real", nil })
	def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("write"), response("done")}}, tool)
	store := &checkpointStore{}
	store.hook = func(cp agent.Checkpoint) error {
		if p := pending(cp); p != nil && p.Invocations[0].Disposition == loop.InvocationCompleted {
			return errStore
		}
		return nil
	}
	if _, err := newDurableRun(t, agent.New(def), store).Run(t.Context()); !errors.Is(err, loop.ErrCheckpointCommit) {
		t.Fatal(err)
	}
	store.hook = nil
	cp := store.load()
	inv := pending(cp).Invocations[0]
	verified := &ai.ToolResult{ToolCallID: inv.Call.ID, Name: inv.Call.Name, Parts: ai.TextParts("verified")}
	input := agent.ResumeInput{CommandID: "resolve", ToolResolutions: []loop.ToolRecoveryResolution{{InvocationID: inv.ID, Action: loop.UseVerifiedResult, Result: verified, Execution: loop.ToolExecution{State: loop.ToolSucceeded}}}}
	w, err := agent.New(def).Resume(t.Context(), cp, input, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	verified.Parts[0].Text = "changed after Resume"
	if got := pending(*w.Result().Checkpoint).Invocations[0].Result.Text(); got != "verified" {
		t.Fatal("checkpoint aliases the caller's verified result", got)
	}
}

func TestDurablePausedUpstreamReportsDownstreamStages(t *testing.T) {
	child := agent.New(durableDefinition(&durableModel{responses: [][]ai.Token{response("child")}}))
	def := durableDefinition(&durableModel{responses: [][]ai.Token{response("main")}})
	def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(child, agent.AgentMiddlewareConfig{ID: "post", Name: "post", Output: agent.AppendOutput})}
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	if err := w.RequestPause(); err != nil {
		t.Fatal(err)
	}
	events := collectAgentEvents(w.RunEvents(t.Context()))
	if result, err := w.Wait(); err != nil || result.Status != agent.StatusPaused {
		t.Fatal(err, result.Status)
	}
	var started, skipped bool
	for _, event := range events {
		if event.Source.Name != "post" {
			continue
		}
		started = started || event.Type == agent.EventStageStart
		skipped = skipped || (event.Type == agent.EventStageFinish && event.StageOutcome == agent.StageSkipped && event.StageReason == "upstream_unfinished")
	}
	if !started || !skipped {
		t.Fatal("paused upstream hid the downstream stage", started, skipped)
	}
	if events[len(events)-1].Type != agent.EventPaused {
		t.Fatal("terminal event", events[len(events)-1].Type)
	}
}

type flakySource struct{ fail *atomic.Bool }

func (flakySource) Name() string { return "retrieval" }
func (s flakySource) Function(context.Context, int) (modelinput.Part, error) {
	if s.fail.Load() {
		return nil, errors.New("vector store timeout")
	}
	return modelinput.NewTextPart("doc"), nil
}

func TestDurableRunStoppedBeforeFirstCheckpointStoresNothing(t *testing.T) {
	var fail atomic.Bool
	fail.Store(true)
	def := durableDefinition(&durableModel{responses: [][]ai.Token{response("done"), response("done")}})
	def.Prompt = func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
		return modelinput.New(modelinput.Definition{ContextSources: []modelinput.ContextSource{flakySource{&fail}}}), nil
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	for name, run := range map[string]func(*agent.Workflow) (agent.WorkflowResult, error){
		"context source fails": func(w *agent.Workflow) (agent.WorkflowResult, error) { return w.Run(t.Context()) },
		"already canceled":     func(w *agent.Workflow) (agent.WorkflowResult, error) { return w.Run(canceled) },
	} {
		t.Run(name, func(t *testing.T) {
			fail.Store(true)
			store := &checkpointStore{}
			result, err := run(newDurableRun(t, agent.New(def), store))
			if err == nil || result.Checkpoint != nil || store.load().Revision != 0 {
				t.Fatal("stored a checkpoint without a continuation", err, result.Checkpoint, store.load().Revision)
			}
			// Nothing was stored, so the same run ID can start again.
			fail.Store(false)
			if result := runDurable(t, newDurableRun(t, agent.New(def), store)); result.Text != "done" {
				t.Fatal(result.Text)
			}
		})
	}
}

func TestDurableCanceledModelAttemptDoesNotConsumeRetryBudget(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	def := durableDefinition(&cancellationModel{ready: make(chan struct{})})
	def.RetryPolicy = &loop.RetryPolicy{TotalTimeout: time.Hour} // no retries allowed
	store := &checkpointStore{}
	var lost agent.Checkpoint
	store.hook = func(cp agent.Checkpoint) error {
		if c := cp.Stages[0].Continuation; c != nil && c.Phase == loop.PhaseGenerating && lost.Revision == 0 {
			lost = cp.Clone()
		}
		return nil
	}
	d := store.durability()
	w, err := agent.New(def).NewRun(ctx, agent.RunInput{Durability: &d})
	if err != nil {
		t.Fatal(err)
	}
	_ = w.RunEvents(ctx)
	<-def.Model.(*cancellationModel).ready
	cancel()
	if result, _ := w.Wait(); result.Status != agent.StatusCanceled {
		t.Fatal(result.Status)
	}
	store.hook = nil
	cp := store.load()
	if phase := cp.Stages[0].Continuation.Phase; phase != loop.PhaseReady {
		t.Fatal("canceled attempt not settled", phase)
	}
	def.Model = &durableModel{responses: [][]ai.Token{response("done")}}
	resumed, err := agent.New(def).Resume(t.Context(), cp, agent.ResumeInput{}, d)
	if err != nil {
		t.Fatal(err)
	}
	if result := runDurable(t, resumed); result.Text != "done" {
		t.Fatal(result.Text)
	}

	// A worker lost mid-generation leaves an unknown attempt, which does cost a
	// retry; with none left, Resume refuses before claiming.
	if _, err := agent.New(def).Resume(t.Context(), lost, agent.ResumeInput{}, d); !errors.Is(err, loop.ErrMaxRetries) {
		t.Fatal(err)
	}
	def.RetryPolicy = &loop.RetryPolicy{TotalTimeout: time.Hour, MaxRetries: 1}
	def.Model = &durableModel{responses: [][]ai.Token{response("recovered")}}
	fresh := &checkpointStore{saved: lost}
	resumed, err = agent.New(def).Resume(t.Context(), lost, agent.ResumeInput{}, fresh.durability())
	if err != nil {
		t.Fatal(err)
	}
	if result := runDurable(t, resumed); result.Text != "recovered" {
		t.Fatal(result.Text)
	}
}

func TestDurableCancelBeforeHandlerStartRevertsDispatch(t *testing.T) {
	var effects atomic.Int32
	tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { effects.Add(1); return "ok", nil })
	def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("write"), response("done")}}, tool)
	store := &checkpointStore{}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	w := newDurableRun(t, agent.New(def), store)
	store.hook = func(cp agent.Checkpoint) error {
		if p := pending(cp); p != nil && p.Invocations[0].Disposition == loop.InvocationDispatched {
			cancel()
		}
		return nil
	}
	if result, _ := w.Run(ctx); result.Status != agent.StatusCanceled {
		t.Fatal(result.Status)
	}
	store.hook = nil
	cp := store.load()
	inv := pending(cp).Invocations[0]
	if effects.Load() != 0 || inv.Disposition != loop.InvocationPlanned || inv.Attempt != 0 || inv.Execution.State != loop.ToolNotStarted {
		t.Fatal("handler never started but dispatch was kept", effects.Load(), inv.Disposition, inv.Attempt, inv.Execution.State)
	}
	resumed, err := agent.New(def).Resume(t.Context(), cp, agent.ResumeInput{}, store.durability())
	if err != nil {
		t.Fatal("recovery demanded for a tool that never ran", err)
	}
	if result := runDurable(t, resumed); result.Text != "done" || effects.Load() != 1 {
		t.Fatal(result.Text, effects.Load())
	}
}

func TestDurablePauseBetweenStagesAcceptsSteering(t *testing.T) {
	childModel := &durableModel{responses: [][]ai.Token{response("post")}}
	child := agent.New(durableDefinition(childModel))
	def := durableDefinition(&durableModel{responses: [][]ai.Token{response("main")}})
	def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(child, agent.AgentMiddlewareConfig{ID: "post", Name: "post", Output: agent.AppendOutput})}
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	store.hook = func(cp agent.Checkpoint) error {
		if c := cp.Stages[0].Continuation; c != nil && c.Phase == loop.PhaseDone {
			_ = w.RequestPause()
		}
		return nil
	}
	if result := runDurable(t, w); result.Status != agent.StatusPaused {
		t.Fatal(result.Status)
	}
	store.hook = nil
	cp := store.load()
	if cp.ActiveStageID != "post" || !cp.Stages[0].Complete || cp.Stages[1].Continuation == nil || childModel.count() != 0 {
		t.Fatal("pause did not leave the next stage with a continuation", cp.ActiveStageID, cp.Stages[1].Continuation != nil, childModel.count())
	}
	input := agent.ResumeInput{CommandID: "steer", Prompt: modelinput.PromptInput{User: ai.TextParts("please add more")}}
	resumed, err := agent.New(def).Resume(t.Context(), cp, input, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	if result := runDurable(t, resumed); result.Text != "mainpost" {
		t.Fatal(result.Text)
	}
	if !strings.Contains(requestText(childModel.requests[0]), "please add more") {
		t.Fatal("steering did not reach the stage", requestText(childModel.requests[0]))
	}
}

func TestDurableAutoCompactedHistoryIsFrozenAcrossResume(t *testing.T) {
	history := autoHistory(strings.Repeat("old context ", 500))
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{autoText("short summary"), autoText("answer")}}
	def := autoHistoryDefinition(model, history)
	store := &checkpointStore{}
	d := store.durability()
	w, err := agent.New(def).NewRun(t.Context(), agent.RunInput{Prompt: modelinput.PromptInput{User: ai.TextParts("question")}, Durability: &d})
	if err != nil {
		t.Fatal(err)
	}
	if err := w.RequestPause(); err != nil {
		t.Fatal(err)
	}
	if result := runDurable(t, w); result.Status != agent.StatusPaused {
		t.Fatal(result.Status)
	}
	loads, writes := history.loads, history.writes
	if writes != 1 || len(model.Requests()) != 1 {
		t.Fatal("compaction did not run once before the pause", writes, len(model.Requests()))
	}
	resumed, err := agent.New(def).Resume(t.Context(), store.load(), agent.ResumeInput{}, d)
	if err != nil {
		t.Fatal(err)
	}
	result := runDurable(t, resumed)
	if result.Text != "answer" || history.loads != loads || history.writes != writes || len(model.Requests()) != 2 {
		t.Fatal("resume selected or compacted history again", result.Text, history.loads, history.writes, len(model.Requests()))
	}
	if !strings.Contains(requestText(model.Requests()[1]), "Conversation summary:\nshort summary") {
		t.Fatal("resumed request lost the frozen summary")
	}
}

// Review finding 1: an admission write canceled with the execution must not
// latch a storage failure that blocks settling tools that already finished.
func TestDurableCanceledAdmissionWriteStillSettlesStartedTools(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	started := make(chan struct{}, 2)
	tool := durableTool(t, "write", func(ctx context.Context, call ai.ToolCall) (string, error) {
		if call.ID == "c" {
			t.Error("third handler ran although its dispatch was never saved")
		}
		started <- struct{}{}
		<-ctx.Done()
		return "processed:" + call.ID, nil
	})
	def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("write", "write", "write"), response("done")}}, tool)
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	store.hook = func(cp agent.Checkpoint) error {
		if p := pending(cp); p != nil && p.Invocations[2].Disposition == loop.InvocationDispatched {
			<-started
			<-started
			cancel()
			return context.Canceled // the storage driver saw the canceled context
		}
		return nil
	}
	result, err := w.Run(ctx)
	if !errors.Is(err, context.Canceled) || result.Status != agent.StatusCanceled {
		t.Fatal(err, result.Status)
	}
	store.hook = nil
	cp := store.load()
	if cp.Status != agent.StatusCanceled {
		t.Fatal("terminal status not saved", cp.Status)
	}
	invocations := pending(cp).Invocations
	for _, inv := range invocations[:2] {
		if inv.Disposition != loop.InvocationCompleted || inv.Result.Text() != "processed:"+inv.Call.ID {
			t.Fatal("finished tool not settled", inv.Disposition)
		}
	}
	if invocations[2].Disposition != loop.InvocationPlanned || invocations[2].Attempt != 0 {
		t.Fatal("unsaved dispatch leaked into storage", invocations[2].Disposition, invocations[2].Attempt)
	}
	if _, err := agent.New(def).Resume(t.Context(), cp, agent.ResumeInput{}, store.durability()); err != nil {
		t.Fatal("finished work requires recovery", err)
	}
}

// Review finding 2: a commit event that cannot be delivered because the run
// was canceled is a cancellation, never a recorded stage failure.
func TestDurableCancelDuringAdmissionCommitIsCancellation(t *testing.T) {
	for i := range 30 {
		child := agent.New(durableDefinition(&durableModel{responses: [][]ai.Token{response("post")}}))
		def := durableDefinition(&durableModel{responses: [][]ai.Token{response("main")}})
		def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(child, agent.AgentMiddlewareConfig{ID: "post", Name: "post", ErrorPolicy: agent.RecordError})}
		ctx, cancel := context.WithCancel(t.Context())
		store := &checkpointStore{}
		w := newDurableRun(t, agent.New(def), store)
		store.hook = func(cp agent.Checkpoint) error {
			if c := cp.Stages[1].Continuation; c != nil && c.Phase == loop.PhaseGenerating {
				cancel() // the write is acknowledged; only event delivery sees the cancel
			}
			return nil
		}
		result, _ := w.Run(ctx)
		cancel()
		cp := store.load()
		if result.Status != agent.StatusCanceled || cp.Stages[1].Complete || len(cp.Stages[1].Errors) != 0 {
			t.Fatalf("run %d: status=%s stored complete=%v errors=%v", i, result.Status, cp.Stages[1].Complete, cp.Stages[1].Errors)
		}
	}
}

// Review finding 3: a handler interrupted because the run stopped keeps an
// unknown outcome; a handler that hit its own tool timeout does not.
func TestDurableInterruptedToolRequiresRecovery(t *testing.T) {
	t.Run("run canceled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		tool := durableTool(t, "write", func(ctx context.Context, _ ai.ToolCall) (string, error) {
			cancel()
			<-ctx.Done()
			return "", ctx.Err()
		})
		def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("write")}}, tool)
		store := &checkpointStore{}
		if result, _ := newDurableRun(t, agent.New(def), store).Run(ctx); result.Status != agent.StatusCanceled {
			t.Fatal(result.Status)
		}
		cp := store.load()
		if inv := pending(cp).Invocations[0]; inv.Disposition != loop.InvocationDispatched || inv.Result != nil {
			t.Fatal("interrupted handler saved as a known outcome", inv.Disposition)
		}
		if _, err := agent.New(def).Resume(t.Context(), cp, agent.ResumeInput{}, store.durability()); !errors.Is(err, loop.ErrRecoveryRequired) {
			t.Fatal(err)
		}
	})
	t.Run("own timeout", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			tool := durableTool(t, "write", func(ctx context.Context, _ ai.ToolCall) (string, error) {
				<-ctx.Done()
				return "", ctx.Err()
			})
			model := &durableModel{responses: [][]ai.Token{calls("write"), response("done")}}
			def := durableDefinition(model, tool)
			def.ToolExecution.DefaultTimeout = time.Second
			store := &checkpointStore{}
			result := runDurable(t, newDurableRun(t, agent.New(def), store))
			if result.Text != "done" || model.count() != 2 {
				t.Fatal("tool timeout should be an ordinary tool error", result.Text, model.count())
			}
			inv := store.load().Stages[0].Continuation.Generations[0].Invocations[0]
			if inv.Disposition != loop.InvocationCompleted || inv.Execution.State != loop.ToolTimedOut {
				t.Fatal(inv.Disposition, inv.Execution.State)
			}
		})
	})
}

// Review finding 5: the attempt timeout must not include checkpoint storage.
func TestDurableAttemptTimeoutExcludesCommitLatency(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		model := &durableModel{responses: [][]ai.Token{response("done")}, onRequest: func(ctx context.Context, _ int) {
			select {
			case <-time.After(time.Second):
			case <-ctx.Done():
			}
		}}
		def := durableDefinition(model)
		def.RetryPolicy = &loop.RetryPolicy{AttemptTimeout: 2 * time.Second}
		store := &checkpointStore{}
		store.hook = func(cp agent.Checkpoint) error {
			if c := cp.Stages[0].Continuation; c != nil && c.Phase == loop.PhaseGenerating {
				time.Sleep(1800 * time.Millisecond) // slow compare-and-swap write
			}
			return nil
		}
		result := runDurable(t, newDurableRun(t, agent.New(def), store))
		if result.Text != "done" || model.count() != 1 {
			t.Fatal("storage latency consumed the model timeout", result.Text, model.count())
		}
	})
}

// Review finding 6: a pause and a cancel landing on the dispatch commit leave
// the dispatch reverted, never saved as an unknown outcome.
func TestDurablePauseAndCancelAtDispatchRevertDispatch(t *testing.T) {
	for i := range 30 {
		var effects atomic.Int32
		tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { effects.Add(1); return "ok", nil })
		def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("write")}}, tool)
		ctx, cancel := context.WithCancel(t.Context())
		store := &checkpointStore{}
		w := newDurableRun(t, agent.New(def), store)
		store.hook = func(cp agent.Checkpoint) error {
			if p := pending(cp); p != nil && p.Invocations[0].Disposition == loop.InvocationDispatched {
				_ = w.RequestPause()
				cancel()
			}
			return nil
		}
		_, _ = w.Run(ctx)
		cancel()
		inv := pending(store.load()).Invocations[0]
		if effects.Load() != 0 || inv.Disposition != loop.InvocationPlanned || inv.Attempt != 0 {
			t.Fatalf("run %d: effects=%d disposition=%s attempt=%d", i, effects.Load(), inv.Disposition, inv.Attempt)
		}
	}
}

// Review finding 7: iterations accepted live keep their diagnostics.
func TestDurableLiveIterationsKeepDiagnostics(t *testing.T) {
	tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { return "ok", nil })
	def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("write"), response("done")}}, tool)
	def.ToolPolicy = loop.ToolPolicyFunc(func(context.Context, loop.ToolPolicyInput) (loop.ToolDecision, error) {
		return loop.ToolDecision{Action: loop.ToolDeny, Code: "blocked"}, nil
	})
	result := runDurable(t, newDurableRun(t, agent.New(def), &checkpointStore{}))
	var denied bool
	for _, it := range result.Primary.Iterations {
		for _, part := range it.Parts {
			if part.ToolResp != nil && errors.Is(part.ToolResp.Err, loop.ErrToolDenied) {
				denied = true
			}
		}
	}
	if !denied {
		t.Fatal("typed tool error lost in a durable run")
	}
}

// Review finding 8: a pause during a final answer without tool calls finishes the run.
func TestDurablePauseDuringFinalAnswerFinishes(t *testing.T) {
	var w *agent.Workflow
	model := &durableModel{responses: [][]ai.Token{response("done")}, onRequest: func(context.Context, int) {
		if err := w.RequestPause(); err != nil {
			t.Error(err)
		}
	}}
	store := &checkpointStore{}
	w = newDurableRun(t, agent.New(durableDefinition(model)), store)
	result := runDurable(t, w)
	if result.Status != agent.StatusSucceeded || result.Text != "done" || store.load().Status != agent.StatusSucceeded {
		t.Fatal(result.Status, result.Text, store.load().Status)
	}
}

// Review finding 9: a setup failure after a canceled execution records the
// failure instead of producing an invalid checkpoint.
func TestDurableSetupFailureAfterCanceledStage(t *testing.T) {
	var maps atomic.Int32
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	child := agent.New(durableDefinition(&durableModel{responses: [][]ai.Token{response("post")}}))
	def := durableDefinition(&durableModel{responses: [][]ai.Token{response("main")}})
	def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(child, agent.AgentMiddlewareConfig{ID: "post", Name: "post", ErrorPolicy: agent.RecordError, MapInput: func(context.Context, agent.WorkflowResult) (agent.RunInput, error) {
		if maps.Add(1) == 1 {
			cancel() // the child is canceled before its loop saves anything
			return agent.RunInput{}, nil
		}
		return agent.RunInput{}, errors.New("mapping failure")
	}})}
	store := &checkpointStore{}
	if result, _ := newDurableRun(t, agent.New(def), store).Run(ctx); result.Status != agent.StatusCanceled {
		t.Fatal(result.Status)
	}
	cp := store.load()
	if !cp.Stages[1].Canceled || cp.Stages[1].Complete {
		t.Skip("child failed before cancellation was recorded")
	}
	resumed, err := agent.New(def).Resume(t.Context(), cp, agent.ResumeInput{}, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	result := runDurable(t, resumed)
	stage := result.Checkpoint.Stages[1]
	if result.Status != agent.StatusSucceeded || !stage.Complete || stage.Canceled || len(stage.Errors) != 1 {
		t.Fatal(result.Status, stage.Complete, stage.Canceled, stage.Errors)
	}
}

// Review finding 10: provisional output of a paused attempt is not forwarded
// by a ReplaceOutput stage.
func TestDurablePausedAttemptOutputIsNotForwarded(t *testing.T) {
	tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { return "ok", nil })
	plan := append(response("thinking"), calls("write")...)
	def := durableDefinition(&durableModel{responses: [][]ai.Token{plan}}, tool)
	child := agent.New(durableDefinition(&durableModel{responses: [][]ai.Token{response("post")}}))
	def.Middleware = []agent.Middleware{agent.NewAgentMiddleware(child, agent.AgentMiddlewareConfig{ID: "post", Name: "post", Output: agent.ReplaceOutput})}
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	store.hook = func(cp agent.Checkpoint) error {
		if pending(cp) != nil {
			return w.RequestPause()
		}
		return nil
	}
	events := collectAgentEvents(w.RunEvents(t.Context()))
	if result, err := w.Wait(); err != nil || result.Status != agent.StatusPaused {
		t.Fatal(err, result.Status)
	}
	if text := outputText(events); text != "" {
		t.Fatal("paused attempt output forwarded", text)
	}
}

// Review finding 14: a commit failure is reported once.
func TestDurableCommitFailureIsNotDuplicated(t *testing.T) {
	tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { return "ok", nil })
	def := durableDefinition(&durableModel{responses: [][]ai.Token{calls("write")}}, tool)
	store := &checkpointStore{}
	store.hook = func(cp agent.Checkpoint) error {
		if pending(cp) != nil {
			return errStore
		}
		return nil
	}
	result, err := newDurableRun(t, agent.New(def), store).Run(t.Context())
	if !errors.Is(err, errStore) || len(result.Primary.Errors) == 0 {
		t.Fatal(err)
	}
	if n := strings.Count(result.Primary.Errors[0].Error(), loop.ErrCheckpointCommit.Error()); n != 1 {
		t.Fatalf("commit failure reported %d times: %v", n, result.Primary.Errors[0])
	}
}
