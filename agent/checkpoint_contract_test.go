package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/modelinput"
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
		"version":            func(c *agent.Checkpoint) { c.Version = 2 },
		"run id":             func(c *agent.Checkpoint) { c.RunID = "" },
		"execution id":       func(c *agent.Checkpoint) { c.ExecutionID = "" },
		"definition":         func(c *agent.Checkpoint) { c.DefinitionRevision = "" },
		"revision":           func(c *agent.Checkpoint) { c.Revision = 0 },
		"status":             func(c *agent.Checkpoint) { c.Status = "unknown" },
		"no stages":          func(c *agent.Checkpoint) { c.Stages = nil },
		"primary first":      func(c *agent.Checkpoint) { c.Stages[0].ID = "other"; c.ActiveStageID = "other" },
		"active stage":       func(c *agent.Checkpoint) { c.ActiveStageID = "missing" },
		"duplicate stage":    func(c *agent.Checkpoint) { c.Stages = append(c.Stages, c.Stages[0]) },
		"output policy":      func(c *agent.Checkpoint) { c.Stages[0].Output = agent.ReplaceOutput + 1 },
		"skipped incomplete": func(c *agent.Checkpoint) { c.Stages[0].Skipped = true; c.Stages[0].Complete = false },
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
