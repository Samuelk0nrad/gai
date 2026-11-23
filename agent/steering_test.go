package agent_test

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/Samuelk0nrad/gai/agent"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/loop"
	"github.com/Samuelk0nrad/gai/modelinput"
)

const canceledBySteering = "Not executed: canceled by steering."

func steerInput(id, text string, tools agent.PendingToolAction) agent.SteerInput {
	return agent.SteerInput{CommandID: id, Prompt: modelinput.PromptInput{User: ai.TextParts(text)}, PendingTools: tools}
}

// steer reports failures with t.Error, so it is safe from model and tool goroutines.
func steer(t *testing.T, w *agent.Workflow, id, text string, tools agent.PendingToolAction) {
	t.Helper()
	if err := w.Steer(context.Background(), steerInput(id, text, tools)); err != nil {
		t.Errorf("steer %s: %v", id, err)
	}
}

func runEvents(t *testing.T, w *agent.Workflow) ([]agent.Event, agent.WorkflowResult) {
	t.Helper()
	var events []agent.Event
	for event := range w.RunEvents(context.Background()) {
		events = append(events, event)
	}
	result, err := w.Wait()
	if err != nil {
		t.Fatal(err)
	}
	return events, result
}

// messageIndexes returns the positions of messages whose text is text.
func messageIndexes(messages []ai.Message, text string) []int {
	var indexes []int
	for i, m := range messages {
		if m.Text() == text {
			indexes = append(indexes, i)
		}
	}
	return indexes
}

// toolResultTexts returns the tool result texts in request order.
func toolResultTexts(messages []ai.Message) []string {
	var texts []string
	for _, m := range messages {
		for _, result := range m.ToolResults() {
			texts = append(texts, result.Text())
		}
	}
	return texts
}

func eventIndex(events []agent.Event, kind agent.EventType, command string) int {
	return slices.IndexFunc(events, func(e agent.Event) bool {
		return e.Type == kind && e.Steering != nil && slices.Contains(e.Steering.Commands, command)
	})
}

// Steering events keep workflow events comparable.
var _ = agent.Event{} == agent.Event{}

// invocation finds the record of a provider call in the primary continuation.
func invocation(cp agent.Checkpoint, callID string) *loop.InvocationRecord {
	c := cp.Stages[0].Continuation
	if c == nil {
		return nil
	}
	generations := c.Generations
	if c.Pending != nil {
		generations = append(slices.Clone(generations), *c.Pending)
	}
	for _, g := range generations {
		for i := range g.Invocations {
			if g.Invocations[i].Call.ID == callID {
				return &g.Invocations[i]
			}
		}
	}
	return nil
}

func TestSteerDuringModelCallAppliesInOneGenerationAtNextBoundary(t *testing.T) {
	const due, chat = "[System] due date changed", "[Main chat] focus on the summary"
	var w *agent.Workflow
	model := &durableModel{responses: [][]ai.Token{response("first"), response("second")}}
	model.onRequest = func(_ context.Context, index int) {
		if index != 0 {
			return
		}
		steer(t, w, "due", due, agent.ContinuePendingTools)
		steer(t, w, "chat", chat, agent.ContinuePendingTools)
		// Redelivery is a no-op; a different payload under the same ID conflicts.
		steer(t, w, "due", due, agent.ContinuePendingTools)
		if err := w.Steer(context.Background(), steerInput("due", "other", agent.ContinuePendingTools)); !errors.Is(err, agent.ErrCommandConflict) {
			t.Errorf("conflict: %v", err)
		}
	}
	store := &checkpointStore{}
	w = newDurableRun(t, agent.New(durableDefinition(model)), store)
	events, result := runEvents(t, w)
	if result.Status != agent.StatusSucceeded || model.count() != 2 || result.Text != "firstsecond" {
		t.Fatalf("status=%s requests=%d text=%q", result.Status, model.count(), result.Text)
	}
	messages := model.requests[1].Messages
	first, dues, chats := messageIndexes(messages, "first"), messageIndexes(messages, due), messageIndexes(messages, chat)
	if len(first) != 1 || len(dues) != 1 || len(chats) != 1 || !(first[0] < dues[0] && dues[0] < chats[0]) {
		t.Fatalf("second request order: first=%v due=%v chat=%v", first, dues, chats)
	}
	if len(messageIndexes(result.Primary.Messages, due)) != 1 || len(messageIndexes(result.Primary.Messages, chat)) != 1 {
		t.Fatal("steering must appear once in the accepted conversation")
	}
	cp := store.load()
	var steering *loop.Generation
	for i, g := range cp.Stages[0].Continuation.Generations {
		if len(messageIndexes(g.Messages, due)) == 1 {
			steering = &cp.Stages[0].Continuation.Generations[i]
		}
	}
	if steering == nil || len(steering.Messages) != 2 || len(cp.Stages[0].Continuation.Steering) != 0 || len(cp.Commands) != 2 {
		t.Fatalf("steering generation=%+v commands=%+v", steering, cp.Commands)
	}

	acceptedDue, acceptedChat := eventIndex(events, agent.EventSteeringAccepted, "due"), eventIndex(events, agent.EventSteeringAccepted, "chat")
	applied := eventIndex(events, agent.EventSteeringApplied, "due")
	if acceptedDue < 0 || acceptedChat < acceptedDue || applied < acceptedChat {
		t.Fatalf("event order: accepted due=%d chat=%d applied=%d", acceptedDue, acceptedChat, applied)
	}
	event := events[applied]
	if !reflect.DeepEqual(event.Steering.Commands, []string{"due", "chat"}) || event.GenerationID != steering.ID || event.StageID != agent.PrimaryStageID || event.RunID != "logical-run" {
		t.Fatalf("applied event %+v", event)
	}
	if events[acceptedDue].CheckpointRevision == 0 || events[acceptedDue].StageID != agent.PrimaryStageID {
		t.Fatalf("accepted event %+v", events[acceptedDue])
	}
	accepted := 0
	for _, e := range events {
		if e.Type == agent.EventSteeringAccepted {
			accepted++
		}
	}
	if accepted != 2 {
		t.Fatalf("accepted events=%d", accepted)
	}

	// After the workflow returned, new input needs a new run.
	if err := w.Steer(context.Background(), steerInput("late", "too late", agent.ContinuePendingTools)); !errors.Is(err, agent.ErrWorkflowFinished) {
		t.Fatal(err)
	}
	if err := w.Steer(context.Background(), steerInput("due", due, agent.ContinuePendingTools)); err != nil {
		t.Fatal("redelivery after finish", err)
	}
}

func TestSteerDuringToolExecution(t *testing.T) {
	for _, tools := range []agent.PendingToolAction{agent.ContinuePendingTools, agent.CancelUnstartedTools} {
		t.Run(string(tools)+"_", func(t *testing.T) {
			const text = "stop writing"
			var w *agent.Workflow
			var effects atomic.Int32
			tool := durableTool(t, "write", func(_ context.Context, call ai.ToolCall) (string, error) {
				if effects.Add(1) == 1 {
					steer(t, w, "stop", text, tools)
				}
				return "written " + call.ID, nil
			})
			model := &durableModel{responses: [][]ai.Token{calls("write", "write"), response("done")}}
			def := durableDefinition(model, tool)
			def.ToolExecution = loop.ToolExecutionConfig{MaxConcurrent: 1}
			store := &checkpointStore{}
			w = newDurableRun(t, agent.New(def), store)
			events, result := runEvents(t, w)
			cancel := tools == agent.CancelUnstartedTools
			// The accepted event precedes the events the command causes.
			resultB := slices.IndexFunc(events, func(e agent.Event) bool {
				return (e.Type == agent.EventToolResult || e.Type == agent.EventToolError) && e.ToolCall != nil && e.ToolCall.ID == "b"
			})
			if accepted := eventIndex(events, agent.EventSteeringAccepted, "stop"); accepted < 0 || resultB < 0 || cancel && resultB < accepted {
				t.Fatalf("accepted=%d result b=%d", accepted, resultB)
			}
			want := []string{"written a", "written b"}
			if cancel {
				want = []string{"written a", canceledBySteering}
			}
			messages := model.requests[1].Messages
			if result.Status != agent.StatusSucceeded || int(effects.Load()) != len(want)-btoi(cancel) || !reflect.DeepEqual(toolResultTexts(messages), want) {
				t.Fatalf("status=%s effects=%d results=%q", result.Status, effects.Load(), toolResultTexts(messages))
			}
			// Steering follows every tool result of the accepted plan.
			steered := messageIndexes(messages, text)
			last := slices.IndexFunc(messages, func(m ai.Message) bool {
				return slices.ContainsFunc(m.ToolResults(), func(r ai.ToolResult) bool { return r.ToolCallID == "b" })
			})
			if len(steered) != 1 || steered[0] < last {
				t.Fatalf("steering=%v last result=%d", steered, last)
			}
			if b := invocation(store.load(), "b"); b == nil || b.CanceledBySteering != cancel {
				t.Fatalf("invocation b %+v", b)
			}
		})
	}
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}

// The final answer has streamed but is not saved yet, so the loop takes the
// steering with the commit that accepts the answer. The loop test
// TestSteeringMergedIntoFinalCommitContinuesLoop covers steering that the
// owner adds to the commit that marks the answer final.
func TestSteerAfterFinalAnswerStreamedContinuesRun(t *testing.T) {
	var w *agent.Workflow
	model := &durableModel{responses: [][]ai.Token{response("first"), response("second")}}
	model.afterResponse = func(_ context.Context, index int) {
		if index == 0 {
			steer(t, w, "more", "one more thing", agent.ContinuePendingTools)
		}
	}
	store := &checkpointStore{}
	w = newDurableRun(t, agent.New(durableDefinition(model)), store)
	_, result := runEvents(t, w)
	if result.Status != agent.StatusSucceeded || model.count() != 2 || result.Text != "firstsecond" {
		t.Fatalf("status=%s requests=%d text=%q", result.Status, model.count(), result.Text)
	}
	messages := model.requests[1].Messages
	if first, steered := messageIndexes(messages, "first"), messageIndexes(messages, "one more thing"); len(first) != 1 || len(steered) != 1 || steered[0] < first[0] {
		t.Fatalf("first=%v steering=%v", first, steered)
	}
	if c := store.load().Stages[0].Continuation; c.Phase != loop.PhaseDone || len(c.Steering) != 0 {
		t.Fatalf("phase=%s steering=%d", c.Phase, len(c.Steering))
	}
}

func TestSteerAfterFinalCommitIsRefused(t *testing.T) {
	var w *agent.Workflow
	model := &durableModel{responses: [][]ai.Token{response("only")}}
	store := &checkpointStore{}
	late := make(chan error, 1)
	var once sync.Once
	// The commit that saves the final answer holds the commit lock, so this
	// Steer runs right after it.
	store.hook = func(cp agent.Checkpoint) error {
		if c := cp.Stages[0].Continuation; c.Phase == loop.PhaseDone && len(c.Steering) == 0 {
			once.Do(func() {
				go func() {
					late <- w.Steer(context.Background(), steerInput("late", "too late", agent.ContinuePendingTools))
				}()
			})
		}
		return nil
	}
	w = newDurableRun(t, agent.New(durableDefinition(model)), store)
	_, result := runEvents(t, w)
	if err := <-late; !errors.Is(err, agent.ErrWorkflowFinished) {
		t.Fatal(err)
	}
	cp := store.load()
	if result.Status != agent.StatusSucceeded || model.count() != 1 || len(cp.Commands) != 0 || len(cp.Stages[0].Continuation.Steering) != 0 {
		t.Fatalf("status=%s requests=%d checkpoint=%+v", result.Status, model.count(), cp.Commands)
	}
}

func TestSteerCommittedBeforeCrashIsAppliedOnResume(t *testing.T) {
	const text = "[System] wake-up is due"
	var w *agent.Workflow
	var crashed agent.Checkpoint
	store := &checkpointStore{}
	model := &durableModel{responses: [][]ai.Token{response("first"), response("second")}}
	model.onRequest = func(_ context.Context, index int) {
		if index == 0 {
			steer(t, w, "wake", text, agent.ContinuePendingTools)
			// The worker is lost here; storage keeps the steering.
			crashed = store.load()
		}
	}
	def := durableDefinition(model)
	def.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}
	w = newDurableRun(t, agent.New(def), store)
	runDurable(t, w)
	if c := crashed.Stages[0].Continuation; c.Phase != loop.PhaseGenerating || len(c.Steering) != 1 || !reflect.DeepEqual(c.SteeringCommands, []string{"wake"}) {
		t.Fatalf("crashed continuation %+v", c)
	}
	encoded := &checkpointStore{}
	if err := encoded.commit(context.Background(), 0, crashed); err != nil {
		t.Fatal(err)
	}

	next := &durableModel{responses: [][]ai.Token{response("resumed")}}
	def.Model = next
	lost := &checkpointStore{saved: encoded.load()}
	resumed, err := agent.New(def).Resume(context.Background(), lost.load(), agent.ResumeInput{}, lost.durability())
	if err != nil {
		t.Fatal(err)
	}
	events, result := runEvents(t, resumed)
	if result.Status != agent.StatusSucceeded || result.Text != "resumed" {
		t.Fatalf("status=%s text=%q", result.Status, result.Text)
	}
	messages := next.requests[0].Messages
	if original, steered := messageIndexes(messages, "original"), messageIndexes(messages, text); len(original) != 1 || len(steered) != 1 || steered[0] < original[0] {
		t.Fatalf("original=%v steering=%v", original, steered)
	}
	if applied := eventIndex(events, agent.EventSteeringApplied, "wake"); applied < 0 {
		t.Fatal("resumed run did not report the saved command as applied")
	}
	if err := resumed.Steer(context.Background(), steerInput("wake", text, agent.ContinuePendingTools)); err != nil {
		t.Fatal("redelivery", err)
	}
}

// blockingPart blocks its first render, which happens after the workflow has
// started and before its first checkpoint commit.
type blockingPart struct {
	once      *sync.Once
	rendering chan struct{}
	release   chan struct{}
}

func (blockingPart) Name() string { return "blocking" }
func (blockingPart) Tokens(context.Context, ai.TokenCounter) (int, error) {
	return 1, nil
}
func (p blockingPart) Render(context.Context) (modelinput.RenderNode, error) {
	p.once.Do(func() {
		close(p.rendering)
		<-p.release
	})
	return modelinput.NewTextPart("context").Render(context.Background())
}

func TestSteerWaitsForFirstCommit(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		const text = "arrived early"
		part := blockingPart{once: &sync.Once{}, rendering: make(chan struct{}), release: make(chan struct{})}
		steered := make(chan error, 1)
		result := make(chan error, 1)
		model := &durableModel{responses: [][]ai.Token{response("a"), response("b")}}
		// The first model call waits for the command, so it never arrives after
		// the final answer.
		model.onRequest = func(_ context.Context, index int) {
			if index == 0 {
				result <- <-steered
			}
		}
		store := &checkpointStore{}
		durability := store.durability()
		w, err := agent.New(durableDefinition(model)).NewRun(context.Background(), agent.RunInput{ID: "logical-run", Prompt: modelinput.PromptInput{User: ai.TextParts("original"), Context: []modelinput.Part{part}}, Durability: &durability})
		if err != nil {
			t.Fatal(err)
		}
		done := make(chan agent.WorkflowResult)
		go func() {
			_, workflow := runEvents(t, w)
			done <- workflow
		}()
		<-part.rendering
		go func() {
			steered <- w.Steer(context.Background(), steerInput("early", text, agent.ContinuePendingTools))
		}()
		// Steer waits for the first commit instead of failing or committing
		// before the loop has saved its prompt.
		synctest.Wait()
		if len(steered) != 0 {
			t.Fatal("Steer returned before the first commit:", <-steered)
		}
		if store.load().Revision != 0 {
			t.Fatal("checkpoint committed before the run materialized its prompt")
		}
		close(part.release)
		if err := <-result; err != nil {
			t.Fatal(err)
		}
		workflow := <-done
		last := model.requests[model.count()-1].Messages
		if workflow.Status != agent.StatusSucceeded || len(messageIndexes(last, text)) != 1 || len(messageIndexes(workflow.Primary.Messages, text)) != 1 {
			t.Fatalf("status=%s requests=%d", workflow.Status, model.count())
		}
	})
}

func TestSteerCancelsPlanOfResponseInFlight(t *testing.T) {
	const text = "use the new address"
	var w *agent.Workflow
	var effects atomic.Int32
	tool := durableTool(t, "write", func(context.Context, ai.ToolCall) (string, error) { effects.Add(1); return "written", nil })
	model := &durableModel{responses: [][]ai.Token{calls("write"), response("replanned")}}
	model.onRequest = func(_ context.Context, index int) {
		if index == 0 {
			steer(t, w, "address", text, agent.CancelUnstartedTools)
		}
	}
	def := durableDefinition(model, tool)
	// The stale call would otherwise pause for a deferred approval.
	def.ToolPolicy = loop.ToolPolicyFunc(func(context.Context, loop.ToolPolicyInput) (loop.ToolDecision, error) {
		return loop.ToolDecision{Action: loop.ToolRequireApproval}, nil
	})
	store := &checkpointStore{}
	durability := store.durability()
	durability.DeferredApprovals = true
	w, err := agent.New(def).NewRun(context.Background(), agent.RunInput{ID: "logical-run", Prompt: modelinput.PromptInput{User: ai.TextParts("original")}, Durability: &durability})
	if err != nil {
		t.Fatal(err)
	}
	_, result := runEvents(t, w)
	if result.Status != agent.StatusSucceeded || effects.Load() != 0 || model.count() != 2 || result.Text != "replanned" {
		t.Fatalf("status=%s effects=%d requests=%d text=%q", result.Status, effects.Load(), model.count(), result.Text)
	}
	messages := model.requests[1].Messages
	if !reflect.DeepEqual(toolResultTexts(messages), []string{canceledBySteering}) || len(messageIndexes(messages, text)) != 1 {
		t.Fatalf("results=%q", toolResultTexts(messages))
	}
	if a := invocation(store.load(), "a"); a == nil || !a.CanceledBySteering || a.Approval != nil {
		t.Fatalf("invocation %+v", a)
	}
}

// Steering that cancels unstarted calls races the loop's dispatch of the next
// call. Whatever the order, a call that was still planned in the checkpoint
// the command was committed against never runs, and one that had started is
// never canceled.
func TestSteerCancelRacingDispatch(t *testing.T) {
	for _, denied := range []bool{false, true} {
		t.Run(fmt.Sprintf("refused_without_handler=%v", denied), func(t *testing.T) {
			for i := range 40 {
				var w *agent.Workflow
				var ranB atomic.Bool
				steered := make(chan error, 1)
				tool := durableTool(t, "write", func(_ context.Context, call ai.ToolCall) (string, error) {
					if call.ID == "a" {
						go func() {
							steered <- w.Steer(context.Background(), steerInput("stop", "stop", agent.CancelUnstartedTools))
						}()
					} else {
						ranB.Store(true)
					}
					return "written", nil
				})
				// The next model call waits for the command, so it never arrives after
				// the final answer. Accepted during that call, it reopens the run.
				model := &durableModel{responses: [][]ai.Token{calls("write", "write"), response("done"), response("after steering")}}
				model.onRequest = func(_ context.Context, index int) {
					if index == 1 {
						if err := <-steered; err != nil {
							t.Error(err)
						}
					}
				}
				def := durableDefinition(model, tool)
				def.ToolExecution = loop.ToolExecutionConfig{MaxConcurrent: 1}
				if denied {
					def.ToolPolicy = loop.ToolPolicyFunc(func(_ context.Context, input loop.ToolPolicyInput) (loop.ToolDecision, error) {
						if input.Call.ID == "b" {
							return loop.ToolDecision{Action: loop.ToolDeny, Reason: "denied"}, nil
						}
						return loop.ToolDecision{Action: loop.ToolAllow}, nil
					})
				}
				var mu sync.Mutex
				var history []agent.Checkpoint
				store := &checkpointStore{}
				store.hook = func(cp agent.Checkpoint) error {
					mu.Lock()
					history = append(history, cp.Clone())
					mu.Unlock()
					return nil
				}
				w = newDurableRun(t, agent.New(def), store)
				events, result := runEvents(t, w)
				if result.Status != agent.StatusSucceeded {
					t.Fatal(result.Status, result.Errors)
				}
				// Loop commit events report the loop's own revisions, never the
				// one Steer committed in between.
				acceptedEvent := eventIndex(events, agent.EventSteeringAccepted, "stop")
				if acceptedEvent < 0 {
					t.Fatalf("iteration %d: no accepted event", i)
				}
				accepted := events[acceptedEvent].CheckpointRevision
				for _, e := range events {
					if e.Type == agent.EventCheckpointCommitted && e.CheckpointRevision == accepted {
						t.Fatalf("iteration %d: loop commit event reports Steer's revision %d", i, accepted)
					}
				}
				acceptedAt := slices.IndexFunc(history, func(cp agent.Checkpoint) bool {
					return slices.ContainsFunc(cp.Commands, func(c agent.ContinuationCommand) bool { return c.ID == "stop" })
				})
				if acceptedAt < 1 {
					t.Fatal("command not committed")
				}
				before := invocation(history[acceptedAt-1], "b")
				if before == nil {
					t.Fatal("plan not committed before the command")
				}
				wantCanceled := before.Disposition == loop.InvocationPlanned
				for _, cp := range history[acceptedAt:] {
					if b := invocation(cp, "b"); b != nil && b.CanceledBySteering != wantCanceled {
						t.Fatalf("iteration %d revision %d: canceled=%v want %v (%+v)", i, cp.Revision, b.CanceledBySteering, wantCanceled, b)
					}
				}
				final := invocation(store.load(), "b")
				if final.CanceledBySteering && (final.Result == nil || final.Result.Text() != canceledBySteering) {
					t.Fatalf("canceled record keeps another outcome: %+v", final)
				}
				if ranB.Load() == wantCanceled && !denied {
					t.Fatalf("iteration %d: b ran=%v but canceled=%v", i, ranB.Load(), wantCanceled)
				}
				if denied && ranB.Load() {
					t.Fatal("denied call ran")
				}
			}
		})
	}
}

func TestSteerRejectsUnsupportedCalls(t *testing.T) {
	ctx := context.Background()
	plain, err := agent.New(durableDefinition(&durableModel{})).NewRun(ctx, agent.RunInput{Prompt: modelinput.PromptInput{User: ai.TextParts("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	if err := plain.Steer(ctx, steerInput("id", "x", agent.ContinuePendingTools)); !errors.Is(err, agent.ErrDurability) {
		t.Fatal("non-durable", err)
	}
	w := newDurableRun(t, agent.New(durableDefinition(&durableModel{})), &checkpointStore{})
	if err := w.Steer(ctx, steerInput("", "x", agent.ContinuePendingTools)); !errors.Is(err, agent.ErrInvalidCommand) {
		t.Fatal("missing ID", err)
	}
	if err := w.Steer(ctx, steerInput("id", "x", "later")); !errors.Is(err, agent.ErrInvalidCommand) {
		t.Fatal("pending tools", err)
	}
	if err := w.Steer(ctx, steerInput("id", "x", agent.ContinuePendingTools)); !errors.Is(err, agent.ErrWorkflowNotStarted) {
		t.Fatal("not started", err)
	}
}

func TestSteerStopsBeforeLastModelRequest(t *testing.T) {
	var w *agent.Workflow
	var late error
	model := &durableModel{responses: [][]ai.Token{response("first"), response("second")}}
	model.onRequest = func(_ context.Context, index int) {
		switch index {
		case 0:
			steer(t, w, "early", "still time", agent.ContinuePendingTools)
		case 1:
			// The second request is the last allowed one, so no later request
			// could see this input.
			late = w.Steer(context.Background(), steerInput("late", "too late", agent.ContinuePendingTools))
		}
	}
	def := durableDefinition(model)
	def.Limits.MaxLoopIterations = 2
	store := &checkpointStore{}
	w = newDurableRun(t, agent.New(def), store)
	_, result := runEvents(t, w)
	if !errors.Is(late, agent.ErrWorkflowFinished) {
		t.Fatal("late steering", late)
	}
	if result.Status != agent.StatusSucceeded || model.count() != 2 || len(messageIndexes(model.requests[1].Messages, "still time")) != 1 {
		t.Fatalf("status=%s requests=%d errors=%v", result.Status, model.count(), result.Errors)
	}
	if slices.ContainsFunc(store.load().Commands, func(c agent.ContinuationCommand) bool { return c.ID == "late" }) {
		t.Fatal("refused command saved")
	}
}

func TestSteerAndTerminalToolBatch(t *testing.T) {
	cases := []struct {
		name        string
		during      int // 0 steers during the model call, 1 during the terminal tool
		tools       agent.PendingToolAction
		wantErr     error
		wantEffects int32
		wantCalls   int
	}{
		// Once the agent accepted its terminal batch, steering is refused.
		{name: "during terminal tool", during: 1, tools: agent.ContinuePendingTools, wantErr: agent.ErrWorkflowFinished, wantEffects: 1, wantCalls: 1},
		// Steering accepted before the batch was chosen is applied after it.
		{name: "before batch", during: 0, tools: agent.ContinuePendingTools, wantEffects: 1, wantCalls: 2},
		// Canceling the unstarted terminal call keeps the agent running.
		{name: "cancel batch", during: 0, tools: agent.CancelUnstartedTools, wantEffects: 0, wantCalls: 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var w *agent.Workflow
			var effects atomic.Int32
			var steerErr error
			tool := durableTool(t, "submit", func(context.Context, ai.ToolCall) (string, error) {
				effects.Add(1)
				if tc.during == 1 {
					steerErr = w.Steer(context.Background(), steerInput("note", "one more note", tc.tools))
				}
				return "submitted", nil
			})
			tool, err := loop.WithToolOptions(tool, loop.ToolOptions{Terminal: true})
			if err != nil {
				t.Fatal(err)
			}
			model := &durableModel{responses: [][]ai.Token{calls("submit"), response("noted")}}
			model.onRequest = func(_ context.Context, index int) {
				if index == 0 && tc.during == 0 {
					steerErr = w.Steer(context.Background(), steerInput("note", "one more note", tc.tools))
				}
			}
			w = newDurableRun(t, agent.New(durableDefinition(model, tool)), &checkpointStore{})
			_, result := runEvents(t, w)
			if tc.wantErr == nil && steerErr != nil || tc.wantErr != nil && !errors.Is(steerErr, tc.wantErr) {
				t.Fatal("steer", steerErr)
			}
			if result.Status != agent.StatusSucceeded || effects.Load() != tc.wantEffects || model.count() != tc.wantCalls {
				t.Fatalf("status=%s effects=%d calls=%d errors=%v", result.Status, effects.Load(), model.count(), result.Errors)
			}
		})
	}
}

func TestSteerHonorsContextWhileStorageIsBusy(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		release := make(chan struct{})
		blocked := make(chan struct{})
		store := &checkpointStore{}
		var once sync.Once
		// The commit before the first model request is slow.
		store.hook = func(cp agent.Checkpoint) error {
			if cp.Revision == 2 {
				once.Do(func() {
					close(blocked)
					<-release
				})
			}
			return nil
		}
		model := &durableModel{responses: [][]ai.Token{response("answer")}}
		w := newDurableRun(t, agent.New(durableDefinition(model)), store)
		done := make(chan agent.WorkflowResult)
		go func() {
			_, result := runEvents(t, w)
			done <- result
		}()
		<-blocked
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if err := w.Steer(ctx, steerInput("slow", "x", agent.ContinuePendingTools)); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatal(err)
		}
		close(release)
		if result := <-done; result.Status != agent.StatusSucceeded || len(store.load().Commands) != 0 {
			t.Fatal(result.Status, store.load().Commands)
		}
	})
}
