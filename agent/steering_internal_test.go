package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/loop"
)

// A restored continuation whose loop has not started yet can be steered only
// while a model request that could see the steering is still allowed.
func TestSteeringIterationLeft(t *testing.T) {
	settings := &checkpointSettings{MaxIterations: 2}
	cases := []struct {
		name string
		c    loop.Continuation
		want bool
	}{
		{"ready before last request", loop.Continuation{Phase: loop.PhaseReady, NextIteration: 2}, true},
		{"interrupted last request is retried", loop.Continuation{Phase: loop.PhaseGenerating, NextIteration: 2}, true},
		{"done before last iteration", loop.Continuation{Phase: loop.PhaseDone, NextIteration: 2}, true},
		{"done after last iteration", loop.Continuation{Phase: loop.PhaseDone, NextIteration: 3}, false},
		{"plan before last iteration", loop.Continuation{Phase: loop.PhaseTools, NextIteration: 1, Pending: &loop.Generation{Iteration: 1}}, true},
		{"plan of last iteration", loop.Continuation{Phase: loop.PhaseTools, NextIteration: 2, Pending: &loop.Generation{Iteration: 2}}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := tc.c
			if got := steeringIterationLeft(StageCheckpoint{Continuation: &c, Settings: settings}); got != tc.want {
				t.Fatalf("got %v want %v", got, tc.want)
			}
		})
	}
	if steeringIterationLeft(StageCheckpoint{Settings: settings}) {
		t.Fatal("stage without continuation accepted steering")
	}
}

// A cancellation belongs to its command. Once that command's messages are
// applied, a later command that continues pending tools must not cancel the
// plan of a response that saw the first command.
func TestSteeringCancelEndsWithItsCommand(t *testing.T) {
	ctx := context.Background()
	var saved Checkpoint
	d := &checkpointCoordinator{
		state: Checkpoint{Version: CheckpointVersion, RunID: "run", ExecutionID: "execution", DefinitionRevision: "v1", Status: StatusRunning, ActiveStageID: PrimaryStageID,
			Stages: []StageCheckpoint{{ID: PrimaryStageID, Settings: &checkpointSettings{MaxIterations: 10}}}},
		durability:  Durability{DefinitionRevision: "v1", Commit: func(_ context.Context, _ uint64, next Checkpoint) error { saved = next; return nil }},
		pauseSignal: make(chan struct{}),
		steering:    newSteeringInbox(false, false),
	}
	user := func(text string) []ai.Message { return []ai.Message{{Role: ai.RoleUser, Parts: ai.TextParts(text)}} }
	command := func(id string) ContinuationCommand {
		return ContinuationCommand{ID: id, StageID: PrimaryStageID, Digest: id}
	}
	if err := d.commitPrimary(ctx, loop.Continuation{Phase: loop.PhaseGenerating, NextIteration: 1}); err != nil {
		t.Fatal(err)
	}
	if err := d.steer(ctx, command("a"), user("a"), true); err != nil {
		t.Fatal(err)
	}
	// The loop takes A at a boundary; B arrives while that boundary saves A.
	if taken := d.takeSteering(loop.SteeringOpen); len(taken.Messages) != 1 {
		t.Fatalf("taken %+v", taken)
	}
	if err := d.steer(ctx, command("b"), user("b"), false); err != nil {
		t.Fatal(err)
	}
	if err := d.commitPrimary(ctx, loop.Continuation{Phase: loop.PhaseReady, NextIteration: 2}); err != nil {
		t.Fatal(err)
	}
	if got := saved.Stages[0].Continuation.SteeringCommands; len(got) != 1 || got[0] != "b" {
		t.Fatalf("waiting commands %v", got)
	}
	d.takeSteering(loop.SteeringOpen)
	// The next response saw A and plans a call while B still waits.
	call := ai.ToolCall{Type: "function", ID: "call", Name: "write", Args: json.RawMessage(`{}`)}
	plan := loop.Continuation{Phase: loop.PhaseTools, NextIteration: 2, Steering: user("b"), SteeringCommands: []string{"b"},
		Pending: &loop.Generation{ID: "g", Iteration: 2, Invocations: []loop.InvocationRecord{{ID: "i", Call: call, Disposition: loop.InvocationPlanned}}}}
	if err := d.commitPrimary(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if inv := saved.Stages[0].Continuation.Pending.Invocations[0]; inv.CanceledBySteering || inv.Disposition != loop.InvocationPlanned {
		t.Fatalf("fresh call canceled by an applied command: %+v", inv)
	}
}
