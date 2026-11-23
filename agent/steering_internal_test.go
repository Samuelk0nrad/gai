package agent

import (
	"testing"

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
