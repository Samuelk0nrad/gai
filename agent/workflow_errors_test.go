package agent_test

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/lace-ai/gai/agent"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

type nonComparableWorkflowError []string

func (e nonComparableWorkflowError) Error() string { return e[0] }

func TestWorkflowPreservesIndependentStageFailures(t *testing.T) {
	sentinel := errors.New("failed")
	for _, tt := range []struct {
		name       string
		primaryErr error
		stageErrs  []error
	}{
		{name: "same message", stageErrs: []error{errors.New("failed"), errors.New("failed")}},
		{name: "wrapped sentinel", stageErrs: []error{fmt.Errorf("wrapped: %w", sentinel), sentinel}},
		{name: "shared sentinel", stageErrs: []error{sentinel, sentinel}},
		{name: "non-comparable", stageErrs: []error{nonComparableWorkflowError{"failed", "first"}, nonComparableWorkflowError{"failed", "second"}}},
		{name: "primary and middleware", primaryErr: errors.New("failed"), stageErrs: []error{errors.New("failed")}},
		{name: "non-comparable primary", primaryErr: nonComparableWorkflowError{"failed", "primary"}, stageErrs: []error{nonComparableWorkflowError{"failed", "middleware"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var middleware []agent.Middleware
			for index, stageErr := range tt.stageErrs {
				middleware = append(middleware, agent.NewAgentMiddleware(workflowAgent("failure", "unused"), agent.AgentMiddlewareConfig{
					// Identical names must not collapse distinct source indexes.
					Name:        "failure",
					ErrorPolicy: agent.PropagateError,
					ShouldRun:   func(agent.WorkflowResult) bool { return true },
					MapInput: func(_ context.Context, result agent.WorkflowResult) (agent.RunInput, error) {
						wantCount := index
						if tt.primaryErr != nil {
							wantCount++
						}
						if len(result.Errors) != wantCount {
							t.Errorf("upstream errors = %v, want %d failures", result.Errors, wantCount)
						}
						return agent.RunInput{}, stageErr
					},
				}))
			}
			// Pass each failure through another capture layer before finalization.
			middleware = append(middleware, agent.MiddlewareFunc(func(_ context.Context, _ *agent.MiddlewareContext, upstream <-chan agent.Event) <-chan agent.Event {
				return upstream
			}))
			main := workflowAgent("main", "answer", middleware...)
			var want []error
			if tt.primaryErr != nil {
				main = agent.New(agent.Definition{
					Name:       "main",
					Model:      &mocks.MockModel{Responses: []mocks.MockModelResponse{{Err: tt.primaryErr}}},
					Middleware: middleware,
					Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
						return &testPromptBuilder{}, nil
					},
				})
				want = append(want, tt.primaryErr)
			}
			want = append(want, tt.stageErrs...)
			workflow, err := main.NewRun(context.Background(), textRunInput("question"))
			if err != nil {
				t.Fatalf("NewRun failed: %v", err)
			}
			events := collectAgentEvents(workflow.RunEvents(context.Background()))
			result, waitErr := workflow.Wait()
			if !reflect.DeepEqual(result.Errors, want) {
				t.Errorf("Errors = %#v, want each independent failure once: %#v", result.Errors, want)
			}
			if len(result.Stages) != len(tt.stageErrs) {
				t.Errorf("stage count = %d, want %d", len(result.Stages), len(tt.stageErrs))
			}
			if len(events) == 0 || events[len(events)-1].Type != agent.EventError {
				t.Fatalf("events = %#v, want terminal error", events)
			}
			for _, terminalErr := range []error{waitErr, events[len(events)-1].Err} {
				joined, ok := terminalErr.(interface{ Unwrap() []error })
				if !ok || !reflect.DeepEqual(joined.Unwrap(), want) {
					t.Errorf("terminal error = %#v, want all failures: %#v", terminalErr, want)
				}
			}
		})
	}
}
