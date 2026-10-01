package agent

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestAgentMiddlewareRunStagePreservesCanonicalOutputAfterCanceledForwarding(t *testing.T) {
	for _, test := range []struct {
		name   string
		policy OutputPolicy
	}{
		{name: "append", policy: AppendOutput},
		{name: "replace", policy: ReplaceOutput},
	} {
		t.Run(test.name, func(t *testing.T) {
			primaryFinished := make(chan struct{})
			workflowFinished := make(chan struct{})
			var primaryFinishedOnce sync.Once
			var workflowFinishedOnce sync.Once
			nested := New(Definition{
				Name: "nested",
				Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{{
					Res: ai.AIResponse{Message: ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "SAFE REPLACEMENT"}}}},
				}}},
				Prompt: func(context.Context, RunInput) (gaictx.PromptBuilder, error) {
					return gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}}), nil
				},
				Limits: Limits{MaxLoopIterations: 1},
				ObservationSink: gai.ObservationSinkFunc(func(_ context.Context, observation gai.Observation) {
					switch observation.Name {
					case "agent_primary_finished":
						primaryFinishedOnce.Do(func() { close(primaryFinished) })
					case "agent_workflow_finished":
						workflowFinishedOnce.Do(func() { close(workflowFinished) })
					}
				}),
			})
			middleware := NewAgentMiddleware(nested, AgentMiddlewareConfig{Output: test.policy})

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			forwardBlocked := make(chan struct{})
			releaseForward := make(chan struct{})
			var releaseOnce sync.Once
			release := func() { releaseOnce.Do(func() { close(releaseForward) }) }
			defer release()

			type outcome struct {
				result AgentResult
				output []Event
				err    error
			}
			outcomeCh := make(chan outcome, 1)
			go func() {
				result, output, err := middleware.runStage(ctx, RunInput{ID: "nested-test", Prompt: gaictx.PromptInput{User: ai.TextParts("")}}, EventSource{Kind: SourceMiddleware, Index: 1, Name: "nested"}, func(event Event) {
					if event.Type == EventAttemptStart {
						close(forwardBlocked)
						<-releaseForward
					}
				})
				outcomeCh <- outcome{result: result, output: output, err: err}
			}()

			select {
			case <-forwardBlocked:
			case <-time.After(time.Second):
				t.Fatal("nested forwarding did not reach the backpressure point")
			}
			select {
			case <-primaryFinished:
			case <-time.After(time.Second):
				t.Fatal("nested primary did not finish while forwarding was blocked")
			}

			cancel()
			select {
			case <-workflowFinished:
			case <-time.After(time.Second):
				t.Fatal("nested workflow did not finalize after cancellation")
			}
			// Finished observation occurs immediately before terminal retention.
			// Keep forwarding blocked until the pending output has been evicted.
			time.Sleep(10 * time.Millisecond)
			release()

			var got outcome
			select {
			case got = <-outcomeCh:
			case <-time.After(time.Second):
				t.Fatal("runStage did not return after forwarding was released")
			}
			if got.err != nil {
				t.Fatalf("runStage error = %v", got.err)
			}
			if got.result.Text != "SAFE REPLACEMENT" {
				t.Fatalf("stage result text = %q, want SAFE REPLACEMENT", got.result.Text)
			}
			if len(got.output) != 1 || got.output[0].Output == nil || got.output[0].Output.Text != "SAFE REPLACEMENT" {
				t.Fatalf("accepted nested output = %#v", got.output)
			}
			if got.output[0].Source.Kind != SourceMiddleware || got.output[0].Source.Index != 1 || got.output[0].IterationCount != 1 || got.output[0].AttemptID != 1 {
				t.Fatalf("nested output lost source or attempt identity: %#v", got.output[0])
			}
		})
	}
}

func TestPrimaryAccumulatorCanonicalOutputExcludesUnacceptedAttempts(t *testing.T) {
	source := EventSource{Kind: SourcePrimary}
	var accumulator primaryAccumulator
	for _, event := range []Event{
		{Type: EventOutput, Source: source, IterationCount: 1, AttemptID: 1, Output: &OutputPart{Kind: OutputText, Text: "retried"}},
		{Type: EventRetry, Source: source, IterationCount: 1, AttemptID: 1},
		{Type: EventOutput, Source: source, IterationCount: 1, AttemptID: 2, Output: &OutputPart{Kind: OutputText, Text: "accepted"}},
		{Type: EventIterationDone, Source: source, IterationCount: 1, AttemptID: 2},
		{Type: EventOutput, Source: source, IterationCount: 2, AttemptID: 3, Output: &OutputPart{Kind: OutputText, Text: "discarded"}},
		{Type: EventDiscard, Source: source, IterationCount: 2, AttemptID: 3},
	} {
		accumulator.recordOutput(event)
	}

	output := accumulator.acceptedOutput()
	if len(output) != 1 || output[0].Output == nil || output[0].Output.Text != "accepted" {
		t.Fatalf("canonical output = %#v, want only the accepted attempt", output)
	}
	if output[0].IterationCount != 1 || output[0].AttemptID != 2 {
		t.Fatalf("canonical output lost attempt identity: %#v", output[0])
	}
}
