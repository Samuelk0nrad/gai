package agent

import (
	"context"
	"testing"
	"time"

	"github.com/lace-ai/gai"
)

func TestWorkflowPublishesCompleteWithTerminalStreamClosure(t *testing.T) {
	finishObserved := make(chan gai.Observation, 1)
	releaseObserver := make(chan struct{})
	sink := gai.ObservationSinkFunc(func(_ context.Context, observation gai.Observation) {
		if observation.Name != "agent_workflow_finished" {
			return
		}
		finishObserved <- observation
		<-releaseObserver
	})

	workflow := &Workflow{
		debug:       sink,
		done:        make(chan struct{}),
		primaryDone: make(chan struct{}),
	}
	close(workflow.primaryDone)

	ctx, observer := newWorkflowObserver(context.Background(), workflow)
	ctx, runObserver := newAgentRunObserver(ctx, workflow)
	upstream := make(chan Event, 1)
	upstream <- Event{
		Type:           EventOutput,
		Source:         EventSource{Kind: SourcePrimary},
		IterationCount: 1,
		AttemptID:      1,
		Output:         &OutputPart{Kind: OutputText, Text: "answer"},
	}
	close(upstream)

	stream := workflow.finalize(ctx, upstream, observer, runObserver)
	released := false
	defer func() {
		if !released {
			close(releaseObserver)
		}
		for range stream {
		}
		<-workflow.done
	}()

	var observation gai.Observation
	select {
	case observation = <-finishObserved:
	case <-time.After(time.Second):
		t.Fatal("workflow final observation was not emitted")
	}
	if complete, _ := observation.Fields["complete"].(bool); !complete {
		t.Fatalf("observer snapshot complete = %v, want true", observation.Fields["complete"])
	}
	if result := workflow.Result(); result.Complete {
		t.Fatalf("Result().Complete = true before terminal delivery and stream closure")
	}

	var events []Event
	func() {
		// Keep completion publication blocked after terminal delivery. Closing
		// the stream before acquiring the write lock would expose stale state.
		workflow.mu.RLock()
		defer workflow.mu.RUnlock()
		close(releaseObserver)
		released = true
		for range 2 {
			select {
			case event, ok := <-stream:
				if !ok {
					t.Fatal("stream closed before terminal delivery")
				}
				events = append(events, event)
			case <-time.After(time.Second):
				t.Fatal("terminal delivery waited for completion publication")
			}
		}
		select {
		case event, ok := <-stream:
			if ok {
				t.Errorf("unexpected event after terminal: %#v", event)
			} else if !workflow.result.Complete {
				t.Error("stream closed before completion was published")
			}
		case <-time.After(20 * time.Millisecond):
		}
	}()
	for event := range stream {
		events = append(events, event)
	}
	if len(events) != 2 || events[len(events)-1].Type != EventDone {
		t.Fatalf("events = %#v, want output followed by done", events)
	}
	if result := workflow.Result(); !result.Complete {
		t.Fatal("Result().Complete = false after terminal delivery and stream closure")
	}
	<-workflow.done
}
