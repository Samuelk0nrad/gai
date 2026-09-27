package agent

import (
	"context"
	"errors"
)

func sendWorkflowEvent(ctx context.Context, out chan<- Event, event Event, deliver bool) bool {
	if !deliver {
		return false
	}
	// Prefer a receiver that is already waiting, even when cancellation and the
	// send become ready together. This keeps internal pipeline stages draining
	// while allowing an abandoned public stream to be released by cancellation.
	select {
	case out <- event:
		return true
	default:
	}

	select {
	case out <- event:
		return true
	case <-ctx.Done():
		return false
	}
}

func (w *Workflow) sendEvent(ctx context.Context, out chan<- Event, event Event, deliver bool) bool {
	delivered := sendWorkflowEvent(ctx, out, event, deliver)
	if deliver && !delivered {
		w.mu.Lock()
		w.deliveryIncomplete = true
		w.mu.Unlock()
	}
	return delivered
}

func sendTerminalWorkflowEvent(ctx context.Context, out chan Event, event Event, deliver bool) {
	if sendWorkflowEvent(ctx, out, event, deliver) || ctx.Err() == nil {
		return
	}
	// The finalizer owns out and gives it one bounded slot. Cancellation may
	// evict one pending non-terminal event, but the terminal outcome is retained
	// for a delayed consumer without letting an abandoned stream block Wait.
	select {
	case <-out:
	default:
	}

	out <- event
}

type attemptKey struct {
	source    int
	iteration int
	attempt   int
}

func eventAttemptKey(event Event) attemptKey {
	return attemptKey{source: event.Source.Index, iteration: event.IterationCount, attempt: event.AttemptID}
}

type outputRecord struct {
	key  attemptKey
	part OutputPart
}

type workflowOutputAccumulator struct {
	outputs         []outputRecord
	invalid         map[attemptKey]struct{}
	stageErrs       []error
	canceled        bool
	cancellationErr error
}

func (a *workflowOutputAccumulator) invalidate(key attemptKey) {
	if a.invalid == nil {
		a.invalid = make(map[attemptKey]struct{})
	}
	a.invalid[key] = struct{}{}
}

func (a *workflowOutputAccumulator) add(event Event) {
	switch event.Type {
	case EventOutput:
		if event.Output != nil {
			a.outputs = append(a.outputs, outputRecord{
				key:  eventAttemptKey(event),
				part: cloneOutputPart(*event.Output),
			})
		}
	case EventRetry, EventDiscard:
		a.invalidate(eventAttemptKey(event))
	case EventStageFinish:
		if event.StageOutcome != StageFailed && event.StageOutcome != StageCanceled {
			return
		}
		if event.AttemptID != 0 {
			a.invalidate(eventAttemptKey(event))
		}
		if event.StageOutcome == StageFailed && event.Err != nil && event.StageReason != stageReasonRecorded {
			a.stageErrs = append(a.stageErrs, event.Err)
		}
		if event.StageOutcome == StageCanceled && !a.canceled {
			a.canceled = true
			a.cancellationErr = event.Err
		}
	}
}

func (a *workflowOutputAccumulator) result() (accepted []OutputPart, attempted []OutputPart, stageErrs []error, canceled bool, cancellationErr error) {
	for _, output := range a.outputs {
		attempted = append(attempted, cloneOutputPart(output.part))
		if _, discarded := a.invalid[output.key]; !discarded {
			accepted = append(accepted, cloneOutputPart(output.part))
		}
	}
	return accepted, attempted, append([]error(nil), a.stageErrs...), a.canceled, a.cancellationErr
}

func (w *Workflow) captureMiddlewareOutput(ctx context.Context, upstream <-chan Event) <-chan Event {
	out := make(chan Event)
	go func() {
		defer close(out)
		var accumulator workflowOutputAccumulator
		deliver := true
		for event := range upstream {
			accumulator.add(event)
			deliver = w.sendEvent(ctx, out, cloneEvent(event), deliver)
		}
		output, _, stageErrs, canceled, cancellationErr := accumulator.result()
		w.mu.Lock()
		if !w.deliveryIncomplete {
			w.setVisibleOutputLocked(output)
		}
		for _, err := range stageErrs {
			if err != nil && !containsError(w.result.Errors, err) {
				w.result.Errors = append(w.result.Errors, err)
			}
		}
		if canceled {
			w.result.Canceled = true
			w.result.CancellationErr = cancellationErr
		}
		w.mu.Unlock()
	}()
	return out
}

func (w *Workflow) finalize(ctx context.Context, upstream <-chan Event, obs *workflowObserver, runObs *agentRunObserver) <-chan Event {
	out := make(chan Event, 1)
	go func() {
		var accumulator workflowOutputAccumulator
		deliver := true
		for event := range upstream {
			if event.Type == EventDone || event.Type == EventError || event.Type == EventCanceled {
				continue
			}
			accumulator.add(event)
			deliver = w.sendEvent(ctx, out, cloneEvent(event), deliver)
		}
		<-w.primaryDone

		output, attempted, stageErrs, _, _ := accumulator.result()
		w.mu.Lock()
		if !w.deliveryIncomplete {
			w.setVisibleOutputLocked(output)
			w.result.AttemptedText = outputTextOnly(attempted)
		}
		for _, err := range stageErrs {
			if err != nil && !containsError(w.result.Errors, err) {
				w.result.Errors = append(w.result.Errors, err)
			}
		}
		terminal := Event{Type: EventDone, Source: EventSource{Kind: SourceWorkflow}}
		switch {
		case w.result.Canceled:
			terminal.Type = EventCanceled
			terminal.Err = w.result.CancellationErr
			w.terminalErr = w.result.CancellationErr
		case len(w.result.Errors) > 0:
			terminal.Type = EventError
			terminal.Err = errors.Join(w.result.Errors...)
			w.terminalErr = terminal.Err
		}
		w.result.Complete = true
		result := cloneWorkflowResult(w.result)
		w.mu.Unlock()

		obs.Finished(ctx, result)
		runObs.Finished(result)
		sendTerminalWorkflowEvent(ctx, out, terminal, deliver)
		close(out)
		close(w.done)
	}()
	return out
}

func containsError(errs []error, target error) bool {
	for _, err := range errs {
		if errors.Is(err, target) || (err != nil && target != nil && err.Error() == target.Error()) {
			return true
		}
	}
	return false
}
