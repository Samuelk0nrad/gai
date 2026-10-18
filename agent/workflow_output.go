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
	if !delivered && eventAffectsOutputReduction(event) {
		w.mu.Lock()
		w.deliveryIncomplete = true
		w.mu.Unlock()
	}
	return delivered
}

func eventAffectsOutputReduction(event Event) bool {
	switch event.Type {
	case EventOutput, EventRetry, EventDiscard:
		return true
	case EventStageFinish:
		return event.AttemptID != 0 && (event.StageOutcome == StageFailed || event.StageOutcome == StageCanceled || event.StageOutcome == StagePaused)
	default:
		return false
	}
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

type stageErrorKey struct {
	kind    EventSourceKind
	attempt attemptKey
}

type stageError struct {
	key stageErrorKey
	err error
}

type workflowOutputAccumulator struct {
	outputs         []outputRecord
	invalid         map[attemptKey]struct{}
	stageErrs       []stageError
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
		if event.StageOutcome != StageFailed && event.StageOutcome != StageCanceled && event.StageOutcome != StagePaused {
			return
		}
		if event.AttemptID != 0 {
			a.invalidate(eventAttemptKey(event))
		}
		// mapPrimary records primary errors independently of event delivery.
		if event.StageOutcome == StageFailed && event.Err != nil && event.StageReason != stageReasonRecorded && event.Source.Kind != SourcePrimary {
			a.stageErrs = append(a.stageErrs, stageError{
				key: stageErrorKey{kind: event.Source.Kind, attempt: eventAttemptKey(event)},
				err: event.Err,
			})
		}
		if event.StageOutcome == StageCanceled && !a.canceled {
			a.canceled = true
			a.cancellationErr = event.Err
		}
	}
}

func (a *workflowOutputAccumulator) result() (accepted []OutputPart, attempted []OutputPart, stageErrs []stageError, canceled bool, cancellationErr error) {
	for _, output := range a.outputs {
		attempted = append(attempted, cloneOutputPart(output.part))
		if _, discarded := a.invalid[output.key]; !discarded {
			accepted = append(accepted, cloneOutputPart(output.part))
		}
	}
	return accepted, attempted, append([]stageError(nil), a.stageErrs...), a.canceled, a.cancellationErr
}

func (w *Workflow) captureMiddlewareOutput(ctx context.Context, upstream <-chan Event) <-chan Event {
	out := make(chan Event)
	w.middlewareDone.Add(1)
	go func() {
		defer w.middlewareDone.Done()
		defer close(out)
		var accumulator workflowOutputAccumulator
		deliver := true
		downstreamIncomplete := false
		for event := range upstream {
			accumulator.add(event)
			deliver = sendWorkflowEvent(ctx, out, cloneEvent(event), deliver)
			if !deliver && eventAffectsOutputReduction(event) {
				downstreamIncomplete = true
			}
		}
		output, _, stageErrs, canceled, cancellationErr := accumulator.result()
		w.mu.Lock()
		if !w.deliveryIncomplete {
			w.setVisibleOutputLocked(output)
		}
		if downstreamIncomplete {
			w.deliveryIncomplete = true
		}
		w.recordStageErrorsLocked(stageErrs)
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
		var sequence uint64
		stamp := func(event Event) Event {
			if w.durability != nil {
				runID, executionID, middlewareStageID := w.durability.eventIdentity(event.Source.Index)
				sequence++
				event.RunID = runID
				event.ExecutionID = executionID
				event.Sequence = sequence
				if event.Checkpoint != nil {
					event.CheckpointRevision = event.Checkpoint.Revision
				}
				if event.StageID == "" {
					event.StageID = w.stageID
				}
				if event.Source.Kind == SourceMiddleware {
					event.StageID = middlewareStageID
				}
			}
			return event
		}
		if w.durability != nil && w.durability.resumed && !w.durableChild {
			checkpoint := w.durability.resumeCheckpoint.Clone()
			deliver = sendWorkflowEvent(ctx, out, stamp(Event{Type: EventResumed, Source: EventSource{Kind: SourceWorkflow}, Checkpoint: &checkpoint}), deliver)
		}
		for event := range upstream {
			if event.Type == EventDone || event.Type == EventError || event.Type == EventCanceled {
				continue
			}
			accumulator.add(event)
			if !event.Replayed {
				deliver = sendWorkflowEvent(ctx, out, stamp(cloneEvent(event)), deliver)
			}
		}
		w.middlewareDone.Wait()
		<-w.primaryDone

		output, attempted, stageErrs, _, _ := accumulator.result()
		w.mu.Lock()
		if !w.deliveryIncomplete {
			w.setVisibleOutputLocked(output)
			w.result.AttemptedText = outputTextOnly(attempted)
		}
		w.recordStageErrorsLocked(stageErrs)
		terminal := Event{Type: EventDone, Source: EventSource{Kind: SourceWorkflow}}
		w.result.Status = StatusSucceeded
		if w.durability != nil && w.durability.current().Status == StatusPaused {
			w.result.Status = StatusPaused
		}
		switch {
		case w.result.Canceled:
			w.result.Status = StatusCanceled
			terminal.Type = EventCanceled
			terminal.Err = w.result.CancellationErr
			w.terminalErr = w.result.CancellationErr
		case len(w.result.Errors) > 0:
			w.result.Status = StatusFailed
			terminal.Type = EventError
			terminal.Err = errors.Join(w.result.Errors...)
			w.terminalErr = terminal.Err
		case w.result.Status == StatusPaused:
			terminal.Type = EventPaused
		}
		result := cloneWorkflowResult(w.result)
		result.Complete = true
		w.mu.Unlock()

		if w.durability != nil {
			if !w.durableChild && result.Status != StatusPaused {
				if err := w.durability.commit(context.WithoutCancel(ctx), func(cp *Checkpoint) { cp.Status = result.Status }); err != nil {
					terminal.Type = EventError
					terminal.Err = err
					w.mu.Lock()
					w.terminalErr = err
					w.result.Errors = append(w.result.Errors, err)
					w.result.Status = StatusFailed
					result = cloneWorkflowResult(w.result)
					result.Complete = true
					w.mu.Unlock()
				}
			}
			terminal.Checkpoint = w.durability.snapshot()
			result.Checkpoint = terminal.Checkpoint
		}
		obs.Finished(ctx, result)
		runObs.Finished(result)
		sendTerminalWorkflowEvent(ctx, out, stamp(terminal), deliver)
		w.mu.Lock()
		w.result.Complete = true
		close(out)
		w.mu.Unlock()
		close(w.done)
	}()
	return out
}

func (w *Workflow) recordStageErrorsLocked(errs []stageError) {
	if len(errs) > 0 && w.stageErrorsSeen == nil {
		w.stageErrorsSeen = make(map[stageErrorKey]struct{})
	}
	for _, failure := range errs {
		// Capture layers and finalization see the same stage-finish event.
		// Distinct stages must retain their failures even if errors share a
		// message, wrap each other, or use the same sentinel value.
		if _, seen := w.stageErrorsSeen[failure.key]; seen {
			continue
		}
		w.stageErrorsSeen[failure.key] = struct{}{}
		w.result.Errors = append(w.result.Errors, failure.err)
	}
}
