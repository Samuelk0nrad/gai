package agent

import (
	"context"
	"errors"

	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/modelinput"
)

// Durable MapInput/ShouldRun callbacks must be pure. A crash before their output
// is committed can evaluate them again; after commit their output is reused.
func (m *AgentMiddleware) processDurable(ctx context.Context, run *MiddlewareContext, out chan<- Event, upstream []Event, result WorkflowResult, deliver bool) {
	d := run.workflow.durability
	source := run.Source()
	emit := func(event Event) { deliver = run.workflow.sendEvent(ctx, out, event, deliver) }
	forward := func(events []Event) {
		for _, event := range events {
			emit(event)
		}
	}
	fail := func(err error) {
		forward(upstream)
		emit(Event{Type: EventStageFinish, Source: source, StageOutcome: StageFailed, Err: err})
	}
	stageFailure := func(err error) {
		if commitErr := d.commit(ctx, func(cp *Checkpoint) {
			cp.ActiveStageID = m.config.ID
			changeStage(cp, m.config.ID, func(s *StageCheckpoint) {
				s.DecisionMade = true
				s.ShouldRun = true
				s.Complete = true
				s.Errors = []string{checkpointError(err)}
			})
		}); commitErr != nil {
			fail(commitErr)
			return
		}
		forward(upstream)
		run.workflow.addStage(StageResult{Name: m.name(), Output: m.config.Output, Result: AgentResult{Errors: []error{err}}})
		reason := ""
		if m.config.ErrorPolicy == RecordError {
			reason = stageReasonRecorded
		}
		emit(Event{Type: EventStageFinish, Source: source, StageOutcome: StageFailed, StageReason: reason, Err: err})
	}
	if result.Status == StatusPaused || d.current().Status == StatusPaused {
		forward(upstream)
		return
	}
	// An unfinished earlier stage owns the continuation boundary. Do not let
	// downstream skip decisions move that boundary away from recoverable work.
	for _, earlier := range d.current().Stages {
		if earlier.ID == m.config.ID {
			break
		}
		if !earlier.Complete {
			forward(upstream)
			return
		}
	}
	stage := d.stage(m.config.ID)
	if stage.Complete {
		if stage.Skipped {
			forward(upstream)
			emit(Event{Type: EventStageFinish, Source: source, StageOutcome: StageSkipped, Replayed: true})
			return
		}
		saved := checkpointAgentResult(stage)
		run.workflow.addStage(StageResult{Name: m.name(), Output: m.config.Output, Result: saved})
		if len(saved.Errors) > 0 || saved.Canceled {
			forward(upstream)
		} else if m.config.Output != PreserveOutput {
			forward(replayResultOutput(saved, source, stage.Continuation))
		}
		outcome := StageSucceeded
		var err error
		reason := ""
		if len(saved.Errors) > 0 {
			outcome = StageFailed
			err = errors.Join(saved.Errors...)
			if m.config.ErrorPolicy == RecordError {
				reason = stageReasonRecorded
			}
		}
		if saved.Canceled {
			outcome = StageCanceled
			err = saved.CancellationErr
		}
		emit(Event{Type: EventStageFinish, Source: source, StageOutcome: outcome, Err: err, StageReason: reason, Replayed: true})
		return
	}
	if d.pause.Load() {
		if err := d.commit(ctx, func(cp *Checkpoint) { cp.ActiveStageID = m.config.ID; cp.Status = StatusPaused }); err != nil {
			fail(err)
			return
		}
		run.workflow.mu.Lock()
		run.workflow.result.Status = StatusPaused
		run.workflow.mu.Unlock()
		forward(upstream)
		return
	}
	if !stage.DecisionMade {
		shouldRun := !result.Canceled && m.shouldRun(result)
		if err := d.commit(ctx, func(cp *Checkpoint) {
			cp.ActiveStageID = m.config.ID
			changeStage(cp, m.config.ID, func(s *StageCheckpoint) {
				s.DecisionMade = true
				s.ShouldRun = shouldRun
				s.Skipped = !shouldRun
				s.Complete = !shouldRun
			})
		}); err != nil {
			fail(err)
			return
		}
		stage = d.stage(m.config.ID)
	}
	if !stage.ShouldRun {
		forward(upstream)
		emit(Event{Type: EventStageFinish, Source: source, StageOutcome: StageSkipped})
		return
	}
	emit(Event{Type: EventStageStart, Source: source})
	var child *Workflow
	var err error
	if stage.Continuation != nil {
		var l *loop.Loop
		l, err = m.agent.restoreCheckpointLoop(stage)
		if err == nil {
			input := RunInput{ID: d.current().RunID, Meta: stage.Input.Meta, TraceContext: stage.Input.TraceContext}
			input.Prompt = stage.Continuation.Prompt.Input()
			input.Execution = restoreOverrides(stage.Input.ScalarExecution)
			child = newWorkflow(input, l, m.name(), m.agent.debugSink(), nil)
		}
	} else {
		var input RunInput
		input, err = m.input(ctx, result)
		if err == nil {
			err = validateDurableOverrides(input.Execution)
		}
		if err == nil {
			input.Meta, err = normalizeDurableMeta(input.Meta)
		}
		if err == nil {
			input.ID = d.current().RunID
			input.Durability = nil
			child, err = m.agent.NewRun(ctx, input)
		}
	}
	if err != nil {
		stageFailure(err)
		return
	}
	switch child.loop.PromptBuilder.(type) {
	case *modelinput.Builder, *compactingPromptBuilder:
	default:
		fail(ErrUnsupportedDurablePrompt)
		return
	}
	child.durability = d
	child.stageID = m.config.ID
	child.durableChild = true
	control := d.control(m.config.ID)
	settings := settingsForLoop(child.loop)
	input := checkpointInput{Meta: child.result.Input.Meta, TraceContext: child.result.Input.TraceContext, ScalarExecution: snapshotOverrides(child.result.Input.Execution)}
	control.Commit = func(ctx context.Context, c loop.Continuation) error {
		return d.commit(ctx, func(cp *Checkpoint) {
			cp.ActiveStageID = m.config.ID
			changeStage(cp, m.config.ID, func(s *StageCheckpoint) { s.Continuation = &c; s.Settings = settings; s.Input = input })
		})
	}
	child.loop.Continuation = control
	for event := range child.RunEvents(ctx) {
		switch event.Type {
		case EventStageStart, EventStageFinish, EventDone, EventError, EventCanceled, EventPaused, EventOutput:
			continue
		}
		event.Source = source
		event.StageID = m.config.ID
		emit(event)
	}
	nested, terminalErr := child.Wait()
	if nested.Status == StatusPaused {
		run.workflow.mu.Lock()
		run.workflow.result.Status = StatusPaused
		run.workflow.mu.Unlock()
		forward(upstream)
		emit(Event{Type: EventStageFinish, Source: source, StageOutcome: StagePaused})
		return
	}
	if terminalErr != nil && errors.Is(terminalErr, loop.ErrCheckpointCommit) {
		fail(terminalErr)
		return
	}
	saved := nested.Primary
	recoveryRequired := false
	if c := d.stage(m.config.ID).Continuation; c != nil {
		recoveryRequired = c.ResolveRecovery(nil) != nil
	}
	complete := terminalErr == nil && !saved.Canceled && len(saved.Errors) == 0
	if m.config.ErrorPolicy == RecordError && !saved.Canceled && !recoveryRequired {
		complete = true
	}
	// Persist the final stage/output decision before downstream stages begin.
	if err := d.commit(ctx, func(cp *Checkpoint) { changeStage(cp, m.config.ID, func(s *StageCheckpoint) { s.Complete = complete }) }); err != nil {
		fail(err)
		return
	}
	run.workflow.addStage(StageResult{Name: m.name(), Output: m.config.Output, Result: saved})
	outcome := StageSucceeded
	reason := ""
	if terminalErr != nil || saved.Canceled || len(saved.Errors) > 0 {
		forward(upstream)
		outcome = StageFailed
		if terminalErr == nil {
			terminalErr = errors.Join(saved.Errors...)
		}
		if m.config.ErrorPolicy == RecordError && !recoveryRequired {
			reason = stageReasonRecorded
		}
		if saved.Canceled {
			outcome = StageCanceled
			terminalErr = saved.CancellationErr
		}
	} else if m.config.Output != PreserveOutput {
		events := child.canonicalPrimaryOutput()
		for i := range events {
			events[i].Source = source
			events[i].StageID = m.config.ID
			events[i].Replayed = false
		}
		forward(events)
	}
	emit(Event{Type: EventStageFinish, Source: source, StageOutcome: outcome, StageReason: reason, Err: terminalErr})
}
