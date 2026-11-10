package agent

import (
	"context"
	"errors"

	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/modelinput"
)

// stageReasonUpstreamUnfinished marks a durable stage that waits for an earlier
// failed, canceled or paused stage to be resumed. No decision is saved for it.
const stageReasonUpstreamUnfinished = "upstream_unfinished"

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
	// The observation and StageStart event begin only for work in this execution,
	// not for stages restored from the checkpoint.
	var obs *middlewareObserver
	stageCtx := ctx
	start := func() {
		stageCtx, obs = newMiddlewareObserver(ctx, run, m, result)
		obs.Started(stageCtx)
		emit(Event{Type: EventStageStart, Source: source})
	}
	fail := func(err error) {
		if obs == nil {
			start()
		}
		forward(upstream)
		obs.Finished(stageCtx, AgentResult{Errors: []error{err}}, false)
		emit(Event{Type: EventStageFinish, Source: source, StageOutcome: StageFailed, Err: err})
	}
	stageFailure := func(err error) {
		if commitErr := d.commit(context.WithoutCancel(ctx), func(cp *Checkpoint) {
			cp.ActiveStageID = m.config.ID
			changeStage(cp, m.config.ID, func(s *StageCheckpoint) {
				s.DecisionMade = true
				s.ShouldRun = true
				s.Complete = m.config.ErrorPolicy == RecordError
				s.Errors = []string{checkpointError(err)}
			})
		}); commitErr != nil {
			fail(commitErr)
			return
		}
		forward(upstream)
		failed := AgentResult{Errors: []error{err}}
		run.workflow.addStage(StageResult{Name: m.name(), Output: m.config.Output, Result: failed})
		obs.Finished(stageCtx, failed, false)
		reason := stageReasonPropagated
		if m.config.ErrorPolicy == RecordError {
			reason = stageReasonRecorded
		}
		emit(Event{Type: EventStageFinish, Source: source, StageOutcome: StageFailed, StageReason: reason, Err: err})
	}
	// An unfinished earlier stage owns the continuation boundary. This covers a
	// failed, canceled or paused earlier stage. This stage is decided only after
	// that stage is resumed and completes, so ShouldRun is not called and no
	// decision is saved.
	if !d.earlierStagesComplete(m.config.ID) {
		start()
		forward(upstream)
		obs.Skipped(stageCtx, stageReasonUpstreamUnfinished)
		emit(Event{Type: EventStageFinish, Source: source, StageOutcome: StageSkipped, StageReason: stageReasonUpstreamUnfinished})
		return
	}
	if result.Status == StatusPaused || d.status() == StatusPaused {
		forward(upstream)
		return
	}
	stage := d.stage(m.config.ID)
	if stage.Complete {
		if stage.Skipped {
			forward(upstream)
			emit(Event{Type: EventStageFinish, Source: source, StageOutcome: StageSkipped, replayed: true})
			return
		}
		// A completed stage either succeeded or kept its error via RecordError;
		// Validate rejects completed canceled stages.
		saved := checkpointAgentResult(stage)
		run.workflow.addStage(StageResult{Name: m.name(), Output: m.config.Output, Result: saved})
		outcome := StageSucceeded
		var err error
		reason := ""
		if len(saved.Errors) > 0 {
			forward(upstream)
			outcome = StageFailed
			err = errors.Join(saved.Errors...)
			reason = stageReasonRecorded
		} else if m.config.Output != PreserveOutput {
			forward(replayResultOutput(saved, source, stage.Continuation))
		}
		emit(Event{Type: EventStageFinish, Source: source, StageOutcome: outcome, Err: err, StageReason: reason, replayed: true})
		return
	}
	// A pause request does not stop here. The stage is decided and its prompt is
	// saved; its loop then pauses at its first boundary, so a resume command has
	// a continuation to apply to.
	if !stage.DecisionMade {
		// Every earlier stage has completed, so the upstream result is neither
		// canceled nor failed here.
		shouldRun := m.shouldRun(result)
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
	start()
	if !stage.ShouldRun {
		forward(upstream)
		obs.Skipped(stageCtx, "predicate")
		emit(Event{Type: EventStageFinish, Source: source, StageOutcome: StageSkipped, StageReason: "predicate"})
		return
	}
	var child *Workflow
	var err error
	if stage.Continuation != nil {
		var l *loop.Loop
		l, err = m.agent.restoreCheckpointLoop(stage)
		if err == nil {
			input := RunInput{ID: d.runID(), Meta: stage.Input.Meta, TraceContext: stage.Input.TraceContext}
			input.Prompt = stage.Continuation.Prompt.Input()
			input.Execution = restoreOverrides(stage.Input.ScalarExecution)
			child = newWorkflow(input, l, m.name(), m.agent.debugSink(), nil)
		}
	} else {
		var input RunInput
		input, err = m.input(stageCtx, result)
		if err == nil {
			err = validateDurableOverrides(input.Execution)
		}
		if err == nil {
			input.Meta, err = normalizeDurableMeta(input.Meta)
		}
		if err == nil {
			input.ID = d.runID()
			input.Durability = nil
			child, err = m.agent.NewRun(stageCtx, input)
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
	input := checkpointInput{Meta: cloneJSONObject(child.result.Input.Meta), TraceContext: cloneTraceContext(child.result.Input.TraceContext), ScalarExecution: snapshotOverrides(child.result.Input.Execution)}
	control.Commit = func(ctx context.Context, c loop.Continuation) error {
		return d.commit(ctx, func(cp *Checkpoint) {
			cp.ActiveStageID = m.config.ID
			changeStage(cp, m.config.ID, func(s *StageCheckpoint) { s.Continuation = &c; s.Settings = settings; s.Input = input })
		})
	}
	child.loop.Continuation = control
	for event := range child.RunEvents(stageCtx) {
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
		obs.Paused(stageCtx)
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
	if err := d.commit(context.WithoutCancel(ctx), func(cp *Checkpoint) { changeStage(cp, m.config.ID, func(s *StageCheckpoint) { s.Complete = complete }) }); err != nil {
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
		reason = stageReasonPropagated
		if m.config.ErrorPolicy == RecordError && !recoveryRequired {
			reason = stageReasonRecorded
		}
		if saved.Canceled {
			outcome = StageCanceled
			reason = "canceled"
			terminalErr = saved.CancellationErr
		}
		observed := saved
		if len(observed.Errors) == 0 && terminalErr != nil {
			observed.Errors = []error{terminalErr}
		}
		obs.Finished(stageCtx, observed, false)
	} else {
		if m.config.Output != PreserveOutput {
			events := child.canonicalPrimaryOutput()
			for i := range events {
				events[i].Source = source
				events[i].StageID = m.config.ID
				events[i].replayed = false
			}
			forward(events)
		}
		obs.Finished(stageCtx, saved, m.config.Output != PreserveOutput)
	}
	emit(Event{Type: EventStageFinish, Source: source, StageOutcome: outcome, StageReason: reason, Err: terminalErr})
}
