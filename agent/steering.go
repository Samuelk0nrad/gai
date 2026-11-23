package agent

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/loop"
	"github.com/Samuelk0nrad/gai/modelinput"
)

// ErrInvalidCommand reports a steering command without a command ID, with an
// unknown PendingToolAction, or whose rendered messages fail ai.Message.Validate.
// Errors from rendering the prompt itself are returned as they are.
var ErrInvalidCommand = errors.New("invalid continuation command")

// SteerInput is one idempotent steering command for a running workflow.
// Context parts must be pure and deterministic, as for ResumeInput: they are
// rendered before the command is compared with an accepted one.
type SteerInput struct {
	CommandID    string
	Prompt       modelinput.PromptInput
	PendingTools PendingToolAction
}

// Steer queues input for the primary agent of a running durable workflow
// without pausing it. It returns once the command is committed through
// Durability.Commit, so a later Resume from the checkpoint applies it even if
// this execution stops first. The running loop applies it at its next
// continuation boundary, as Resume applies steering: after the results of the
// accepted plan. Commands taken at one boundary become one generation, in the
// order they were accepted. Input that arrives while the final answer is
// generated continues the run, and so does input accepted before the agent
// chose a terminal tool batch.
//
// CancelUnstartedTools resolves accepted calls that have not started with the
// steering_canceled result. It also covers the plan of a model response that
// was being generated when the command was accepted, because that response
// did not see the steering. Calls that already started finish normally.
//
// Redelivering an accepted CommandID returns nil, also after the workflow has
// finished; a different payload under the same ID returns ErrCommandConflict.
// Steer returns ErrWorkflowNotStarted before Run or RunEvents, waits for the
// first checkpoint commit of a new run, and returns an error matching
// ErrWorkflowFinished once the primary agent could no longer apply it: it has
// given its final answer, accepted a terminal tool batch with no steering
// waiting, started its last allowed model request, or its loop has ended. A
// command that a paused, failed or canceled loop accepted just before it ended
// stays saved for Resume. Start or resume a run with new input instead.
// Workflows without Durability return ErrDurability. ctx bounds the waits
// before the command is saved; the storage write itself ignores its
// cancellation. Commit callbacks must not call Steer.
func (w *Workflow) Steer(ctx context.Context, input SteerInput) error {
	if w == nil || w.durability == nil || w.durability.steering == nil || w.durableChild {
		return ErrDurability
	}
	if input.CommandID == "" || input.PendingTools != ContinuePendingTools && input.PendingTools != CancelUnstartedTools {
		return ErrInvalidCommand
	}
	w.mu.RLock()
	started := w.started
	w.mu.RUnlock()
	if !started {
		return ErrWorkflowNotStarted
	}
	messages, err := modelinput.MaterializeInput(ctx, input.Prompt)
	if err != nil {
		return err
	}
	// Coordinator commits do not validate continuations again, so invalid
	// input must not reach the checkpoint.
	for _, m := range messages {
		if err := m.Validate(); err != nil {
			return fmt.Errorf("%w: %w", ErrInvalidCommand, err)
		}
	}
	digest, err := commandDigest(PrimaryStageID, messages, ResumeInput{PendingTools: input.PendingTools})
	if err != nil {
		return err
	}
	command := ContinuationCommand{ID: input.CommandID, StageID: PrimaryStageID, Digest: digest}
	return w.durability.steer(ctx, command, messages, input.PendingTools == CancelUnstartedTools)
}

// steeringInbox holds steering that is saved in the primary continuation but
// not yet taken by the running loop. Every primary loop commit adds it to the
// continuation it saves, because each loop commit replaces the stage
// continuation. All fields except the notices are guarded by the
// coordinator's commitMu.
type steeringInbox struct {
	messages []ai.Message
	commands []string
	// canceled holds records that steering canceled on the loop's behalf.
	canceled []loop.InvocationRecord
	// cancels lists CancelUnstartedTools commands whose messages still wait
	// to be applied, so a plan from a response that did not see them is
	// canceled. A command without messages stays until the next loop commit.
	cancels []string
	closed  bool
	// wake tells a loop waiting to admit tools that steering canceled calls.
	wake chan struct{}
	// ready closes after the first acknowledged commit, or when the inbox
	// closes first.
	ready     chan struct{}
	readyOnce sync.Once

	noticeMu sync.Mutex
	notices  []steeringNotice
	notify   chan struct{}
}

// steeringNotice is an accepted command waiting for its EventSteeringAccepted.
type steeringNotice struct {
	command  string
	revision uint64
}

// newSteeringInbox returns an open inbox. A checkpoint that is already
// acknowledged, as on Resume, makes it ready at once.
func newSteeringInbox(acknowledged, closed bool) *steeringInbox {
	s := &steeringInbox{ready: make(chan struct{}), notify: make(chan struct{}, 1), wake: make(chan struct{}, 1)}
	if acknowledged {
		s.markReady()
	}
	if closed {
		s.close()
	}
	return s
}

func (s *steeringInbox) markReady() { s.readyOnce.Do(func() { close(s.ready) }) }

// close stops accepting steering and releases callers waiting for readiness.
func (s *steeringInbox) close() {
	s.closed = true
	s.markReady()
}

// notice queues an accepted event and wakes the primary event mapper.
func (s *steeringInbox) notice(command string, revision uint64) {
	s.noticeMu.Lock()
	s.notices = append(s.notices, steeringNotice{command: command, revision: revision})
	s.noticeMu.Unlock()
	select {
	case s.notify <- struct{}{}:
	default:
	}
}

// takeNotices returns the accepted events waiting for delivery.
func (s *steeringInbox) takeNotices() []steeringNotice {
	s.noticeMu.Lock()
	defer s.noticeMu.Unlock()
	notices := s.notices
	s.notices = nil
	return notices
}

// merge adds the untaken steering to a loop snapshot, which the loop copied
// for this commit and does not keep. It returns the records it newly canceled;
// the inbox records them only once the commit is acknowledged.
func (s *steeringInbox) merge(c *loop.Continuation) []loop.InvocationRecord {
	c.Steering = append(c.Steering, s.messages...)
	c.SteeringCommands = append(c.SteeringCommands, s.commands...)
	if c.Pending == nil {
		return nil
	}
	// An acknowledged cancellation is never undone. This replaces a dispatch
	// intent the loop prepared before it took the steering, and the outcome of
	// a call refused without a handler.
	for _, saved := range s.canceled {
		for i := range c.Pending.Invocations {
			if c.Pending.Invocations[i].ID == saved.ID {
				c.Pending.Invocations[i] = saved
			}
		}
	}
	if len(s.cancels) > 0 {
		return c.CancelPlanned()
	}
	return nil
}

// steer commits one command into the primary continuation and hands it to the
// running loop.
func (d *checkpointCoordinator) steer(ctx context.Context, command ContinuationCommand, messages []ai.Message, cancel bool) error {
	s := d.steering
	select {
	case <-s.ready:
	case <-ctx.Done():
		return ctx.Err()
	}
	// Storage may be slow for another commit; the caller's ctx bounds the wait.
	if err := d.commitMu.Lock(ctx); err != nil {
		return err
	}
	defer d.commitMu.Unlock()
	// State changes only under commitMu, which this goroutine holds.
	state := d.state
	for _, previous := range state.Commands {
		if previous.ID == command.ID {
			if previous != command {
				return ErrCommandConflict
			}
			return nil
		}
	}
	if s.closed || !d.acknowledged || state.Stages[0].Complete || !steeringIterationLeft(state.Stages[0]) {
		return fmt.Errorf("%w: the primary agent no longer accepts steering", ErrWorkflowFinished)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	var canceled []loop.InvocationRecord
	// A write the caller abandons could still land, and the loop's next
	// compare-and-swap would then fail the run, so storage gets no caller
	// cancellation; Commit enforces its own timeout.
	err := d.commitLocked(context.WithoutCancel(ctx), func(cp *Checkpoint) {
		cp.Commands = append(slices.Clone(cp.Commands), command)
		changeStage(cp, PrimaryStageID, func(stage *StageCheckpoint) {
			c := *stage.Continuation
			c.Steering = append(slices.Clone(c.Steering), messages...)
			if len(messages) > 0 {
				c.SteeringCommands = append(slices.Clone(c.SteeringCommands), command.ID)
			}
			if cancel && c.Pending != nil {
				pending := *c.Pending
				pending.Invocations = slices.Clone(pending.Invocations)
				c.Pending = &pending
				canceled = c.CancelPlanned()
			}
			stage.Continuation = &c
		})
	})
	if err != nil {
		return err
	}
	s.messages = append(s.messages, messages...)
	if len(messages) > 0 {
		s.commands = append(s.commands, command.ID)
	}
	s.canceled = append(s.canceled, canceled...)
	if cancel {
		s.cancels = append(s.cancels, command.ID)
	}
	if len(canceled) > 0 {
		select {
		case s.wake <- struct{}{}:
		default:
		}
	}
	// Queued before commitMu is released, so the accepted event precedes the
	// event that reports the steering applied.
	s.notice(command.ID, d.state.Revision)
	return nil
}

// steeringIterationLeft reports whether a model request that could see new
// steering is still allowed. The running loop stops steering before its last
// request; this covers a restored continuation whose loop has not started.
func steeringIterationLeft(stage StageCheckpoint) bool {
	c, settings := stage.Continuation, stage.Settings
	if c == nil || settings == nil {
		return false
	}
	next := c.NextIteration
	if c.Pending != nil {
		next = c.Pending.Iteration + 1
	}
	return next <= settings.MaxIterations
}

// commitPrimary saves a primary loop snapshot together with the steering the
// loop has not taken yet.
func (d *checkpointCoordinator) commitPrimary(ctx context.Context, c loop.Continuation) error {
	d.lockCommits()
	defer d.commitMu.Unlock()
	s := d.steering
	canceled := s.merge(&c)
	if err := d.commitLocked(ctx, func(cp *Checkpoint) {
		cp.ActiveStageID = PrimaryStageID
		changeStage(cp, PrimaryStageID, func(stage *StageCheckpoint) { stage.Continuation = &c })
	}); err != nil {
		return err
	}
	d.primaryRevision.Store(d.state.Revision)
	s.canceled = append(s.canceled, canceled...)
	// A cancellation ends once its messages are applied; each command keeps
	// its own, so a later command that continues pending tools is not
	// affected by an earlier one.
	s.cancels = slices.DeleteFunc(s.cancels, func(id string) bool { return !slices.Contains(c.SteeringCommands, id) })
	// The final answer is saved and nothing waits, so the loop finishes.
	// Closing here, under the same lock as Steer, means no steering can be
	// accepted for a stage that is about to complete.
	if len(c.Steering) == 0 && c.Phase == loop.PhaseDone {
		s.close()
	}
	return nil
}

// takeSteering hands the untaken steering to the primary loop, and stops
// accepting steering that the loop could no longer apply.
func (d *checkpointCoordinator) takeSteering(take loop.SteeringTake) loop.Steering {
	d.lockCommits()
	defer d.commitMu.Unlock()
	s := d.steering
	switch take {
	case loop.SteeringLast:
		s.close()
	case loop.SteeringFinishing:
		// Waiting messages or a pending cancellation keep the loop running
		// after its terminal batch, so it can still apply new steering.
		if len(s.messages) == 0 && len(s.cancels) == 0 {
			s.close()
		}
	}
	taken := loop.Steering{Messages: s.messages, Commands: s.commands, Canceled: s.canceled}
	s.messages, s.commands, s.canceled = nil, nil, nil
	return taken
}

// closeSteering stops accepting steering once the primary loop has ended.
// Steering it did not take is saved in the checkpoint for Resume.
func (d *checkpointCoordinator) closeSteering() {
	if d.steering == nil {
		return
	}
	d.lockCommits()
	defer d.commitMu.Unlock()
	d.steering.close()
}
