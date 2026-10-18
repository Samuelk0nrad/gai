package agent

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/modelinput"
)

// CheckpointVersion identifies the workflow snapshot schema accepted by Resume.
const CheckpointVersion = 1

// PrimaryStageID is the stable identity of the workflow's primary agent.
const PrimaryStageID = "primary"

// WorkflowStatus distinguishes a finished execution from the logical run outcome.
type WorkflowStatus string

const (
	StatusRunning   WorkflowStatus = "running"
	StatusSucceeded WorkflowStatus = "succeeded"
	StatusPaused    WorkflowStatus = "paused"
	StatusFailed    WorkflowStatus = "failed"
	StatusCanceled  WorkflowStatus = "canceled"
)

var (
	ErrDurability                   = errors.New("invalid durability configuration")
	ErrCheckpoint                   = errors.New("invalid workflow checkpoint")
	ErrDefinitionRevision           = errors.New("incompatible workflow definition revision")
	ErrUnsupportedDurableMiddleware = errors.New("durable workflows require AgentMiddleware with a unique ID")
	ErrUnsupportedDurablePrompt     = errors.New("durable workflows require the standard modelinput.Builder")
	ErrCommandConflict              = errors.New("continuation command conflicts with a previously accepted command")
	ErrWorkflowFinished             = errors.New("workflow execution has finished")
)

// Durability opts into acknowledged checkpoint boundaries. Returning nil from
// Commit promises that exactly next was durably saved by compare-and-swap from
// expectedRevision. On an ambiguous acknowledgement, the application must reload
// and compare the proposed revision and content before returning nil. An error
// stops admission; GAI does not retry the callback or a side-effecting tool.
// Storage, worker leases/fencing, cancellation intent and external idempotency
// remain application-owned. Commit may call Result or RequestPause.
// Settled outcomes and terminal status use a context that retains values but
// detaches execution cancellation. Commit must enforce its own storage timeout.
type Durability struct {
	DefinitionRevision string
	Commit             func(context.Context, uint64, Checkpoint) error
	// DeferredApprovals explicitly selects persisted approvals instead of the
	// in-process resolver. A missing resolver otherwise retains refusal behavior.
	DeferredApprovals bool
}

// PendingToolAction selects how steering handles undispatched accepted calls.
type PendingToolAction string

const (
	ContinuePendingTools PendingToolAction = ""
	CancelUnstartedTools PendingToolAction = "cancel_unstarted"
)

// ResumeInput applies one optional idempotent command. Context parts must be
// pure and deterministic: they are rendered before the ownership claim, including
// on redelivery to compare the command with its saved materialized payload.
type ResumeInput struct {
	CommandID       string
	StageID         string
	Prompt          modelinput.PromptInput
	PendingTools    PendingToolAction
	Approvals       []loop.ToolApprovalDecision
	ToolResolutions []loop.ToolRecoveryResolution
}

// ContinuationCommand contains a materialized command, including both its
// context and user messages. Exact redelivery never reapplies the command.
type ContinuationCommand struct {
	ID              string                        `json:"id"`
	StageID         string                        `json:"stage_id"`
	Messages        []ai.Message                  `json:"messages,omitempty"`
	PendingTools    PendingToolAction             `json:"pending_tools,omitempty"`
	Approvals       []loop.ToolApprovalDecision   `json:"approvals,omitempty"`
	ToolResolutions []loop.ToolRecoveryResolution `json:"tool_resolutions,omitempty"`
}

// StageCheckpoint uses a registration ID, never EventSource.Index or a display
// name, as its durable identity. Completed stages retain their output decision.
type StageCheckpoint struct {
	ID           string              `json:"id"`
	Name         string              `json:"name"`
	Output       OutputPolicy        `json:"output"`
	DecisionMade bool                `json:"decision_made"`
	ShouldRun    bool                `json:"should_run"`
	Complete     bool                `json:"complete"`
	Skipped      bool                `json:"skipped"`
	Continuation *loop.Continuation  `json:"continuation,omitempty"`
	Settings     *checkpointSettings `json:"settings,omitempty"`
	Input        checkpointInput     `json:"input"`
	Errors       []string            `json:"errors,omitempty"`
	Canceled     bool                `json:"canceled,omitempty"`
}
type checkpointInput struct {
	ScalarExecution *checkpointOverrides `json:"execution,omitempty"`
	Meta            map[string]any       `json:"meta,omitempty"`
	TraceContext    *gai.TraceContext    `json:"trace_context,omitempty"`
}

// Checkpoint is a versioned semantic snapshot. It contains no runtime objects,
// functions, token estimates, or raw pre-redaction tool results. Treat storage as
// sensitive: materialized model messages can contain application/user content.
type Checkpoint struct {
	Version            int                   `json:"version"`
	RunID              string                `json:"run_id"`
	ExecutionID        string                `json:"execution_id"`
	Revision           uint64                `json:"revision"`
	DefinitionRevision string                `json:"definition_revision"`
	DeferredApprovals  bool                  `json:"deferred_approvals"`
	Status             WorkflowStatus        `json:"status"`
	ActiveStageID      string                `json:"active_stage_id"`
	Stages             []StageCheckpoint     `json:"stages"`
	Commands           []ContinuationCommand `json:"commands,omitempty"`
}

// UnmarshalJSON retains lossless JSON numbers in application metadata. Durable
// metadata uses JSON types (including json.Number), consistently from NewRun.
func (c *Checkpoint) UnmarshalJSON(data []byte) error {
	type wireCheckpoint Checkpoint
	var decoded wireCheckpoint
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&decoded); err != nil {
		return err
	}
	*c = Checkpoint(decoded)
	return nil
}

// normalizeDurableMeta makes initial and restored inputs share lossless JSON types.
func normalizeDurableMeta(meta map[string]any) (map[string]any, error) {
	data, err := json.Marshal(meta)
	if err != nil {
		return nil, fmt.Errorf("%w: metadata: %w", ErrCheckpoint, err)
	}
	var normalized map[string]any
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&normalized); err != nil {
		return nil, err
	}
	return normalized, nil
}

// Clone detaches a validated checkpoint for callback or caller ownership.
func (c Checkpoint) Clone() Checkpoint {
	data, _ := json.Marshal(c)
	var copy Checkpoint
	_ = json.Unmarshal(data, &copy)
	return copy
}

// Validate rejects incompatible and structurally invalid saved state before any
// resume side effects. Checkpoints must come from trusted application storage.
func (c Checkpoint) Validate() error {
	if _, err := json.Marshal(c); err != nil {
		return fmt.Errorf("%w: %w", ErrCheckpoint, err)
	}
	if c.Version != CheckpointVersion || c.RunID == "" || c.ExecutionID == "" || c.DefinitionRevision == "" || c.Revision == 0 || len(c.Stages) == 0 || c.Stages[0].ID != PrimaryStageID {
		return ErrCheckpoint
	}
	if c.Status != StatusRunning && c.Status != StatusPaused && c.Status != StatusFailed && c.Status != StatusCanceled && c.Status != StatusSucceeded {
		return ErrCheckpoint
	}
	seen := map[string]bool{}
	active := false
	for _, s := range c.Stages {
		if s.ID == "" || seen[s.ID] || s.Output > ReplaceOutput {
			return ErrCheckpoint
		}
		seen[s.ID] = true
		if s.ID == c.ActiveStageID {
			active = true
		}
		if s.Skipped && (!s.Complete || s.ShouldRun) {
			return ErrCheckpoint
		}
		if s.Continuation != nil {
			if err := s.Continuation.Validate(); err != nil {
				return fmt.Errorf("%w: %w", ErrCheckpoint, err)
			}
		}
		if s.Complete && !s.Skipped && s.Continuation == nil && len(s.Errors) == 0 {
			return ErrCheckpoint
		}
	}
	if !active {
		return ErrCheckpoint
	}
	commandIDs := map[string]bool{}
	for _, command := range c.Commands {
		if command.ID == "" || commandIDs[command.ID] || !seen[command.StageID] {
			return ErrCheckpoint
		}
		commandIDs[command.ID] = true
	}
	return nil
}

type checkpointCoordinator struct {
	mu               sync.RWMutex
	commitMu         sync.Mutex
	state            Checkpoint
	latest           *Checkpoint
	durability       Durability
	pause            atomic.Bool
	pauseOnce        sync.Once
	pauseSignal      chan struct{}
	failure          error
	resumed          bool
	resumeCheckpoint *Checkpoint
}

// snapshot returns only acknowledged state, or nil before the first commit.
func (d *checkpointCoordinator) snapshot() *Checkpoint {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.latest == nil {
		return nil
	}
	c := d.latest.Clone()
	return &c
}

// current detaches coordinator state before proposing the next revision.
func (d *checkpointCoordinator) current() Checkpoint {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.state.Clone()
}

// eventIdentity reads immutable stream identifiers without copying payloads.
func (d *checkpointCoordinator) eventIdentity(stageIndex int) (runID, executionID, stageID string) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if stageIndex >= 0 && stageIndex < len(d.state.Stages) {
		stageID = d.state.Stages[stageIndex].ID
	}
	return d.state.RunID, d.state.ExecutionID, stageID
}

// revision reads the last acknowledged revision without copying its contents.
func (d *checkpointCoordinator) revision() uint64 {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.state.Revision
}

// commit serializes CAS transitions and latches storage failures to fence admission.
func (d *checkpointCoordinator) commit(ctx context.Context, change func(*Checkpoint)) error {
	d.commitMu.Lock()
	defer d.commitMu.Unlock()
	if d.failure != nil {
		return d.failure
	}
	next := d.current()
	expected := next.Revision
	change(&next)
	next.Revision = expected + 1
	// Validate serializability before cloning, so no unsupported value disappears.
	if _, err := json.Marshal(next); err != nil {
		return fmt.Errorf("%w: %w", ErrCheckpoint, err)
	}
	if err := next.Validate(); err != nil {
		return err
	}
	if err := d.durability.Commit(ctx, expected, next.Clone()); err != nil {
		d.failure = fmt.Errorf("%w: %w", loop.ErrCheckpointCommit, err)
		return d.failure
	}
	d.mu.Lock()
	d.state = next.Clone()
	latest := next.Clone()
	d.latest = &latest
	d.mu.Unlock()
	return nil
}

// stage returns an isolated stage record without exposing coordinator-owned data.
func (d *checkpointCoordinator) stage(id string) StageCheckpoint {
	for _, s := range d.current().Stages {
		if s.ID == id {
			return s
		}
	}
	return StageCheckpoint{}
}

// changeStage applies a transition to the registered durable stage identity.
func changeStage(c *Checkpoint, id string, change func(*StageCheckpoint)) {
	for i := range c.Stages {
		if c.Stages[i].ID == id {
			change(&c.Stages[i])
			return
		}
	}
}

// control connects one loop to the aggregate checkpoint coordinator.
func (d *checkpointCoordinator) control(stageID string) *loop.ContinuationControl {
	stage := d.stage(stageID)
	state := d.current()
	return &loop.ContinuationControl{RunID: state.RunID, ExecutionID: state.ExecutionID, StageID: stageID, Initial: stage.Continuation, Revision: d.revision, DeferredApprovals: d.durability.DeferredApprovals, PauseRequested: d.pause.Load, PauseSignal: d.pauseSignal, Commit: func(ctx context.Context, c loop.Continuation) error {
		return d.commit(ctx, func(cp *Checkpoint) {
			cp.ActiveStageID = stageID
			changeStage(cp, stageID, func(s *StageCheckpoint) { s.Continuation = &c })
		})
	}}
}

// RequestPause cooperatively stops new work at an execution boundary.
// It returns immediately; Wait or EventPaused acknowledges the committed pause.
func (w *Workflow) RequestPause() error {
	if w == nil || w.durability == nil {
		return ErrDurability
	}
	w.mu.RLock()
	complete := w.result.Complete
	w.mu.RUnlock()
	if complete {
		return ErrWorkflowFinished
	}
	d := w.durability
	d.pause.Store(true)
	d.pauseOnce.Do(func() { close(d.pauseSignal) })
	return nil
}

// validateDurability rejects unsupported definitions before durable execution begins.
func validateDurability(a *Agent, input RunInput) error {
	if input.Durability == nil {
		return nil
	}
	d := input.Durability
	if d.DefinitionRevision == "" || d.Commit == nil {
		return ErrDurability
	}
	if err := validateDurableOverrides(input.Execution); err != nil {
		return err
	}
	if a == nil {
		return loop.ErrNilLoop
	}
	if err := validateMiddleware(a.def.Middleware); err != nil {
		return err
	}
	seen := map[string]bool{PrimaryStageID: true}
	for _, mw := range a.def.Middleware {
		m, ok := mw.(*AgentMiddleware)
		if !ok || m == nil || m.config.ID == "" || seen[m.config.ID] {
			return ErrUnsupportedDurableMiddleware
		}
		seen[m.config.ID] = true
	}
	_, err := json.Marshal(input.Meta)
	if err != nil {
		return fmt.Errorf("%w: metadata must be JSON: %w", ErrDurability, err)
	}
	return nil
}

// validateDurableOverrides keeps live dependencies in the compatible Definition.
func validateDurableOverrides(o *ExecutionOverrides) error {
	if o != nil && (o.Model != nil || o.Tools != nil || o.ToolPolicy.Set || o.ToolApprovalResolver.Set || o.TokenCounter.Set || o.ToolResultProcessor.Set || o.RetryPolicy.Set) {
		return fmt.Errorf("%w: durable dependencies and retry policy must be registered in Definition", ErrDurability)
	}
	return nil
}

// configureDurability registers stable stages without starting execution or persistence.
func (w *Workflow) configureDurability(a *Agent, input RunInput) error {
	if input.Durability == nil {
		return nil
	}
	switch w.loop.PromptBuilder.(type) {
	case *modelinput.Builder, *compactingPromptBuilder:
	default:
		return ErrUnsupportedDurablePrompt
	}
	cp := Checkpoint{Version: CheckpointVersion, RunID: input.ID, ExecutionID: rand.Text(), DefinitionRevision: input.Durability.DefinitionRevision, DeferredApprovals: input.Durability.DeferredApprovals, Status: StatusRunning, ActiveStageID: PrimaryStageID, Stages: []StageCheckpoint{{ID: PrimaryStageID, Name: w.name, DecisionMade: true, ShouldRun: true, Settings: settingsForLoop(w.loop), Input: checkpointInput{Meta: input.Meta, TraceContext: input.TraceContext, ScalarExecution: snapshotOverrides(input.Execution)}}}}
	for _, item := range a.def.Middleware {
		m := item.(*AgentMiddleware)
		cp.Stages = append(cp.Stages, StageCheckpoint{ID: m.config.ID, Name: m.name(), Output: m.config.Output})
	}
	w.durability = &checkpointCoordinator{state: cp, durability: *input.Durability, pauseSignal: make(chan struct{})}
	w.stageID = PrimaryStageID
	w.loop.Continuation = w.durability.control(PrimaryStageID)
	return nil
}

// Resume claims the loaded revision before executing any model/tool work. It
// creates a fresh single-use handle and preserves the logical RunID. Terminal
// success/cancellation require a new run. Failed checkpoints may be recovered
// only through their last acknowledged continuation and explicit resolutions.
func (a *Agent) Resume(ctx context.Context, checkpoint Checkpoint, input ResumeInput, durability Durability) (*Workflow, error) {
	if err := checkpoint.Validate(); err != nil {
		return nil, err
	}
	checkpoint = checkpoint.Clone()
	if checkpoint.Status == StatusSucceeded || checkpoint.Status == StatusCanceled {
		return nil, ErrWorkflowFinished
	}
	if checkpoint.DefinitionRevision != durability.DefinitionRevision || checkpoint.DeferredApprovals != durability.DeferredApprovals {
		return nil, ErrDefinitionRevision
	}
	if err := validateDurability(a, RunInput{Durability: &durability}); err != nil {
		return nil, err
	}
	if len(checkpoint.Stages) != len(a.def.Middleware)+1 {
		return nil, ErrDefinitionRevision
	}
	for i, item := range a.def.Middleware {
		m := item.(*AgentMiddleware)
		s := checkpoint.Stages[i+1]
		if s.ID != m.config.ID || s.Name != m.name() || s.Output != m.config.Output {
			return nil, ErrDefinitionRevision
		}
	}
	var previous *ContinuationCommand
	for i := range checkpoint.Commands {
		if checkpoint.Commands[i].ID == input.CommandID {
			previous = &checkpoint.Commands[i]
			break
		}
	}
	stageID := input.StageID
	if stageID == "" {
		stageID = checkpoint.ActiveStageID
		if previous != nil {
			stageID = previous.StageID
		}
	}
	if previous == nil && stageID != checkpoint.ActiveStageID {
		return nil, ErrCheckpoint
	}
	hasCommand := input.CommandID != "" || len(input.Prompt.User) > 0 || len(input.Prompt.Context) > 0 || input.PendingTools != "" || len(input.Approvals) > 0 || len(input.ToolResolutions) > 0
	if hasCommand {
		if input.CommandID == "" || (input.PendingTools != ContinuePendingTools && input.PendingTools != CancelUnstartedTools) {
			return nil, ErrCheckpoint
		}
		messages, err := modelinput.MaterializeInput(ctx, input.Prompt)
		if err != nil {
			return nil, err
		}
		command := ContinuationCommand{ID: input.CommandID, StageID: stageID, Messages: messages, PendingTools: input.PendingTools, Approvals: input.Approvals, ToolResolutions: input.ToolResolutions}
		if previous != nil {
			a, _ := json.Marshal(previous)
			b, err := json.Marshal(command)
			if err != nil || string(a) != string(b) {
				return nil, ErrCommandConflict
			}
		} else {
			var stage *StageCheckpoint
			for i := range checkpoint.Stages {
				if checkpoint.Stages[i].ID == stageID {
					stage = &checkpoint.Stages[i]
				}
			}
			if stage == nil || stage.Complete || stage.Continuation == nil {
				return nil, ErrCheckpoint
			}
			if err := stage.Continuation.ResolveRecovery(input.ToolResolutions); err != nil {
				return nil, err
			}
			if err := stage.Continuation.ApplyApprovals(input.Approvals); err != nil {
				return nil, err
			}
			if input.PendingTools == CancelUnstartedTools {
				if err := stage.Continuation.CancelUnstarted(); err != nil {
					return nil, err
				}
			}
			stage.Continuation.Steering = append(stage.Continuation.Steering, messages...)
			checkpoint.Commands = append(checkpoint.Commands, command)
		}
	}
	for _, s := range checkpoint.Stages {
		if s.Continuation != nil {
			if err := s.Continuation.ResolveRecovery(nil); err != nil {
				return nil, err
			}
		}
	}
	if err := checkpoint.Validate(); err != nil {
		return nil, err
	}
	// Validate restored dependencies before changing checkpoint ownership.
	primary := checkpoint.Stages[0]
	l, err := a.restoreCheckpointLoop(primary)
	if err != nil {
		return nil, err
	}
	for i, item := range a.def.Middleware {
		stage := checkpoint.Stages[i+1]
		if stage.Continuation != nil && !stage.Complete {
			if _, err := item.(*AgentMiddleware).agent.restoreCheckpointLoop(stage); err != nil {
				return nil, err
			}
		}
	}
	checkpoint.ExecutionID = rand.Text()
	checkpoint.Status = StatusRunning
	d := &checkpointCoordinator{state: checkpoint, durability: durability, pauseSignal: make(chan struct{}), resumed: true}
	if err := d.commit(ctx, func(*Checkpoint) {}); err != nil {
		return nil, err
	}
	d.resumeCheckpoint = d.snapshot()
	runInput := RunInput{ID: checkpoint.RunID, Meta: primary.Input.Meta, TraceContext: primary.Input.TraceContext, Durability: &durability, Execution: restoreOverrides(primary.Input.ScalarExecution)}
	if primary.Continuation != nil && primary.Continuation.Prompt != nil {
		runInput.Prompt = primary.Continuation.Prompt.Input()
	}
	w := newWorkflow(runInput, l, a.name(), a.debugSink(), a.middleware())
	w.durability = d
	w.stageID = PrimaryStageID
	l.Continuation = d.control(PrimaryStageID)
	return w, nil
}
