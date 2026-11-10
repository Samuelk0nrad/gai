package agent

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/loop"
	"github.com/Samuelk0nrad/gai/modelinput"
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
	errUnfinishedStage              = errors.New("workflow ended before every stage completed")
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
// The checkpoint passed to Commit shares memory with the workflow and with later
// revisions, so Commit must not modify it. Encoding or keeping it unchanged is
// safe; use Checkpoint.Clone for a copy to modify.
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

// ContinuationCommand records an accepted command by ID and by a SHA-256 digest
// of its materialized content. Exact redelivery never reapplies the command; a
// different payload under the same ID is a conflict. The applied content itself
// lives in the stage continuation (steering, approvals and tool resolutions).
type ContinuationCommand struct {
	ID      string `json:"id"`
	StageID string `json:"stage_id"`
	Digest  string `json:"digest"`
}

// commandDigest identifies the materialized command content, including the
// rendered context and user messages.
func commandDigest(stageID string, messages []ai.Message, input ResumeInput) (string, error) {
	data, err := json.Marshal(struct {
		StageID         string                        `json:"stage_id"`
		Messages        []ai.Message                  `json:"messages,omitempty"`
		PendingTools    PendingToolAction             `json:"pending_tools,omitempty"`
		Approvals       []loop.ToolApprovalDecision   `json:"approvals,omitempty"`
		ToolResolutions []loop.ToolRecoveryResolution `json:"tool_resolutions,omitempty"`
	}{stageID, messages, input.PendingTools, input.Approvals, input.ToolResolutions})
	if err != nil {
		return "", fmt.Errorf("%w: command: %w", ErrCheckpoint, err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
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

// Clone returns a deep copy that shares no mutable memory with c. Strings are
// immutable and stay shared. Metadata maps and slices are copied; other
// metadata values, which JSON-decoded checkpoints never contain, are shared.
func (c Checkpoint) Clone() Checkpoint {
	if c.Stages != nil {
		stages := make([]StageCheckpoint, len(c.Stages))
		for i, s := range c.Stages {
			stages[i] = s.clone()
		}
		c.Stages = stages
	}
	c.Commands = slices.Clone(c.Commands)
	return c
}

// clone deep-copies one stage record.
func (s StageCheckpoint) clone() StageCheckpoint {
	if s.Continuation != nil {
		continuation := s.Continuation.Clone()
		s.Continuation = &continuation
	}
	if s.Settings != nil {
		settings := *s.Settings
		settings.ToolChoice = cloneToolChoice(settings.ToolChoice)
		settings.ResponseFormat = cloneResponseFormat(settings.ResponseFormat)
		settings.RequestBudget = clonePointer(settings.RequestBudget)
		s.Settings = &settings
	}
	if s.Input.ScalarExecution != nil {
		s.Input.ScalarExecution = snapshotOverrides(restoreOverrides(s.Input.ScalarExecution))
	}
	s.Input.Meta = cloneJSONObject(s.Input.Meta)
	s.Input.TraceContext = cloneTraceContext(s.Input.TraceContext)
	s.Errors = slices.Clone(s.Errors)
	return s
}

// cloneJSONObject deep-copies JSON-shaped metadata.
func cloneJSONObject(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for key, value := range m {
		out[key] = cloneJSONValue(value)
	}
	return out
}

func cloneJSONValue(value any) any {
	switch v := value.(type) {
	case map[string]any:
		return cloneJSONObject(v)
	case []any:
		out := make([]any, len(v))
		for i := range v {
			out[i] = cloneJSONValue(v[i])
		}
		return out
	default:
		return value
	}
}

func cloneTraceContext(trace *gai.TraceContext) *gai.TraceContext {
	if trace == nil {
		return nil
	}
	copy := *trace
	copy.Tags = slices.Clone(trace.Tags)
	if trace.Metadata != nil {
		copy.Metadata = make(map[string]string, len(trace.Metadata))
		for key, value := range trace.Metadata {
			copy.Metadata[key] = value
		}
	}
	return &copy
}

// Validate rejects incompatible and structurally invalid saved state before any
// resume side effects. Checkpoints must come from trusted application storage.
func (c Checkpoint) Validate() error {
	if _, err := json.Marshal(c); err != nil {
		return fmt.Errorf("%w: %w", ErrCheckpoint, err)
	}
	return c.validate(true)
}

// validate checks checkpoint structure. With deep, it also validates every
// stage continuation; commits skip that because the loop validates each
// continuation before handing it over, and Resume validates loaded ones.
func (c Checkpoint) validate(deep bool) error {
	// The loop's first commit saves the primary prompt, so every stored
	// checkpoint has a primary continuation.
	if c.Version != CheckpointVersion || c.RunID == "" || c.ExecutionID == "" || c.DefinitionRevision == "" || c.Revision == 0 || len(c.Stages) == 0 || c.Stages[0].ID != PrimaryStageID || c.Stages[0].Continuation == nil {
		return ErrCheckpoint
	}
	if c.Status != StatusRunning && c.Status != StatusPaused && c.Status != StatusFailed && c.Status != StatusCanceled && c.Status != StatusSucceeded {
		return ErrCheckpoint
	}
	seen := map[string]bool{}
	active := false
	for _, s := range c.Stages {
		// A succeeded run is final, so every stage must be complete.
		if c.Status == StatusSucceeded && !s.Complete {
			return ErrCheckpoint
		}
		if s.ID == "" || seen[s.ID] || s.Output > ReplaceOutput {
			return ErrCheckpoint
		}
		seen[s.ID] = true
		if s.ID == c.ActiveStageID {
			active = true
		}
		if s.Skipped && (!s.Complete || s.ShouldRun) || s.Complete && s.Canceled {
			return ErrCheckpoint
		}
		if deep && s.Continuation != nil {
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
		if command.ID == "" || command.Digest == "" || commandIDs[command.ID] || !seen[command.StageID] {
			return ErrCheckpoint
		}
		commandIDs[command.ID] = true
	}
	return nil
}

// checkpointCoordinator owns the aggregate checkpoint. state is never modified
// in place: each commit builds a new value that shares unchanged stage data with
// the previous one, so readers only need mu to load the current value.
type checkpointCoordinator struct {
	mu       sync.RWMutex
	commitMu sync.Mutex
	state    Checkpoint
	// acknowledged reports whether state has been durably committed.
	acknowledged     bool
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
	state, acknowledged := d.state, d.acknowledged
	d.mu.RUnlock()
	if !acknowledged {
		return nil
	}
	c := state.Clone()
	return &c
}

// isAcknowledged reports whether any checkpoint has been durably committed.
func (d *checkpointCoordinator) isAcknowledged() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.acknowledged
}

// status reads the workflow status without copying stage payloads.
func (d *checkpointCoordinator) status() WorkflowStatus {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.state.Status
}

// runID reads the logical run identity without copying stage payloads.
func (d *checkpointCoordinator) runID() string {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.state.RunID
}

// earlierStagesComplete reports whether every stage before id has completed.
func (d *checkpointCoordinator) earlierStagesComplete(id string) bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	for _, s := range d.state.Stages {
		if s.ID == id {
			return true
		}
		if !s.Complete {
			return false
		}
	}
	return true
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
// change may replace top-level and stage fields of its argument, but must not
// modify values reachable through them: those are shared with earlier states.
func (d *checkpointCoordinator) commit(ctx context.Context, change func(*Checkpoint)) error {
	d.commitMu.Lock()
	defer d.commitMu.Unlock()
	if d.failure != nil {
		return d.failure
	}
	d.mu.RLock()
	next := d.state
	d.mu.RUnlock()
	next.Stages = slices.Clone(next.Stages)
	expected := next.Revision
	change(&next)
	next.Revision = expected + 1
	if err := next.validate(false); err != nil {
		return err
	}
	// The callback reads the shared revision without a copy: state is never
	// modified in place. Metadata is normalized to JSON values and the loop
	// validates every continuation, so the revision is serializable.
	if err := d.durability.Commit(ctx, expected, next); err != nil {
		err = fmt.Errorf("%w: %w", loop.ErrCheckpointCommit, err)
		// A write that failed only because the execution was canceled does not
		// prove storage is broken: later settlement writes may still succeed,
		// and compare-and-swap rejects them if that write did land.
		if ctx.Err() == nil {
			d.failure = err
		}
		return err
	}
	d.mu.Lock()
	d.state = next
	d.acknowledged = true
	d.mu.Unlock()
	return nil
}

// stage returns one stage record. It shares memory with coordinator state, so
// callers must not modify anything reachable from it.
func (d *checkpointCoordinator) stage(id string) StageCheckpoint {
	d.mu.RLock()
	defer d.mu.RUnlock()
	for _, s := range d.state.Stages {
		if s.ID == id {
			return s
		}
	}
	return StageCheckpoint{}
}

// allStagesComplete reports whether every stage has completed.
func (d *checkpointCoordinator) allStagesComplete() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	for _, s := range d.state.Stages {
		if !s.Complete {
			return false
		}
	}
	return true
}

// stageComplete reads one completion flag without copying stage payloads.
func (d *checkpointCoordinator) stageComplete(id string) bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	for _, s := range d.state.Stages {
		if s.ID == id {
			return s.Complete
		}
	}
	return false
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
	runID, executionID, _ := d.eventIdentity(-1)
	return &loop.ContinuationControl{RunID: runID, ExecutionID: executionID, StageID: stageID, Initial: stage.Continuation, Revision: d.revision, DeferredApprovals: d.durability.DeferredApprovals, PauseRequested: d.pause.Load, PauseSignal: d.pauseSignal, Commit: func(ctx context.Context, c loop.Continuation) error {
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
	cp := Checkpoint{Version: CheckpointVersion, RunID: input.ID, ExecutionID: rand.Text(), DefinitionRevision: input.Durability.DefinitionRevision, DeferredApprovals: input.Durability.DeferredApprovals, Status: StatusRunning, ActiveStageID: PrimaryStageID, Stages: []StageCheckpoint{{ID: PrimaryStageID, Name: w.name, DecisionMade: true, ShouldRun: true, Settings: settingsForLoop(w.loop), Input: checkpointInput{Meta: cloneJSONObject(input.Meta), TraceContext: cloneTraceContext(input.TraceContext), ScalarExecution: snapshotOverrides(input.Execution)}}}}
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
// creates a fresh single-use handle and preserves the logical RunID. Only a
// succeeded run is finished. A failed or canceled execution resumes from its
// last acknowledged continuation, subject to explicit tool resolutions. A stage
// that cannot continue because of its TotalTimeout, retry or iteration limit is
// rejected before the claim (see loop.Continuation.LimitError). Ending a
// logical run early is an application decision recorded in application storage.
func (a *Agent) Resume(ctx context.Context, checkpoint Checkpoint, input ResumeInput, durability Durability) (*Workflow, error) {
	if err := checkpoint.Validate(); err != nil {
		return nil, err
	}
	checkpoint = checkpoint.Clone()
	if checkpoint.Status == StatusSucceeded {
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
		if input.PendingTools == CancelUnstartedTools && slices.ContainsFunc(input.ToolResolutions, func(r loop.ToolRecoveryResolution) bool { return r.Action == loop.ReplayInvocation }) {
			return nil, fmt.Errorf("%w: ReplayInvocation cannot be combined with CancelUnstartedTools", ErrCheckpoint)
		}
		messages, err := modelinput.MaterializeInput(ctx, input.Prompt)
		if err != nil {
			return nil, err
		}
		digest, err := commandDigest(stageID, messages, input)
		if err != nil {
			return nil, err
		}
		command := ContinuationCommand{ID: input.CommandID, StageID: stageID, Digest: digest}
		if previous != nil {
			if *previous != command {
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
	// Validate restored dependencies and the remaining timeout budget before
	// changing checkpoint ownership.
	primary := checkpoint.Stages[0]
	l, err := a.restoreCheckpointLoop(primary)
	if err != nil {
		return nil, err
	}
	if !primary.Complete {
		if err := primary.Continuation.LimitError(l); err != nil {
			return nil, err
		}
	}
	for i, item := range a.def.Middleware {
		stage := checkpoint.Stages[i+1]
		if stage.Continuation != nil && !stage.Complete {
			child, err := item.(*AgentMiddleware).agent.restoreCheckpointLoop(stage)
			if err != nil {
				return nil, err
			}
			if err := stage.Continuation.LimitError(child); err != nil {
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
	runInput := RunInput{ID: checkpoint.RunID, Meta: cloneJSONObject(primary.Input.Meta), TraceContext: cloneTraceContext(primary.Input.TraceContext), Durability: &durability, Execution: restoreOverrides(primary.Input.ScalarExecution)}
	if primary.Continuation != nil && primary.Continuation.Prompt != nil {
		runInput.Prompt = primary.Continuation.Prompt.Input()
	}
	w := newWorkflow(runInput, l, a.name(), a.debugSink(), a.middleware())
	w.durability = d
	w.stageID = PrimaryStageID
	l.Continuation = d.control(PrimaryStageID)
	return w, nil
}
