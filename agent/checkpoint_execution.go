package agent

import (
	"context"
	"errors"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/modelinput"
)

type checkpointSettings struct {
	MaxIterations  int                      `json:"max_iterations"`
	MaxTokens      int                      `json:"max_tokens"`
	ToolExecution  loop.ToolExecutionConfig `json:"tool_execution"`
	ToolChoice     ai.ToolChoice            `json:"tool_choice"`
	ToolTransport  loop.ToolTransportMode   `json:"tool_transport"`
	ResponseFormat ai.ResponseFormat        `json:"response_format"`
	Reasoning      ai.ReasoningConfig       `json:"reasoning"`
	RequestBudget  *ai.RequestBudgetConfig  `json:"request_budget,omitempty"`
}

// settingsForLoop captures serializable settings, excluding runtime dependencies.
func settingsForLoop(l *loop.Loop) *checkpointSettings {
	return &checkpointSettings{MaxIterations: l.MaxLoopIterations, MaxTokens: l.MaxTokens, ToolExecution: l.ToolExecution, ToolChoice: cloneToolChoice(l.ToolChoice), ToolTransport: l.ToolTransport, ResponseFormat: cloneResponseFormat(l.ResponseFormat), Reasoning: l.Reasoning, RequestBudget: clonePointer(l.RequestBudget)}
}

// restoreCheckpointLoop validates a loop rebuilt from frozen input and live dependencies.
func (a *Agent) restoreCheckpointLoop(stage StageCheckpoint) (*loop.Loop, error) {
	if stage.Settings == nil || stage.Continuation == nil || stage.Continuation.Prompt == nil {
		return nil, ErrCheckpoint
	}
	builder, err := modelinput.Restore(*stage.Continuation.Prompt)
	if err != nil {
		return nil, err
	}
	settings := stage.Settings
	l := loop.New(a.def.Model, cloneTools(a.def.Tools), builder, a.def.ToolResultProcessor)
	l.ObservationSink = a.def.ObservationSink
	l.ToolPolicy = a.def.ToolPolicy
	l.ToolApprovalResolver = a.def.ToolApprovalResolver
	l.TokenCounter = a.def.TokenCounter
	l.RetryPolicy = cloneRetryPolicy(a.def.RetryPolicy)
	l.MaxLoopIterations = settings.MaxIterations
	l.MaxTokens = settings.MaxTokens
	l.ToolExecution = settings.ToolExecution
	l.ToolChoice = cloneToolChoice(settings.ToolChoice)
	l.ToolTransport = settings.ToolTransport
	l.ResponseFormat = cloneResponseFormat(settings.ResponseFormat)
	l.Reasoning = settings.Reasoning
	l.RequestBudget = clonePointer(settings.RequestBudget)
	if err := l.Validate(); err != nil {
		return nil, err
	}
	return l, nil
}

// finishStage saves stage termination even when the execution context has ended.
func (d *checkpointCoordinator) finishStage(ctx context.Context, id string, result AgentResult, paused bool) error {
	return d.commit(context.WithoutCancel(ctx), func(cp *Checkpoint) {
		if paused {
			cp.Status = StatusPaused
		}
		changeStage(cp, id, func(s *StageCheckpoint) {
			s.Complete = !paused && len(result.Errors) == 0 && !result.Canceled
			s.Canceled = result.Canceled
			s.Errors = nil
			for _, err := range result.Errors {
				s.Errors = append(s.Errors, checkpointError(err))
			}
		})
	})
}

// checkpointAgentResult reconstructs accepted conversation and known billed usage.
func checkpointAgentResult(stage StageCheckpoint) AgentResult {
	var result AgentResult
	if stage.Continuation != nil {
		result.Iterations = stage.Continuation.Iterations()
		result.Tokens = iterationTokens(result.Iterations)
		result.Text = tokenText(result.Tokens)
		result.Reasoning = tokenReasoning(result.Tokens)
		for _, it := range result.Iterations {
			result.Messages = append(result.Messages, it.Messages()...)
		}
		result.Usage = iterationUsage(result.Iterations)
		for _, usage := range stage.Continuation.Usage {
			if usage.Reported {
				result.BilledUsage.Add(usage.Usage)
			}
		}
	}
	for _, text := range stage.Errors {
		result.Errors = append(result.Errors, errors.New(text))
	}
	result.Canceled = stage.Canceled
	if result.Canceled {
		result.CancellationErr = context.Canceled
	}
	return result
}

// replayResultOutput projects saved assistant parts through the live output conversion.
func replayResultOutput(result AgentResult, source EventSource, continuation *loop.Continuation) []Event {
	var events []Event
	for index, it := range result.Iterations {
		generation := continuation.Generations[index]
		for _, m := range it.Conversation {
			if m.Role != ai.RoleAssistant {
				continue
			}
			for _, p := range m.Parts {
				event, emit := mapLoopEvent(loop.TokenEvent(it.Count, generation.Attempt, 0, ai.Token{Part: &p}), source)
				if emit {
					event.GenerationID = generation.ID
					event.Replayed = true
					events = append(events, event)
				}
			}
		}
	}
	return events
}

// finishDurablePrimary preserves completed stages instead of writing them again.
func (w *Workflow) finishDurablePrimary(ctx context.Context, result AgentResult, paused bool) error {
	if w.durability.stage(w.stageID).Complete {
		return nil
	}
	return w.durability.finishStage(ctx, w.stageID, result, paused)
}

// checkpointError keeps raw processor failures out of persistent snapshots.
func checkpointError(err error) string {
	switch {
	case errors.Is(err, loop.ErrToolResultProcess):
		return loop.ErrToolResultProcess.Error()
	case errors.Is(err, loop.ErrToolPanic):
		return loop.ErrToolPanic.Error()
	case errors.Is(err, loop.ErrCheckpointCommit):
		return loop.ErrCheckpointCommit.Error()
	default:
		return "stage execution failed"
	}
}

// checkpointOverrides contains only serializable values. Runtime dependencies
// are reattached from the revision-compatible Definition.
type checkpointOverrides struct {
	Limits             LimitsOverrides                   `json:"limits"`
	ToolExecution      *loop.ToolExecutionConfig         `json:"tool_execution,omitempty"`
	ToolChoice         *ai.ToolChoice                    `json:"tool_choice,omitempty"`
	ResponseFormat     *ai.ResponseFormat                `json:"response_format,omitempty"`
	Reasoning          *ai.ReasoningConfig               `json:"reasoning,omitempty"`
	RequestBudget      Optional[*ai.RequestBudgetConfig] `json:"request_budget"`
	AutoCompactHistory *bool                             `json:"auto_compact_history,omitempty"`
}

// snapshotOverrides retains scalar overrides for restored middleware input.
func snapshotOverrides(o *ExecutionOverrides) *checkpointOverrides {
	if o == nil {
		return nil
	}
	o = cloneExecution(o)
	return &checkpointOverrides{Limits: o.Limits, ToolExecution: o.ToolExecution, ToolChoice: o.ToolChoice, ResponseFormat: o.ResponseFormat, Reasoning: o.Reasoning, RequestBudget: o.RequestBudget, AutoCompactHistory: o.AutoCompactHistory}
}

// restoreOverrides rehydrates scalar execution overrides without runtime objects.
func restoreOverrides(o *checkpointOverrides) *ExecutionOverrides {
	if o == nil {
		return nil
	}
	return cloneExecution(&ExecutionOverrides{Limits: o.Limits, ToolExecution: o.ToolExecution, ToolChoice: o.ToolChoice, ResponseFormat: o.ResponseFormat, Reasoning: o.Reasoning, RequestBudget: o.RequestBudget, AutoCompactHistory: o.AutoCompactHistory})
}
