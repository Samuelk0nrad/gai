package agent

import (
	"context"
	"errors"
	"fmt"

	gaictx "github.com/lace-ai/gai/context"
)

var (
	ErrMiddlewareAgentNotConfigured = errors.New("middleware agent is not configured")
	ErrMiddlewareAgentNested        = errors.New("middleware agent cannot define middleware")
	ErrMiddlewareNameMissing        = errors.New("middleware name is missing")
	ErrMiddlewareOutputInvalid      = errors.New("middleware output policy is invalid")
	ErrMiddlewareErrorPolicyInvalid = errors.New("middleware error policy is invalid")
)

const (
	stageReasonRecorded   = "recorded_error"
	stageReasonPropagated = "propagated_error"
)

// OutputPolicy controls how agent middleware transforms upstream output.
type OutputPolicy uint8

const (
	// PreserveOutput forwards upstream output and records the middleware-agent
	// result without emitting its text as EventOutput.
	PreserveOutput OutputPolicy = iota
	// AppendOutput forwards upstream output and then emits accepted
	// middleware-agent output.
	AppendOutput
	// ReplaceOutput buffers upstream output and emits accepted middleware-agent
	// output on success, restoring upstream output on failure.
	ReplaceOutput
)

// ErrorPolicy controls whether an agent-middleware failure fails the workflow.
type ErrorPolicy uint8

const (
	// PropagateError makes a middleware-agent failure fail the workflow.
	PropagateError ErrorPolicy = iota
	// RecordError retains the failure in StageResult without failing the workflow.
	RecordError
)

// AgentMiddlewareConfig configures an agent as ordered event middleware.
type AgentMiddlewareConfig struct {
	// Name identifies the stage. The agent name is used when empty.
	Name string
	// Output controls how the stage changes visible workflow output.
	Output OutputPolicy
	// MapInput maps the accumulated workflow snapshot to the nested run.
	MapInput func(context.Context, WorkflowResult) (RunInput, error)
	// ErrorPolicy controls whether stage failures fail the workflow.
	ErrorPolicy ErrorPolicy
	// ShouldRun overrides the default success-only policy.
	ShouldRun func(WorkflowResult) bool
}

// AgentMiddleware runs an agent after its upstream stream completes.
type AgentMiddleware struct {
	agent  *Agent
	config AgentMiddlewareConfig
}

func NewAgentMiddleware(agent *Agent, config AgentMiddlewareConfig) *AgentMiddleware {
	return &AgentMiddleware{agent: agent, config: config}
}

func (m *AgentMiddleware) validate() error {
	if m == nil || m.agent == nil {
		return ErrMiddlewareAgentNotConfigured
	}
	if len(m.agent.def.Middleware) > 0 {
		return ErrMiddlewareAgentNested
	}
	if m.name() == "" {
		return ErrMiddlewareNameMissing
	}
	if m.config.Output > ReplaceOutput {
		return fmt.Errorf("%w: %d", ErrMiddlewareOutputInvalid, m.config.Output)
	}
	if m.config.ErrorPolicy > RecordError {
		return fmt.Errorf("%w: %d", ErrMiddlewareErrorPolicyInvalid, m.config.ErrorPolicy)
	}
	return nil
}

// Process implements Middleware.
func (m *AgentMiddleware) Process(ctx context.Context, run *MiddlewareContext, upstream <-chan Event) <-chan Event {
	out := make(chan Event)
	go func() {
		defer close(out)
		var upstreamOutput []Event
		invalid := make(map[attemptKey]bool)
		deliver := true
		for event := range upstream {
			switch event.Type {
			case EventRetry, EventDiscard:
				invalid[eventAttemptKey(event)] = true
			case EventStageFinish:
				if event.AttemptID != 0 && (event.StageOutcome == StageFailed || event.StageOutcome == StageCanceled) {
					invalid[eventAttemptKey(event)] = true
				}
			}
			if event.Type == EventOutput && m.config.Output == ReplaceOutput {
				upstreamOutput = append(upstreamOutput, cloneEvent(event))
				continue
			}
			deliver = sendWorkflowEvent(ctx, out, cloneEvent(event), deliver)
		}
		restorable := upstreamOutput[:0:0]
		for _, event := range upstreamOutput {
			if !invalid[eventAttemptKey(event)] {
				restorable = append(restorable, event)
			}
		}
		upstreamOutput = restorable

		result := run.Result()
		stageCtx, obs := newMiddlewareObserver(ctx, run, m, result)
		obs.Started(stageCtx)
		deliver = sendWorkflowEvent(ctx, out, Event{Type: EventStageStart, Source: run.Source()}, deliver)

		if result.Canceled {
			deliver = forwardEvents(ctx, out, upstreamOutput, deliver)
			obs.Skipped(stageCtx, "upstream_canceled")
			sendWorkflowEvent(ctx, out, Event{Type: EventStageFinish, Source: run.Source(), StageOutcome: StageSkipped, StageReason: "upstream_canceled"}, deliver)
			return
		}
		if !m.shouldRun(result) {
			deliver = forwardEvents(ctx, out, upstreamOutput, deliver)
			reason := "predicate"
			if m.config.ShouldRun == nil && len(result.Errors) > 0 {
				reason = "upstream_error"
			}
			obs.Skipped(stageCtx, reason)
			sendWorkflowEvent(ctx, out, Event{Type: EventStageFinish, Source: run.Source(), StageOutcome: StageSkipped, StageReason: reason}, deliver)
			return
		}

		input, err := m.input(stageCtx, result)
		if err != nil {
			m.finishFailure(stageCtx, run, out, upstreamOutput, AgentResult{Errors: []error{err}}, obs, err, false, deliver)
			return
		}

		stageResult, nestedEvents, terminalErr := m.runStage(stageCtx, input, run.Source())
		if terminalErr != nil || stageResult.Canceled || len(stageResult.Errors) > 0 {
			for _, event := range nestedEvents {
				if event.Type != EventOutput {
					deliver = sendWorkflowEvent(ctx, out, cloneEvent(event), deliver)
				}
			}
			if terminalErr == nil {
				terminalErr = errors.Join(stageResult.Errors...)
				if stageResult.Canceled {
					terminalErr = stageResult.CancellationErr
				}
			}
			m.finishFailure(stageCtx, run, out, upstreamOutput, stageResult, obs, terminalErr, stageResult.Canceled, deliver)
			return
		}

		for _, event := range nestedEvents {
			if event.Type == EventOutput && m.config.Output == PreserveOutput {
				continue
			}
			deliver = sendWorkflowEvent(ctx, out, cloneEvent(event), deliver)
		}
		run.workflow.addStage(StageResult{Name: m.name(), Output: m.config.Output, Result: stageResult})
		obs.Finished(stageCtx, stageResult, m.config.Output != PreserveOutput)
		sendWorkflowEvent(ctx, out, Event{Type: EventStageFinish, Source: run.Source(), StageOutcome: StageSucceeded}, deliver)
	}()
	return out
}

func (m *AgentMiddleware) finishFailure(
	ctx context.Context,
	run *MiddlewareContext,
	out chan<- Event,
	upstreamOutput []Event,
	result AgentResult,
	obs *middlewareObserver,
	err error,
	canceled bool,
	deliver bool,
) {
	deliver = forwardEvents(ctx, out, upstreamOutput, deliver)
	run.workflow.addStage(StageResult{Name: m.name(), Output: m.config.Output, Result: result})
	obs.Finished(ctx, result, false)
	outcome := StageFailed
	reason := stageReasonRecorded
	if canceled {
		outcome = StageCanceled
		reason = "canceled"
	} else if m.config.ErrorPolicy == PropagateError {
		reason = stageReasonPropagated
	}
	sendWorkflowEvent(ctx, out, Event{Type: EventStageFinish, Source: run.Source(), StageOutcome: outcome, StageReason: reason, Err: err}, deliver)
}

func forwardEvents(ctx context.Context, out chan<- Event, events []Event, deliver bool) bool {
	for _, event := range events {
		deliver = sendWorkflowEvent(ctx, out, cloneEvent(event), deliver)
	}
	return deliver
}

func (m *AgentMiddleware) runStage(ctx context.Context, input RunInput, source EventSource) (AgentResult, []Event, error) {
	workflow, err := m.agent.NewRun(ctx, input)
	if err != nil {
		return AgentResult{Errors: []error{err}}, nil, err
	}
	var events []Event
	for event := range workflow.RunEvents(ctx) {
		switch event.Type {
		case EventStageStart, EventStageFinish, EventDone, EventError, EventCanceled:
			continue
		}
		event.Source = source
		events = append(events, cloneEvent(event))
	}
	result, err := workflow.Wait()
	return result.Primary, events, err
}

func (m *AgentMiddleware) shouldRun(result WorkflowResult) bool {
	if m.config.ShouldRun != nil {
		return m.config.ShouldRun(result)
	}
	return len(result.Errors) == 0
}

func (m *AgentMiddleware) input(ctx context.Context, result WorkflowResult) (RunInput, error) {
	if m.config.MapInput != nil {
		return m.config.MapInput(ctx, result)
	}
	upstream, err := gaictx.NewNamedPart("upstream_output", result.Text)
	if err != nil {
		return RunInput{}, err
	}
	return RunInput{
		ID: result.Input.ID,
		Prompt: gaictx.PromptInput{
			Context: []gaictx.Part{upstream},
		},
		Meta: cloneRunInput(result.Input).Meta,
	}, nil
}

func (m *AgentMiddleware) name() string {
	if m == nil {
		return ""
	}
	if m.config.Name != "" {
		return m.config.Name
	}
	if m.agent != nil {
		return m.agent.def.Name
	}
	return ""
}
