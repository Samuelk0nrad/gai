package agent

import (
	"context"
	"errors"
	"fmt"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

var (
	ErrMiddlewareAgentNotConfigured = errors.New("middleware agent is not configured")
	ErrMiddlewareAgentNested        = errors.New("middleware agent cannot define middleware")
	ErrMiddlewareNameMissing        = errors.New("middleware name is missing")
	ErrMiddlewareOutputInvalid      = errors.New("middleware output policy is invalid")
	ErrMiddlewareErrorPolicyInvalid = errors.New("middleware error policy is invalid")
)

// OutputPolicy controls how an agent middleware transforms upstream tokens.
type OutputPolicy uint8

const (
	// PreserveOutput forwards upstream tokens and records the middleware agent's
	// result without exposing its tokens to the caller.
	PreserveOutput OutputPolicy = iota
	// AppendOutput forwards upstream tokens followed by middleware-agent tokens.
	AppendOutput
	// ReplaceOutput buffers upstream tokens and emits middleware-agent tokens only
	// when the middleware agent completes successfully.
	ReplaceOutput
)

// ErrorPolicy controls whether an agent-middleware failure fails the workflow.
type ErrorPolicy uint8

const (
	// PropagateError sends middleware failures through the workflow error stream.
	PropagateError ErrorPolicy = iota
	// RecordError records middleware failures in StageResult without failing the
	// surrounding workflow.
	RecordError
)

// AgentMiddlewareConfig configures an agent as stream middleware.
type AgentMiddlewareConfig struct {
	// Name identifies the stage in WorkflowResult.Stages. The agent name is used
	// when Name is empty.
	Name string
	// Output controls how the nested agent changes visible workflow tokens.
	Output OutputPolicy
	// MapInput controls exactly what the nested agent receives. When nil, the
	// current visible text is forwarded as named upstream_output context along
	// with the original run ID and metadata. Execution overrides are not
	// forwarded; the nested agent uses its own definition defaults.
	MapInput func(ctx context.Context, result WorkflowResult) (RunInput, error)
	// ErrorPolicy controls whether input-mapping and nested-agent failures
	// propagate. The zero value is PropagateError.
	ErrorPolicy ErrorPolicy
	// ShouldRun overrides the default success-only policy. It receives the full
	// upstream result and may enable stages such as failure auditing.
	ShouldRun func(result WorkflowResult) bool
}

// AgentMiddleware runs an agent after its upstream stream completes.
type AgentMiddleware struct {
	agent  *Agent
	config AgentMiddlewareConfig
}

// NewAgentMiddleware adapts an ordinary agent into workflow middleware. Use
// AgentMiddlewareConfig.MapInput when the nested agent needs a deliberate
// projection of the upstream workflow state.
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
func (m *AgentMiddleware) Process(ctx context.Context, run *MiddlewareContext, upstream Stream) Stream {
	tokens := make(chan ai.Token, 16)
	statuses := make(chan loop.IterationInformation, 16)
	errs := make(chan error, 1)

	go func() {
		defer close(tokens)
		defer close(statuses)
		defer close(errs)

		upstreamRelay := streamRelay{Statuses: statuses, Errors: errs}
		if m.config.Output != ReplaceOutput {
			upstreamRelay.Tokens = tokens
		}
		upstreamResult := drainStream(ctx, upstream, upstreamRelay)
		stageCtx, obs := newMiddlewareObserver(ctx, run, m, upstreamResult)
		obs.Started(stageCtx)
		result := m.upstreamResult(run, upstreamResult)

		if result.Canceled {
			m.restoreReplacement(stageCtx, tokens, upstreamResult.Tokens)
			obs.Skipped(stageCtx, "upstream_canceled")
			return
		}
		if !m.shouldRun(result) {
			m.restoreReplacement(stageCtx, tokens, upstreamResult.Tokens)
			reason := "predicate"
			if m.config.ShouldRun == nil && len(result.Errors) > 0 {
				reason = "upstream_error"
			}
			obs.Skipped(stageCtx, reason)
			return
		}

		input, err := m.input(stageCtx, result)
		if err != nil {
			m.finishStage(stageCtx, run, tokens, errs, upstreamResult.Tokens, AgentResult{Errors: []error{err}}, obs)
			return
		}

		stageResult := m.runStage(stageCtx, input)
		m.finishStage(stageCtx, run, tokens, errs, upstreamResult.Tokens, stageResult, obs)
	}()

	return Stream{Tokens: tokens, Statuses: statuses, Errors: errs}
}

func (m *AgentMiddleware) upstreamResult(run *MiddlewareContext, captured capturedStream) WorkflowResult {
	result := run.Result()
	result.AttemptedTokens = cloneTokens(captured.Tokens)
	result.AttemptedText = tokenText(captured.Tokens)
	result.Errors = append([]error(nil), captured.Errors...)
	return result
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

func (m *AgentMiddleware) runStage(ctx context.Context, input RunInput) AgentResult {
	workflow, err := m.agent.NewRun(ctx, input)
	if err != nil {
		return AgentResult{Errors: []error{err}}
	}
	tokens, statuses, errs := workflow.Run(ctx)
	captured := drainStream(ctx, Stream{Tokens: tokens, Statuses: statuses, Errors: errs}, streamRelay{})
	result := workflow.Result().Primary
	result.AttemptedTokens = cloneTokens(captured.Tokens)
	result.AttemptedText = tokenText(captured.Tokens)
	result.Errors = append([]error(nil), captured.Errors...)
	return result
}

func (m *AgentMiddleware) finishStage(
	ctx context.Context,
	run *MiddlewareContext,
	tokens chan<- ai.Token,
	errs chan<- error,
	upstreamTokens []ai.Token,
	result AgentResult,
	obs *middlewareObserver,
) {
	failed := len(result.Errors) > 0
	visible := cloneTokens(upstreamTokens)
	if failed {
		if m.config.ErrorPolicy == PropagateError {
			forwardErrors(ctx, errs, result.Errors)
		}
		m.restoreReplacement(ctx, tokens, upstreamTokens)
	} else {
		switch m.config.Output {
		case AppendOutput:
			forwardTokens(ctx, tokens, result.Tokens)
			visible = append(visible, cloneTokens(result.Tokens)...)
		case ReplaceOutput:
			forwardTokens(ctx, tokens, result.Tokens)
			visible = cloneTokens(result.Tokens)
		}
	}
	run.workflow.addStage(StageResult{
		Name:   m.name(),
		Output: m.config.Output,
		Result: result,
	}, visible)
	obs.Finished(ctx, result, !failed && m.config.Output != PreserveOutput)
}

func (m *AgentMiddleware) restoreReplacement(ctx context.Context, tokens chan<- ai.Token, upstream []ai.Token) {
	if m.config.Output == ReplaceOutput {
		forwardTokens(ctx, tokens, upstream)
	}
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

func forwardTokens(ctx context.Context, output chan<- ai.Token, tokens []ai.Token) {
	for _, token := range tokens {
		send(ctx, output, token)
	}
}

func forwardErrors(ctx context.Context, output chan<- error, errs []error) {
	for _, err := range errs {
		if err != nil {
			send(ctx, output, err)
		}
	}
}
