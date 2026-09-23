package loop

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

type runExecution struct {
	owner                     *Loop
	callerCtx                 context.Context
	ctx                       context.Context
	events                    chan<- Event
	state                     *loopRunState
	executionTools            []Tool
	toolDefinitions           []ai.ToolDefinition
	userMessage               *gaictx.Message
	requiredToolCallSatisfied bool
}

type iterationOutcome uint8

const (
	iterationContinue iterationOutcome = iota
	iterationDone
	iterationTerminal
)

type attemptOutcome uint8

const (
	attemptAccepted attemptOutcome = iota
	attemptRetry
	attemptTerminal
)

type attemptExecution struct {
	run            *runExecution
	ctx            context.Context
	state          *loopIterationState
	iteration      Iteration
	toolCalls      []pendingToolCall
	deferredTokens []ai.Token
	deadline       context.Context
	cancel         context.CancelFunc
	cancelOnce     sync.Once
	finishOnce     sync.Once
}

func (l *Loop) executeRun(ctx context.Context, events chan<- Event) {
	callerCtx := ctx
	if l.RetryPolicy != nil && l.RetryPolicy.TotalTimeout > 0 {
		var totalCancel context.CancelFunc
		ctx, totalCancel = context.WithTimeout(ctx, l.RetryPolicy.TotalTimeout)
		defer totalCancel()
	}
	ctx, state := newLoopRunState(ctx, l)
	run := &runExecution{
		owner:     l,
		callerCtx: callerCtx,
		ctx:       ctx,
		events:    events,
		state:     state,
	}
	defer close(events)
	defer state.finish()

	if !run.prepareValidatedRun() {
		return
	}
	for i := range l.MaxLoopIterations {
		switch run.runIteration(i + 1) {
		case iterationContinue:
			continue
		case iterationDone, iterationTerminal:
			return
		}
	}
	if err := ctx.Err(); err != nil {
		sendLoopCanceled(ctx, events, state, err)
		return
	}
	sendLoopError(ctx, events, state, fmt.Errorf("%w: limit=%d", ErrMaxIterations, l.MaxLoopIterations))
}

func (r *runExecution) prepareValidatedRun() bool {
	if err := r.ctx.Err(); err != nil {
		sendLoopCanceled(r.ctx, r.events, r.state, err)
		return false
	}

	executionTools, err := EffectiveTools(r.owner.Tools, r.owner.ToolChoice, r.owner.ToolTransport)
	if err != nil {
		sendLoopError(r.ctx, r.events, r.state, err)
		return false
	}
	r.executionTools = executionTools

	if r.owner.ToolTransport == ToolTransportNative {
		r.toolDefinitions, err = ToolDefinitions(executionTools)
		if err != nil {
			if cancelErr := cancellationError(r.ctx, err); cancelErr != nil {
				sendLoopCanceled(r.ctx, r.events, r.state, cancelErr)
				return false
			}
			sendLoopError(r.ctx, r.events, r.state, err)
			return false
		}
	}

	if _, err = r.owner.PromptBuilder.BuildContext(r.ctx); err != nil {
		if cancelErr := cancellationError(r.ctx, err); cancelErr != nil {
			sendLoopCanceled(r.ctx, r.events, r.state, cancelErr)
			return false
		}
		sendLoopError(r.ctx, r.events, r.state, fmt.Errorf("%w: %w", ErrBuildPrompt, err))
		return false
	}
	if err := r.ctx.Err(); err != nil {
		sendLoopCanceled(r.ctx, r.events, r.state, err)
		return false
	}

	r.requiredToolCallSatisfied = r.owner.ToolChoice.Mode != ai.ToolChoiceRequired
	// Retain the input until an iteration is accepted. A rejected required-tool
	// response consumes an iteration slot but must not lose conversation input.
	r.userMessage = userMessageForIteration(r.owner.PromptBuilder, 0)
	return true
}

func (r *runExecution) runIteration(iterationCount int) iterationOutcome {
	deferTokens := (r.owner.ToolTransport == ToolTransportText && (!r.requiredToolCallSatisfied ||
		(r.owner.ToolChoice.Mode == ai.ToolChoiceRequired && len(r.owner.ToolChoice.Names) > 0))) ||
		(r.owner.ToolTransport == ToolTransportNative && len(r.owner.ToolChoice.Names) > 0)

	var accepted *attemptExecution
	for attemptID := 1; ; attemptID++ {
		attempt, outcome := r.runModelAttempt(iterationCount, attemptID, deferTokens)
		switch outcome {
		case attemptAccepted:
			accepted = attempt
		case attemptRetry:
			continue
		case attemptTerminal:
			return iterationTerminal
		}
		break
	}
	return r.postAttempt(accepted, deferTokens)
}

func (r *runExecution) runModelAttempt(iterationCount, attemptID int, deferTokens bool) (*attemptExecution, attemptOutcome) {
	attemptIteration := Iteration{Count: iterationCount, UserMessage: r.userMessage}
	attemptCtx, state := r.state.startIteration(r.ctx, iterationCount, attemptID)
	attemptCtx, baseCancel := context.WithCancel(attemptCtx)
	attempt := &attemptExecution{
		run:       r,
		ctx:       attemptCtx,
		state:     state,
		iteration: attemptIteration,
		cancel:    baseCancel,
	}

	if err := attemptCtx.Err(); err != nil {
		return attempt, attempt.terminateCanceled(err)
	}
	if err := sendEvent(r.ctx, r.events, AttemptStartEvent(iterationCount, state.attemptID(), r.state.retryCount)); err != nil {
		return attempt, attempt.terminateSendFailure(err)
	}

	request, err := r.owner.buildAttemptRequest(attemptCtx, r.toolDefinitions, r.requiredToolCallSatisfied)
	if err != nil {
		if cancelErr := cancellationError(attemptCtx, err); cancelErr != nil {
			return attempt, attempt.terminateCanceled(cancelErr)
		}
		return attempt, attempt.terminateError(fmt.Errorf("%w: %w", ErrBuildPrompt, err))
	}

	modelCtx := attemptCtx
	if r.owner.RetryPolicy != nil && r.owner.RetryPolicy.AttemptTimeout > 0 {
		var deadlineCancel context.CancelFunc
		attempt.deadline, deadlineCancel = context.WithTimeout(attemptCtx, r.owner.RetryPolicy.AttemptTimeout)
		attempt.cancel = combineCancel(deadlineCancel, baseCancel)
		modelCtx = attempt.deadline
	}

	retrying, outcome, retryErr := attempt.consumeModelStream(modelCtx, request, deferTokens)
	if outcome == attemptTerminal {
		return attempt, outcome
	}

	attemptTimedOut := attempt.timedOut()
	if attemptTimedOut && !retrying {
		retryErr = ErrAttemptTimeout
		if r.owner.RetryPolicy.ShouldRetry(r.state.retryCount, ErrAttemptTimeout) {
			retrying = true
		} else {
			err := fmt.Errorf("%w: limit=%d: %w", ErrMaxRetries, r.owner.RetryPolicy.MaxRetries, ErrAttemptTimeout)
			return attempt, attempt.terminateError(err)
		}
	}
	if err := attemptCtx.Err(); err != nil && !(retrying && attemptTimedOut) {
		return attempt, attempt.terminateCanceled(err)
	}
	if !retrying {
		return attempt, attemptAccepted
	}
	return attempt, attempt.scheduleRetry(retryErr)
}

func (a *attemptExecution) consumeModelStream(modelCtx context.Context, request ai.AIRequest, deferTokens bool) (bool, attemptOutcome, error) {
	tokens := a.run.owner.Model.GenerateStream(modelCtx, request)
	for token := range tokens {
		if token.Err != nil {
			retryErr := token.Err
			attemptTimedOut := a.timedOut()
			var providerErr *ai.ProviderError
			if attemptTimedOut && errors.Is(token.Err, context.DeadlineExceeded) && !errors.As(token.Err, &providerErr) {
				token.Err = ErrAttemptTimeout
				retryErr = token.Err
			} else if !attemptTimedOut {
				if cancelErr := cancellationError(a.ctx, token.Err); cancelErr != nil {
					return false, a.terminateCanceled(cancelErr), nil
				}
			}

			retryLimit := 0
			retryable := false
			canRetry := false
			if a.run.owner.RetryPolicy != nil {
				retryLimit = a.run.owner.RetryPolicy.MaxRetries
				retryable = a.run.owner.RetryPolicy.isRetryable(token.Err)
				canRetry = a.run.owner.RetryPolicy.hasRetryBudget(a.run.state.retryCount) && retryable
			}
			if canRetry {
				return true, attemptRetry, retryErr
			}

			terminalErr := token.Err
			if a.run.owner.RetryPolicy != nil && retryable {
				terminalErr = fmt.Errorf("%w: limit=%d: %w", ErrMaxRetries, retryLimit, token.Err)
			}
			return false, a.terminateError(terminalErr), nil
		}

		if token.Type == ai.TokenTypeToolCall && a.run.owner.ToolChoice.Mode == ai.ToolChoiceNone {
			// A provider can still emit a tool-call token after tools are disabled.
			// Do not expose or retain a disabled call.
			continue
		}
		if token.Type == ai.TokenTypeToolCall && token.ToolCall != nil {
			a.iteration.AppendToken(token)
			a.toolCalls = append(a.toolCalls, pendingToolCall{
				partIndex: len(a.iteration.Parts) - 1,
				call:      *token.ToolCall,
			})
		} else {
			a.iteration.AppendToken(token)
		}
		if deferTokens {
			a.deferredTokens = append(a.deferredTokens, token)
			continue
		}
		a.run.state.recordToken(token)
		a.state.recordToken(token)
		if err := sendEvent(a.run.ctx, a.run.events, TokenEvent(a.iteration.Count, a.state.attemptID(), a.run.state.retryCount, token)); err != nil {
			return false, a.terminateSendFailure(err), nil
		}
	}
	return false, attemptAccepted, nil
}

func (a *attemptExecution) scheduleRetry(retryErr error) attemptOutcome {
	a.run.state.retry()
	delay := time.Duration(0)
	if a.run.owner.RetryPolicy != nil {
		delay = a.run.owner.RetryPolicy.Backoff(a.run.state.retryCount-1, retryErr)
	}
	a.state.recordIteration(a.iteration)
	a.state.markRetrying(a.run.state.retryCount, retryReason(retryErr), delay)
	if err := sendEvent(a.run.ctx, a.run.events, RetryEvent(a.iteration.Count, a.state.attemptID(), a.run.state.retryCount, retryReason(retryErr), delay, a.iteration)); err != nil {
		return a.terminateSendFailure(err)
	}
	if a.run.owner.RetryPolicy != nil && delay > 0 {
		// The backoff is scoped to the run context, not the failed attempt.
		a.cancelAttempt()
		if err := a.run.owner.RetryPolicy.wait(a.run.ctx, delay); err != nil {
			if cancelErr := cancellationError(a.run.ctx, err); cancelErr != nil {
				return a.terminateCanceled(cancelErr)
			}
			return a.terminateError(fmt.Errorf("wait for retry: %w", err))
		}
	}
	a.cancelAttempt()
	a.finish(nil)
	return attemptRetry
}

func (r *runExecution) postAttempt(attempt *attemptExecution, deferTokens bool) iterationOutcome {
	if deferTokens && (!r.requiredToolCallSatisfied || len(attempt.toolCalls) > 0) &&
		!hasPermittedToolCall(attempt.toolCalls, r.owner.Tools, r.owner.ToolChoice.Names) {
		// A text-transport response that does not satisfy a required tool call is
		// not part of the conversation and must not expose its response content.
		attempt.cancelAttempt()
		if err := sendEvent(r.ctx, r.events, DiscardEvent(attempt.iteration.Count, attempt.state.attemptID(), r.state.retryCount, attempt.iteration)); err != nil {
			attempt.terminateSendFailure(err)
			return iterationTerminal
		}
		r.state.resetRetries()
		attempt.finish(nil)
		return iterationContinue
	}
	if r.owner.ToolChoice.Mode == ai.ToolChoiceNone {
		attempt.toolCalls = nil
	}
	for _, token := range attempt.deferredTokens {
		r.state.recordToken(token)
		attempt.state.recordToken(token)
		if err := sendEvent(r.ctx, r.events, TokenEvent(attempt.iteration.Count, attempt.state.attemptID(), r.state.retryCount, token)); err != nil {
			attempt.terminateSendFailure(err)
			return iterationTerminal
		}
	}

	if err := r.owner.executeToolCalls(attempt.ctx, &attempt.iteration, attempt.toolCalls, r.executionTools, r.events, attempt.iteration.Count, attempt.state.attemptID(), r.state.retryCount); err != nil {
		if cancelErr := cancellationError(attempt.ctx, err); cancelErr != nil {
			attempt.terminateCanceled(cancelErr)
		} else {
			attempt.terminateError(err)
		}
		return iterationTerminal
	}
	if err := attempt.ctx.Err(); err != nil {
		attempt.terminateCanceled(err)
		return iterationTerminal
	}
	attempt.state.recordToolResponses(attempt.iteration)

	attemptID := attempt.state.attemptID()
	retryCount := r.state.retryCount
	if err := sendEvent(r.ctx, r.events, IterationDoneEvent(attempt.iteration, attemptID, retryCount)); err != nil {
		attempt.terminateSendFailure(err)
		return iterationTerminal
	}
	// Persistence deliberately follows IterationDone emission.
	r.owner.Iterations = append(r.owner.Iterations, attempt.iteration)
	r.userMessage = nil
	if hasPermittedToolCall(attempt.toolCalls, r.owner.Tools, r.owner.ToolChoice.Names) {
		r.requiredToolCallSatisfied = true
	}
	r.state.resetRetries()

	if r.owner.ToolTransport == ToolTransportText && len(attempt.toolCalls) == 0 && !r.requiredToolCallSatisfied {
		attempt.cancelAttempt()
		attempt.finish(nil)
		return iterationContinue
	}
	if len(attempt.toolCalls) == 0 {
		attempt.cancelAttempt()
		attempt.state.markFinal()
		attempt.finish(nil)
		if err := sendEvent(r.ctx, r.events, DoneEvent()); err != nil {
			if cancelErr := cancellationError(r.ctx, err); cancelErr != nil {
				sendLoopCanceled(r.ctx, r.events, r.state, cancelErr)
			}
		}
		return iterationDone
	}
	attempt.cancelAttempt()
	attempt.finish(nil)
	return iterationContinue
}

func (a *attemptExecution) timedOut() bool {
	return a.deadline != nil && errors.Is(a.deadline.Err(), context.DeadlineExceeded) &&
		a.run.callerCtx.Err() == nil && a.run.ctx.Err() == nil
}

func (a *attemptExecution) terminateSendFailure(err error) attemptOutcome {
	if cancelErr := cancellationError(a.ctx, err); cancelErr != nil {
		return a.terminateCanceled(cancelErr)
	}
	sendAttemptError(a.run.ctx, a.run.events, a.run.state, a.iteration.Count, a.state.attemptID(), a.run.state.retryCount, &a.iteration, err)
	a.cancelAttempt()
	a.finish(nil)
	return attemptTerminal
}

func (a *attemptExecution) terminateError(err error) attemptOutcome {
	sendAttemptError(a.run.ctx, a.run.events, a.run.state, a.iteration.Count, a.state.attemptID(), a.run.state.retryCount, &a.iteration, err)
	a.cancelAttempt()
	a.finish(err)
	return attemptTerminal
}

func (a *attemptExecution) terminateCanceled(err error) attemptOutcome {
	sendAttemptCanceled(a.run.ctx, a.run.events, a.run.state, a.iteration.Count, a.state.attemptID(), a.run.state.retryCount, &a.iteration, err)
	a.cancelAttempt()
	a.state.markCanceled(err)
	a.finish(nil)
	return attemptTerminal
}

func (a *attemptExecution) cancelAttempt() {
	if a == nil || a.cancel == nil {
		return
	}
	a.cancelOnce.Do(a.cancel)
}

func (a *attemptExecution) finish(err error) {
	if a == nil || a.state == nil {
		return
	}
	a.finishOnce.Do(func() { a.state.finish(err) })
}

func combineCancel(cancel ...context.CancelFunc) context.CancelFunc {
	return func() {
		for _, fn := range cancel {
			if fn != nil {
				fn()
			}
		}
	}
}
