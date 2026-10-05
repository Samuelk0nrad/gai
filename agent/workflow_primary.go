package agent

import (
	"context"
	"errors"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
)

type primaryAccumulator struct {
	attempted        []ai.Token
	accepted         []loop.Iteration
	outputs          []Event
	acceptedAttempts map[attemptKey]struct{}
	billed           ai.Usage
	billedKey        map[attemptKey]struct{}
	errs             []error
	canceled         bool
	cancelErr        error
}

func (a *primaryAccumulator) recordOutput(event Event) {
	switch event.Type {
	case EventOutput:
		a.outputs = append(a.outputs, cloneEvent(event))
	case EventIterationDone:
		if a.acceptedAttempts == nil {
			a.acceptedAttempts = make(map[attemptKey]struct{})
		}
		a.acceptedAttempts[eventAttemptKey(event)] = struct{}{}
	}
}

func (a *primaryAccumulator) acceptedOutput() []Event {
	var accepted []Event
	for _, event := range a.outputs {
		if _, ok := a.acceptedAttempts[eventAttemptKey(event)]; ok {
			accepted = append(accepted, cloneEvent(event))
		}
	}
	return accepted
}

func (a *primaryAccumulator) account(event Event) {
	if event.Iteration == nil {
		return
	}
	key := eventAttemptKey(event)
	if _, exists := a.billedKey[key]; exists {
		return
	}
	a.billedKey[key] = struct{}{}
	a.billed.Add(event.Iteration.Usage)
}

func (a *primaryAccumulator) result() AgentResult {
	tokens := iterationTokens(a.accepted)
	var messages []ai.Message
	for _, iteration := range a.accepted {
		messages = append(messages, iteration.Messages()...)
	}
	return AgentResult{
		Tokens:          tokens,
		Text:            tokenText(tokens),
		Reasoning:       tokenReasoning(tokens),
		AttemptedTokens: cloneTokens(a.attempted),
		AttemptedText:   tokenText(a.attempted),
		Messages:        cloneMessages(messages),
		Iterations:      cloneIterations(a.accepted),
		Usage:           iterationUsage(a.accepted),
		BilledUsage:     a.billed,
		Errors:          append([]error(nil), a.errs...),
		Canceled:        a.canceled,
		CancellationErr: a.cancelErr,
	}
}

func (w *Workflow) mapPrimary(ctx context.Context, upstream <-chan loop.Event, obs *workflowObserver) <-chan Event {
	out := make(chan Event)
	source := EventSource{Kind: SourcePrimary, Index: 0, Name: w.name}
	go func() {
		defer close(out)
		defer close(w.primaryDone)
		acc := primaryAccumulator{billedKey: make(map[attemptKey]struct{})}
		deliver := w.sendEvent(ctx, out, Event{Type: EventStageStart, Source: source}, true)
		var terminal Event
		for low := range upstream {
			event, emit := mapLoopEvent(low, source)
			if emit {
				acc.recordOutput(event)
			}
			if low.Type == loop.EventToken && low.Token != nil {
				acc.attempted = append(acc.attempted, cloneTokens([]ai.Token{*low.Token})[0])
			}
			switch low.Type {
			case loop.EventRetry, loop.EventDiscard:
				acc.account(event)
			case loop.EventIterationDone:
				acc.account(event)
				if low.Iteration != nil {
					acc.accepted = append(acc.accepted, cloneIterations([]loop.Iteration{*low.Iteration})[0])
				}
			case loop.EventError:
				if low.Err != nil {
					acc.errs = append(acc.errs, low.Err)
				}
				terminal = Event{Type: EventStageFinish, Source: source, IterationCount: low.IterationCount, AttemptID: low.AttemptID, RetryCount: low.RetryCount, PartCount: low.PartCount, Iteration: cloneIterationPtr(low.Iteration), StageOutcome: StageFailed, Err: low.Err}
				acc.account(terminal)
			case loop.EventCanceled:
				acc.canceled = true
				acc.cancelErr = low.Err
				terminal = Event{Type: EventStageFinish, Source: source, IterationCount: low.IterationCount, AttemptID: low.AttemptID, RetryCount: low.RetryCount, PartCount: low.PartCount, Iteration: cloneIterationPtr(low.Iteration), StageOutcome: StageCanceled, Err: low.Err}
				acc.account(terminal)
			case loop.EventDone:
				terminal = Event{Type: EventStageFinish, Source: source, StageOutcome: StageSucceeded}
			}
			if emit {
				deliver = w.sendEvent(ctx, out, cloneEvent(event), deliver)
			}
		}
		if terminal.Type == "" {
			terminal = Event{Type: EventStageFinish, Source: source, StageOutcome: StageFailed, Err: errors.New("loop stream closed without a terminal event")}
			acc.errs = append(acc.errs, terminal.Err)
		}
		primary := acc.result()
		w.mu.Lock()
		w.result.Primary = cloneAgentResult(primary)
		w.result.Usage = primary.Usage
		w.result.BilledUsage = primary.BilledUsage
		w.result.Errors = append([]error(nil), primary.Errors...)
		w.result.Canceled = primary.Canceled
		w.result.CancellationErr = primary.CancellationErr
		w.primaryOutput = acc.acceptedOutput()
		w.setVisibleOutputLocked(outputPartsFromTokens(primary.Tokens))
		w.result.AttemptedTokens = cloneTokens(primary.AttemptedTokens)
		w.result.AttemptedText = primary.AttemptedText
		w.mu.Unlock()
		obs.PrimaryFinished(ctx, primary)
		w.sendEvent(ctx, out, cloneEvent(terminal), deliver)
	}()
	return out
}

func (w *Workflow) canonicalPrimaryOutput() []Event {
	w.mu.RLock()
	defer w.mu.RUnlock()
	output := make([]Event, len(w.primaryOutput))
	for i, event := range w.primaryOutput {
		output[i] = cloneEvent(event)
	}
	return output
}

// mapLoopEvent converts loop lifecycle events into isolated workflow events.
func mapLoopEvent(low loop.Event, source EventSource) (Event, bool) {
	event := Event{
		Source:         source,
		IterationCount: low.IterationCount,
		AttemptID:      low.AttemptID,
		RetryCount:     low.RetryCount,
		PartCount:      low.PartCount,
		Iteration:      cloneIterationPtr(low.Iteration),
		ToolCall:       cloneToolCall(low.ToolCall),
		ToolResult:     cloneToolResult(low.ToolResult),
		RetryReason:    low.RetryReason,
		RetryDelay:     low.RetryDelay,
		Duration:       low.Duration,
		Err:            low.Err,
	}
	if low.ToolExecution != nil {
		execution := *low.ToolExecution
		event.ToolExecution = &execution
	}
	if low.ToolApproval != nil {
		request := low.ToolApproval.Clone()
		event.ToolApproval = &request
	}
	switch low.Type {
	case loop.EventToolApprovalRequested:
		event.Type = EventToolApprovalRequested
	case loop.EventToolApprovalResolved:
		event.Type = EventToolApprovalResolved
	case loop.EventToolDecision:
		event.Type = EventToolDecision
	case loop.EventAttemptStart:
		event.Type = EventAttemptStart
	case loop.EventToken:
		if low.Token == nil {
			return Event{}, false
		}
		text := low.Token.Text()
		switch low.Token.Type() {
		case ai.TokenTypeText:
			event.Type = EventOutput
			event.Output = &OutputPart{Kind: OutputText, Text: text}
		case ai.TokenTypeThought:
			event.Type = EventOutput
			event.Output = &OutputPart{Kind: OutputReasoning, Text: text}
		default:
			return Event{}, false
		}
	case loop.EventRetry:
		event.Type = EventRetry
	case loop.EventDiscard:
		event.Type = EventDiscard
	case loop.EventIterationDone:
		event.Type = EventIterationDone
	case loop.EventToolStart:
		event.Type = EventToolStart
	case loop.EventToolResult:
		event.Type = EventToolResult
	case loop.EventToolError:
		event.Type = EventToolError
	default:
		return Event{}, false
	}
	return event, true
}
