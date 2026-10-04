package loop

import (
	"context"
	"time"

	"github.com/lace-ai/gai/ai"
)

// EventType identifies the kind of ordered loop event.
type EventType string

const (
	// EventToken carries a streamed model token.
	EventToken EventType = "token"
	// EventAttemptStart reports that a model generation attempt has started.
	EventAttemptStart EventType = "attempt_start"
	// EventRetry reports that a failed attempt will be retried.
	EventRetry EventType = "retry"
	// EventDiscard reports an attempt that is excluded from the conversation but
	// remains available for lifecycle and billing accounting.
	EventDiscard EventType = "discard"
	// EventIterationDone carries a completed agent iteration.
	EventIterationDone EventType = "iteration_done"
	// EventToolDecision reports a policy decision before admission.
	EventToolDecision EventType = "tool_decision"
	// EventToolStart reports that a tool invocation has started.
	EventToolStart EventType = "tool_start"
	// EventToolResult reports a successful tool completion.
	EventToolResult EventType = "tool_result"
	// EventToolError reports a failed tool completion.
	EventToolError EventType = "tool_error"
	// EventDone reports successful loop completion.
	EventDone EventType = "done"
	// EventError reports terminal loop failure.
	EventError EventType = "error"
	// EventCanceled reports terminal cancellation without treating it as a failure.
	EventCanceled EventType = "canceled"
)

// Event is one item from the loop's ordered execution stream.
type Event struct {
	Type EventType

	IterationCount int
	AttemptID      int
	RetryCount     int
	PartCount      int
	// RetryReason classifies the error that triggered a retry.
	RetryReason string
	// RetryDelay is the backoff selected before the next attempt.
	RetryDelay time.Duration

	Token         *ai.Token
	Iteration     *Iteration
	ToolCall      *ai.ToolCall
	ToolResult    *ToolResult
	ToolExecution *ToolExecution
	Duration      time.Duration
	Err           error
}

func AttemptStartEvent(iteration, attempt, retry int) Event {
	return Event{
		Type:           EventAttemptStart,
		IterationCount: iteration,
		AttemptID:      attempt,
		RetryCount:     retry,
	}
}

func TokenEvent(iteration, attempt, retry int, token ai.Token) Event {
	return Event{
		Type:           EventToken,
		IterationCount: iteration,
		AttemptID:      attempt,
		RetryCount:     retry,
		Token:          &token,
	}
}

func RetryEvent(iteration, attempt, retry int, reason string, delay time.Duration, attemptIteration Iteration) Event {
	return Event{
		Type:           EventRetry,
		IterationCount: iteration,
		AttemptID:      attempt,
		RetryCount:     retry,
		RetryReason:    reason,
		RetryDelay:     delay,
		PartCount:      len(attemptIteration.Parts),
		Iteration:      iterationSnapshot(attemptIteration),
	}
}

// DiscardEvent reports a rejected attempt without exposing its response content.
func DiscardEvent(iteration, attempt, retry int, attemptIteration Iteration) Event {
	event := Event{
		Type:           EventDiscard,
		IterationCount: iteration,
		AttemptID:      attempt,
		RetryCount:     retry,
		PartCount:      len(attemptIteration.Parts),
		Iteration: &Iteration{
			Count:         attemptIteration.Count,
			Usage:         attemptIteration.Usage,
			UsageReported: attemptIteration.UsageReported,
		},
	}
	if attemptIteration.RequestBudget != nil {
		budget := *attemptIteration.RequestBudget
		event.Iteration.RequestBudget = &budget
	}
	return event
}

func IterationDoneEvent(iteration Iteration, attempt, retry int) Event {
	return Event{
		Type:           EventIterationDone,
		IterationCount: iteration.Count,
		AttemptID:      attempt,
		RetryCount:     retry,
		PartCount:      len(iteration.Parts),
		Iteration:      iterationSnapshot(iteration),
	}
}

func DoneEvent() Event {
	return Event{Type: EventDone}
}

// ToolStartEvent announces admission of a handler invocation.
func ToolStartEvent(iteration, attempt, retry int, call ai.ToolCall) Event {
	return Event{Type: EventToolStart, IterationCount: iteration, AttemptID: attempt, RetryCount: retry, ToolCall: &call}
}

// ToolResultEvent reports processed output, selecting ToolError when Err is non-nil.
func ToolResultEvent(iteration, attempt, retry int, call ai.ToolCall, response *ToolResult, duration time.Duration) Event {
	eventType := EventToolResult
	var err error
	if response != nil && response.Err != nil {
		eventType, err = EventToolError, response.Err
	}
	return Event{Type: eventType, IterationCount: iteration, AttemptID: attempt, RetryCount: retry, ToolCall: &call, ToolResult: response, Duration: duration, Err: err}
}

// ToolErrorEvent reports a terminal tool-pipeline error with optional safe output.
func ToolErrorEvent(iteration, attempt, retry int, call ai.ToolCall, response *ToolResult, duration time.Duration, err error) Event {
	return Event{Type: EventToolError, IterationCount: iteration, AttemptID: attempt, RetryCount: retry, ToolCall: &call, ToolResult: response, Duration: duration, Err: err}
}

func ErrorEvent(err error) Event {
	return Event{Type: EventError, Err: err}
}

// CanceledEvent reports terminal cancellation that occurred outside a specific
// model generation attempt.
func CanceledEvent(err error) Event {
	return Event{Type: EventCanceled, Err: err}
}

// AttemptErrorEvent reports a terminal error that occurred inside a specific
// model generation attempt.
func AttemptErrorEvent(iterationCount, attemptID, retryCount int, attemptIteration *Iteration, err error) Event {
	return attemptTerminalEvent(EventError, iterationCount, attemptID, retryCount, attemptIteration, err)
}

// AttemptCanceledEvent reports terminal cancellation that occurred inside a
// specific model generation attempt.
func AttemptCanceledEvent(iterationCount, attemptID, retryCount int, attemptIteration *Iteration, err error) Event {
	return attemptTerminalEvent(EventCanceled, iterationCount, attemptID, retryCount, attemptIteration, err)
}

func attemptTerminalEvent(eventType EventType, iterationCount, attemptID, retryCount int, attemptIteration *Iteration, err error) Event {
	event := Event{
		Type:           eventType,
		IterationCount: iterationCount,
		AttemptID:      attemptID,
		RetryCount:     retryCount,
		Err:            err,
	}
	if attemptIteration != nil {
		iteration := attemptIteration.Clone()
		event.PartCount = len(iteration.Parts)
		event.Iteration = &iteration
	}
	return event
}

// sendEvent copies mutable tool payloads and waits for delivery or cancellation.
func sendEvent(ctx context.Context, ch chan<- Event, event Event) error {
	event.ToolExecution = cloneToolExecution(event.ToolExecution)
	if event.Token != nil {
		t := event.Token.Clone()
		event.Token = &t
	}
	if event.ToolCall != nil {
		c := event.ToolCall.Clone()
		event.ToolCall = &c
	}
	if event.ToolResult != nil {
		r := *event.ToolResult
		event.ToolResult = &r
	}

	select {
	case ch <- event:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func iterationSnapshot(i Iteration) *Iteration { snapshot := i.Clone(); return &snapshot }
