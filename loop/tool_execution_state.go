package loop

import (
	"context"
	"errors"
)

// ToolExecutionState describes invocation independently of policy/output errors.
// A timeout or failed call does not prove external side effects did not happen.
type ToolExecutionState string

const (
	ToolNotStarted ToolExecutionState = "not_started"
	ToolRunning    ToolExecutionState = "running"
	ToolSucceeded  ToolExecutionState = "succeeded"
	ToolFailed     ToolExecutionState = "failed"
	ToolTimedOut   ToolExecutionState = "timed_out"
	ToolCanceled   ToolExecutionState = "canceled"
)

type ToolOutputState string

const (
	ToolOutputAccepted ToolOutputState = "accepted"
	ToolOutputChanged  ToolOutputState = "changed"
	ToolOutputRejected ToolOutputState = "rejected"
)

// ToolExecution is a copy-safe execution record, distinct from handler results.
// Decision can be absent for malformed or unresolved calls. Output describes only
// the final publication; a successful side effect may have rejected output.
type ToolExecution struct {
	Decision ToolDecision
	State    ToolExecutionState
	Output   ToolOutputState
}

func cloneToolExecution(value *ToolExecution) *ToolExecution {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}
func invocationState(err error) ToolExecutionState {
	switch {
	case err == nil:
		return ToolSucceeded
	case errors.Is(err, context.DeadlineExceeded):
		return ToolTimedOut
	case errors.Is(err, context.Canceled):
		return ToolCanceled
	default:
		return ToolFailed
	}
}
func outputState(before ToolResult, after *ToolResult) ToolOutputState {
	if after == nil || errors.Is(after.Err, ErrToolResultRejected) {
		return ToolOutputRejected
	}
	if (before.Err != nil) != (after.Err != nil) || before.String() != after.String() {
		return ToolOutputChanged
	}
	return ToolOutputAccepted
}
