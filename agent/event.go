package agent

import (
	"encoding/json"
	"time"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
)

// EventType identifies one ordered workflow event.
type EventType string

const (
	// EventOutput contributes to externally visible workflow output. Every
	// EventOutput is visible regardless of its source index.
	EventOutput EventType = "output"

	EventAttemptStart  EventType = "attempt_start"
	EventRetry         EventType = "retry"
	EventDiscard       EventType = "discard"
	EventIterationDone EventType = "iteration_done"

	EventToolStart  EventType = "tool_start"
	EventToolResult EventType = "tool_result"
	EventToolError  EventType = "tool_error"

	EventStageStart  EventType = "stage_start"
	EventStageFinish EventType = "stage_finish"

	EventDone     EventType = "done"
	EventError    EventType = "error"
	EventCanceled EventType = "canceled"
)

// EventSourceKind identifies the logical kind of event producer.
type EventSourceKind string

const (
	SourceWorkflow   EventSourceKind = "workflow"
	SourcePrimary    EventSourceKind = "primary"
	SourceMiddleware EventSourceKind = "middleware"
)

// EventSource identifies provenance and the attempt namespace within one
// workflow. Index is local to one execution: zero is the primary agent and
// one through n are middleware in declaration order. It is neither a
// visibility flag nor a durable recovery identity.
type EventSource struct {
	Kind  EventSourceKind
	Index int
	Name  string
}

// OutputKind identifies the semantic kind of visible workflow output.
type OutputKind string

const (
	OutputText      OutputKind = "text"
	OutputReasoning OutputKind = "reasoning"
	OutputData      OutputKind = "data"
)

// OutputPart is one externally visible item in final workflow output.
type OutputPart struct {
	Kind OutputKind
	Text string
	Data *StructuredOutput
}

// StructuredOutput contains application-defined structured output.
type StructuredOutput struct {
	Name string
	JSON json.RawMessage
}

// StageOutcome identifies how a primary or middleware-agent stage ended.
type StageOutcome string

const (
	StageSucceeded StageOutcome = "succeeded"
	StageFailed    StageOutcome = "failed"
	StageCanceled  StageOutcome = "canceled"
	StageSkipped   StageOutcome = "skipped"
)

// Event is one immutable snapshot from the ordered, middleware-aware workflow
// stream. Fields are meaningful according to Type.
type Event struct {
	Type   EventType
	Source EventSource

	IterationCount int
	AttemptID      int
	RetryCount     int
	PartCount      int

	Output *OutputPart

	Iteration *loop.Iteration

	ToolCall     *ai.ToolCall
	ToolResponse *loop.ToolResponse

	RetryReason string
	RetryDelay  time.Duration
	Duration    time.Duration

	StageOutcome StageOutcome
	StageReason  string
	Err          error
}

func cloneOutputPart(part OutputPart) OutputPart {
	cloned := part
	if part.Data != nil {
		data := *part.Data
		data.JSON = append(json.RawMessage(nil), part.Data.JSON...)
		cloned.Data = &data
	}
	return cloned
}

func cloneOutputParts(parts []OutputPart) []OutputPart {
	if parts == nil {
		return nil
	}
	cloned := make([]OutputPart, len(parts))
	for i := range parts {
		cloned[i] = cloneOutputPart(parts[i])
	}
	return cloned
}

func cloneEvent(event Event) Event {
	cloned := event
	if event.Output != nil {
		output := cloneOutputPart(*event.Output)
		cloned.Output = &output
	}
	if event.Iteration != nil {
		iteration := cloneIterations([]loop.Iteration{*event.Iteration})[0]
		cloned.Iteration = &iteration
	}
	cloned.ToolCall = cloneToolCall(event.ToolCall)
	cloned.ToolResponse = cloneToolResponse(event.ToolResponse)
	return cloned
}
