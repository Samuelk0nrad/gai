package agent

import (
	"bytes"
	"testing"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
)

func TestWorkflowOutputAccumulatorDoesNotRetainLargeExecutionEvents(t *testing.T) {
	payload := bytes.Repeat([]byte("x"), 64<<10)
	response := string(payload)
	event := Event{
		Type: EventToolResult,
		Iteration: &loop.Iteration{Parts: []loop.IterationPart{{
			Response: &ai.AIResponse{Raw: payload},
			ToolReq:  &ai.ToolCall{Args: payload},
			ToolResp: loop.NewToolSuccess(response),
		}}},
		ToolCall:     &ai.ToolCall{Args: payload},
		ToolResponse: loop.NewToolSuccess(response),
	}

	var retained int
	allocations := testing.AllocsPerRun(5, func() {
		var accumulator workflowOutputAccumulator
		for range 64 {
			accumulator.add(event)
		}
		retained = len(accumulator.outputs) + len(accumulator.invalid) + len(accumulator.stageErrs)
	})

	if retained != 0 {
		t.Fatalf("retained reduction entries = %d, want 0", retained)
	}
	if allocations != 0 {
		t.Fatalf("allocations while reducing non-output events = %f, want 0", allocations)
	}
}
