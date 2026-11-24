package loop

import (
	"context"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
)

// A processor that edits media in place must still report a changed output.
func TestInPlaceMediaEditIsReportedAsChanged(t *testing.T) {
	tool, err := NewResultTool("test", "Test.", ai.ToolParameters{}, func(context.Context, ai.ToolCall) (ToolResult, error) {
		return ToolResult{Text: "file", Media: []ai.MediaPart{{MIMEType: "image/png", Data: []byte{1, 2, 3}, Name: "a.png"}}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	processor := ToolResultProcessorFunc(func(_ context.Context, _ ToolPolicyInput, result ToolResult) (ToolResult, error) {
		result.Media[0].Name = "b.png"
		result.Media[0].Data[0] = 9
		return result, nil
	})
	l := &Loop{Tools: []Tool{tool}, ToolResultProcessor: processor}
	iteration, calls := schedulerCalls("test")
	events := make(chan Event, 8)
	if err := l.executeToolCalls(t.Context(), iteration, calls, l.Tools, events, 1, 1, 0); err != nil {
		t.Fatal(err)
	}
	if execution := iteration.Parts[0].ToolExecution; execution.State != ToolSucceeded || execution.Output != ToolOutputChanged {
		t.Fatalf("execution=%#v", execution)
	}
	parts := iteration.Conversation[0].Parts[0].ToolResult.Parts
	if len(parts) != 2 || parts[1].Media == nil || parts[1].Media.Name != "b.png" || parts[1].Media.Data[0] != 9 {
		t.Fatalf("parts=%#v", parts)
	}
}
