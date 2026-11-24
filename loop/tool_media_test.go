package loop_test

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/loop"
)

func TestTextProcessorsKeepMedia(t *testing.T) {
	media := []ai.MediaPart{{MIMEType: "image/png", Data: []byte{1, 2, 3}, Name: "a.png"}}
	redact, err := loop.RedactToolResult(func(_ context.Context, text string) (string, error) {
		return strings.ReplaceAll(text, "secret", "[redacted]"), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	limit, err := loop.LimitToolResultBytes(64)
	if err != nil {
		t.Fatal(err)
	}
	chain, err := loop.ChainToolResultProcessors(redact, limit)
	if err != nil {
		t.Fatal(err)
	}
	input := loop.ToolPolicyInput{Call: ai.ToolCall{ID: "a", Type: "function", Name: "open_file", Args: []byte(`{}`)}}
	got, err := chain.Process(t.Context(), input, loop.ToolResult{Text: "secret file", Media: media})
	if err != nil {
		t.Fatal(err)
	}
	if got.Text != "[redacted] file" || !reflect.DeepEqual(got.Media, media) {
		t.Fatalf("processed = %+v", got)
	}
	// The byte limit applies to text only, so large media passes.
	big := []ai.MediaPart{{MIMEType: "image/png", Data: make([]byte, 1<<20)}}
	if got, err := limit.Process(t.Context(), input, loop.ToolResult{Text: "ok", Media: big}); err != nil || got.Err != nil || len(got.Media) != 1 {
		t.Fatalf("limit = %+v, %v", got, err)
	}
	// Truncation cuts the text and keeps the media.
	truncate, err := loop.TruncateToolResultBytes(loop.MinTruncateToolResultBytes)
	if err != nil {
		t.Fatal(err)
	}
	got, err = truncate.Process(t.Context(), input, loop.ToolResult{Text: strings.Repeat("x", 1024), Media: media})
	if err != nil || got.Err != nil || len(got.Text) > loop.MinTruncateToolResultBytes || !strings.Contains(got.Text, "[tool result truncated") || !reflect.DeepEqual(got.Media, media) {
		t.Fatalf("truncate = %+v, %v", got, err)
	}
}

func TestResultToolFunctionReturnsTextOnly(t *testing.T) {
	tool, err := loop.NewResultTool("open_file", "Open.", ai.ToolParameters{}, func(context.Context, ai.ToolCall) (loop.ToolResult, error) {
		return loop.ToolResult{Text: "a.png", Media: []ai.MediaPart{{MIMEType: "image/png", URI: "https://example.test/a.png"}}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := tool.(loop.ResultTool); !ok {
		t.Fatal("NewResultTool did not return a ResultTool")
	}
	call := ai.ToolCall{ID: "a", Type: "function", Name: "open_file", Args: []byte(`{}`)}
	if text, err := tool.Function(t.Context(), call); err != nil || text != "a.png" {
		t.Fatalf("Function = %q, %v", text, err)
	}
	result := loop.InvokeTool(t.Context(), tool, call)
	if result.Text != "a.png" || len(result.Media) != 1 || result.Media[0].URI != "https://example.test/a.png" {
		t.Fatalf("InvokeTool = %+v", result)
	}
	text, err := loop.NewTool("echo", "Echo.", ai.ToolParameters{}, func(context.Context, ai.ToolCall) (string, error) { return "plain", nil })
	if err != nil {
		t.Fatal(err)
	}
	if result := loop.InvokeTool(t.Context(), text, call); result.Text != "plain" || result.Media != nil || result.Err != nil {
		t.Fatalf("text InvokeTool = %+v", result)
	}
}
