package loop

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/obstest"
)

func TestTruncateToolResultBytesValidatesBudget(t *testing.T) {
	for _, limit := range []int{-1, 1, MinToolResultBytes, MinTruncateToolResultBytes - 1} {
		if _, err := TruncateToolResultBytes(limit); !errors.Is(err, ErrToolResultProcess) {
			t.Fatalf("limit %d accepted: %v", limit, err)
		}
	}
	for _, limit := range []int{0, MinTruncateToolResultBytes} {
		if _, err := TruncateToolResultBytes(limit); err != nil {
			t.Fatalf("limit %d rejected: %v", limit, err)
		}
	}
	// The longest possible note must leave room for content at the minimum budget.
	if note := truncationNote(int(^uint(0)>>1), int(^uint(0)>>1)); len(note) >= MinTruncateToolResultBytes {
		t.Fatalf("note %d bytes does not fit minimum budget", len(note))
	}
}

func TestTruncateToolResultBytesCutsSuccessfulText(t *testing.T) {
	const limit = 200
	for _, tc := range []struct {
		name string
		text string
	}{
		{"ascii", strings.Repeat("x", 1000)},
		{"json", `{"items":[` + strings.Repeat(`{"id":1,"name":"item"},`, 100) + `]}`},
		{"two-byte", strings.Repeat("é", 500)},
		{"three-byte", strings.Repeat("€", 500)},
		{"four-byte", strings.Repeat("😀", 500)},
		{"mixed", strings.Repeat("a€😀é", 100)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			processor, _ := TruncateToolResultBytes(limit)
			result, err := processor.Process(t.Context(), ToolPolicyInput{}, ToolResult{Text: tc.text})
			if err != nil || result.Err != nil {
				t.Fatalf("result=%#v err=%v", result, err)
			}
			if len(result.Text) > limit {
				t.Fatalf("len=%d exceeds %d", len(result.Text), limit)
			}
			if !utf8.ValidString(result.Text) {
				t.Fatalf("invalid UTF-8: %q", result.Text)
			}
			kept, _, ok := strings.Cut(result.Text, "\n\n[tool result truncated:")
			if !ok || !strings.HasPrefix(tc.text, kept) {
				t.Fatalf("result does not keep the start: %q", result.Text)
			}
			if want := kept + truncationNote(len(kept), len(tc.text)); result.Text != want {
				t.Fatalf("note=%q want %q", result.Text, want)
			}
			// Cutting may only back off a partial rune, never more.
			if limit-len(result.Text) >= utf8.UTFMax {
				t.Fatalf("kept %d bytes, wasted %d", len(kept), limit-len(result.Text))
			}
			again, _ := processor.Process(t.Context(), ToolPolicyInput{}, result)
			if again != result {
				t.Fatal("reprocessing a truncated result changed it")
			}
			repeat, _ := processor.Process(t.Context(), ToolPolicyInput{}, ToolResult{Text: tc.text})
			if repeat != result {
				t.Fatal("truncation is not deterministic")
			}
		})
	}
}

func TestTruncateToolResultBytesKeepsFittingAndDisabledResults(t *testing.T) {
	const limit = MinTruncateToolResultBytes
	enabled, _ := TruncateToolResultBytes(limit)
	disabled, _ := TruncateToolResultBytes(0)
	for _, tc := range []struct {
		processor ToolResultProcessor
		result    ToolResult
	}{
		{enabled, ToolResult{Text: strings.Repeat("x", limit)}},
		{enabled, ToolResult{Text: ""}},
		{enabled, ToolResult{Err: errors.New(strings.Repeat("x", limit))}},
		{disabled, ToolResult{Text: strings.Repeat("x", 10*limit)}},
		{disabled, ToolResult{Err: errors.New(strings.Repeat("x", 10*limit))}},
	} {
		got, err := tc.processor.Process(t.Context(), ToolPolicyInput{}, tc.result)
		if err != nil || got != tc.result {
			t.Fatalf("result changed: %#v -> %#v (%v)", tc.result, got, err)
		}
	}
	got, _ := enabled.Process(t.Context(), ToolPolicyInput{}, ToolResult{Text: strings.Repeat("x", limit+1)})
	if len(got.Text) > limit || !strings.Contains(got.Text, fmt.Sprintf("of %d bytes]", limit+1)) {
		t.Fatalf("one byte over the limit: %q", got.Text)
	}
}

func TestTruncateToolResultBytesRejectsOversizedErrors(t *testing.T) {
	processor, _ := TruncateToolResultBytes(MinTruncateToolResultBytes)
	long := strings.Repeat("x", 1000)
	for _, tc := range []struct {
		kind error
		want string
	}{
		{nil, "tool output too large"},
		{ErrToolDenied, ErrToolDenied.Error()},
		{ErrToolApprovalRequired, ErrToolApprovalRequired.Error()},
	} {
		original := ToolResult{Err: errors.New(long)}
		if tc.kind != nil {
			original = ToolResult{Err: safeToolError{text: long, kind: tc.kind}}
		}
		result, err := processor.Process(t.Context(), ToolPolicyInput{}, original)
		if err != nil || result.String() != tc.want || !errors.Is(result.Err, ErrToolOutputLimit) {
			t.Fatalf("kind=%v result=%#v err=%v", tc.kind, result, err)
		}
		if tc.kind != nil && !errors.Is(result.Err, tc.kind) {
			t.Fatalf("refusal %v lost", tc.kind)
		}
	}
}

func TestTruncateToolResultBytesInChainPublishesCutResult(t *testing.T) {
	recorder := obstest.Install(t)
	payload := "secret-marker " + strings.Repeat("y", 1000)
	tool := schedulerTool(t, "test", ToolOptions{}, func(context.Context, ai.ToolCall) (string, error) {
		return payload, nil
	})
	redactor, _ := RedactToolResult(func(_ context.Context, text string) (string, error) {
		return strings.ReplaceAll(text, "secret-marker", "safe"), nil
	})
	truncate, _ := TruncateToolResultBytes(MinTruncateToolResultBytes)
	processor, _ := ChainToolResultProcessors(redactor, truncate)
	l := &Loop{Tools: []Tool{tool}, ToolResultProcessor: processor}
	iteration, calls := schedulerCalls("test")
	events := make(chan Event, 8)
	ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{ToolOutput: gai.CaptureEnabled})
	if err := l.executeToolCalls(ctx, iteration, calls, l.Tools, events, 1, 1, 0); err != nil {
		t.Fatal(err)
	}
	close(events)
	redacted := strings.ReplaceAll(payload, "secret-marker", "safe")
	want := truncateToolText(redacted, MinTruncateToolResultBytes)
	if !strings.HasPrefix(want, "safe yyy") || !strings.HasSuffix(want, fmt.Sprintf("of %d bytes]", len(redacted))) {
		t.Fatalf("unexpected cut: %q", want)
	}
	if got := iteration.Conversation[0].Parts[0].ToolResult.Text(); got != want {
		t.Fatalf("conversation=%q want %q", got, want)
	}
	var published bool
	for event := range events {
		if event.ToolResult != nil {
			published = true
			if event.ToolResult.String() != want {
				t.Fatalf("event result=%q", event.ToolResult.String())
			}
		}
	}
	if !published {
		t.Fatal("no tool result event")
	}
	execution := iteration.Parts[0].ToolExecution
	if execution.State != ToolSucceeded || execution.Output != ToolOutputChanged {
		t.Fatalf("execution=%#v", execution)
	}
	span := requireToolSpans(t, recorder, 1)[0]
	var captured string
	for _, attr := range span.Attributes() {
		if attr.Key == "tool.output" {
			captured = attr.Value.AsString()
		}
	}
	if captured != want {
		t.Fatalf("span output=%q want %q", captured, want)
	}
	if strings.Contains(toolSpanText(span), "secret-marker") {
		t.Fatal("original output escaped telemetry")
	}
}
