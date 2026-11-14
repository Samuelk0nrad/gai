package loop_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/obstest"
	"github.com/Samuelk0nrad/gai/loop"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// runContentCaptureLoop runs one tool-call iteration and one final iteration
// and returns the attributes of the ended run span.
func runContentCaptureLoop(t *testing.T, recorder *tracetest.SpanRecorder, ctx context.Context) map[string]attribute.Value {
	t.Helper()
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{
			{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "Let me check the secret. "}},
			{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"tool secret"}`)}}},
		},
		{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "The secret answer."}}},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, &stubPromptBuilder{
		systemPrompt: "System secret",
		userPrompt:   "User secret question",
	}, nil)

	if err := loopError(collectLoopEvents(t, l, ctx)); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	var run sdktrace.ReadOnlySpan
	for _, span := range recorder.Ended() {
		if span.Name() == "loop.run" {
			if run != nil {
				t.Fatal("duplicate loop.run span")
			}
			run = span
		}
	}
	if run == nil {
		t.Fatal("loop.run span was not recorded")
	}
	return obstest.Attributes(run)
}

// TestLoopRunSpanCapturesInputAndOutput checks that the run span, the root
// observation of an agent run in Langfuse v4, carries the redacted user input
// and the accepted assistant text of every iteration.
func TestLoopRunSpanCapturesInputAndOutput(t *testing.T) {
	recorder := obstest.Install(t)
	ctx := gai.WithContentCapturePolicy(context.Background(), gai.ContentCapturePolicy{
		Prompt: gai.CaptureEnabled, Completion: gai.CaptureEnabled,
		Redact: func(_ context.Context, _ gai.ContentKind, value []byte) ([]byte, error) {
			return []byte(strings.ReplaceAll(string(value), "secret", "[redacted]")), nil
		},
	})

	attrs := runContentCaptureLoop(t, recorder, ctx)

	if got, want := attrs["langfuse.observation.input"].AsString(), "User [redacted] question"; got != want {
		t.Fatalf("langfuse.observation.input = %q, want %q", got, want)
	}
	if got, want := attrs["langfuse.observation.output"].AsString(), "Let me check the [redacted]. The [redacted] answer."; got != want {
		t.Fatalf("langfuse.observation.output = %q, want %q", got, want)
	}
	if !attrs["loop.input.redaction_applied"].AsBool() || !attrs["loop.output.redaction_applied"].AsBool() {
		t.Fatalf("redaction metadata missing: %#v", attrs)
	}
	if attrs["loop.input.original_bytes"].AsInt64() != int64(len("User secret question")) {
		t.Fatalf("loop.input.original_bytes = %d", attrs["loop.input.original_bytes"].AsInt64())
	}
}

// TestLoopRunSpanContentRespectsPolicy checks the kind toggles and the byte
// limit on the run span.
func TestLoopRunSpanContentRespectsPolicy(t *testing.T) {
	t.Run("prompt only", func(t *testing.T) {
		recorder := obstest.Install(t)
		ctx := gai.WithContentCapturePolicy(context.Background(), gai.ContentCapturePolicy{Prompt: gai.CaptureEnabled})
		attrs := runContentCaptureLoop(t, recorder, ctx)
		if _, ok := attrs["langfuse.observation.input"]; !ok {
			t.Fatal("prompt capture did not export the run input")
		}
		if _, ok := attrs["langfuse.observation.output"]; ok {
			t.Fatalf("disabled completion capture exported the run output: %#v", attrs)
		}
	})

	t.Run("completion only", func(t *testing.T) {
		recorder := obstest.Install(t)
		ctx := gai.WithContentCapturePolicy(context.Background(), gai.ContentCapturePolicy{Completion: gai.CaptureEnabled})
		attrs := runContentCaptureLoop(t, recorder, ctx)
		if _, ok := attrs["langfuse.observation.input"]; ok {
			t.Fatalf("disabled prompt capture exported the run input: %#v", attrs)
		}
		if _, ok := attrs["langfuse.observation.output"]; !ok {
			t.Fatal("completion capture did not export the run output")
		}
	})

	t.Run("max bytes", func(t *testing.T) {
		recorder := obstest.Install(t)
		ctx := gai.WithContentCapturePolicy(context.Background(), gai.ContentCapturePolicy{
			Prompt: gai.CaptureEnabled, Completion: gai.CaptureEnabled, MaxBytes: 8,
		})
		attrs := runContentCaptureLoop(t, recorder, ctx)
		for _, prefix := range []string{"input", "output"} {
			if got := attrs["langfuse.observation."+prefix].AsString(); len(got) > 8 {
				t.Fatalf("langfuse.observation.%s = %q, want at most 8 bytes", prefix, got)
			}
			if !attrs["loop."+prefix+".truncated"].AsBool() {
				t.Fatalf("loop.%s.truncated missing: %#v", prefix, attrs)
			}
		}
	})
}

// TestLoopRunSpanOmitsContentByDefault checks that no content reaches the run
// span without a ContentCapturePolicy.
func TestLoopRunSpanOmitsContentByDefault(t *testing.T) {
	recorder := obstest.Install(t)
	attrs := runContentCaptureLoop(t, recorder, context.Background())
	for key, value := range attrs {
		if strings.HasPrefix(key, "langfuse.observation.input") || strings.HasPrefix(key, "langfuse.observation.output") ||
			strings.HasPrefix(key, "loop.input") || strings.HasPrefix(key, "loop.output") {
			t.Fatalf("content attribute %q was exported without a capture policy", key)
		}
		if strings.Contains(value.String(), "secret") {
			t.Fatalf("attribute %q exported content: %q", key, value.String())
		}
	}
}
