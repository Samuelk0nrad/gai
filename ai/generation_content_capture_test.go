package ai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/Samuelk0nrad/gai"
	"go.opentelemetry.io/otel/attribute"
)

func contentCaptureRequest() AIRequest {
	return AIRequest{Messages: []Message{
		TextMessage(RoleSystem, "system secret"),
		TextMessage(RoleUser, "user secret"),
		{Role: RoleAssistant, Parts: []ContentPart{
			{Kind: ContentReasoning, Text: "reasoning secret"},
			{Kind: ContentText, Text: "assistant secret"},
			{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "call-1", Type: "function", Name: "lookup", Args: json.RawMessage(`{"q":"args secret"}`)}},
		}},
		{Role: RoleTool, Parts: []ContentPart{{Kind: ContentToolResult, ToolResult: &ToolResult{
			ToolCallID: "call-1", Name: "lookup", Parts: TextParts("result secret"),
		}}}},
	}}
}

func redactSecret(_ context.Context, _ gai.ContentKind, value []byte) ([]byte, error) {
	return []byte(strings.ReplaceAll(string(value), "secret", "[redacted]")), nil
}

func finishedGenerationAttributes(t *testing.T, ctx context.Context, req AIRequest, streaming bool, observe func(*GenerationObservation), result GenerationResult) map[string]attribute.Value {
	t.Helper()
	recorder, restore := installGenerationSpanRecorder(t)
	defer restore()
	_, observation := StartGenerationObservation(ctx, req, GenerationConfig{Provider: "test", Model: "model", Streaming: streaming})
	if observe != nil {
		observe(observation)
	}
	observation.Finish(result)
	return spanAttributes(generationSpan(t, recorder.Ended()).Attributes())
}

// TestGenerationObservationCapturesInputAndOutput checks that, with capture
// enabled, the ended generation span carries the redacted request messages
// and response as Langfuse observation input and output.
func TestGenerationObservationCapturesInputAndOutput(t *testing.T) {
	ctx := gai.WithContentCapturePolicy(context.Background(), gai.ContentCapturePolicy{
		Prompt: gai.CaptureEnabled, Completion: gai.CaptureEnabled, Redact: redactSecret,
	})
	output := Message{Role: RoleAssistant, Parts: []ContentPart{
		{Kind: ContentReasoning, Text: "hidden secret"},
		{Kind: ContentText, Text: "answer secret"},
		{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "call-2", Name: "lookup", Args: json.RawMessage(`{"q":"args secret"}`)}},
	}}
	attrs := finishedGenerationAttributes(t, ctx, contentCaptureRequest(), false, nil, GenerationResult{Output: &output})

	wantInput := `[{"role":"system","content":"system [redacted]"},` +
		`{"role":"user","content":"user [redacted]"},` +
		`{"role":"assistant","content":"assistant [redacted]","tool_calls":[{"id":"call-1","name":"lookup"}]},` +
		`{"role":"tool","tool_call_id":"call-1","name":"lookup"}]`
	assertStringAttribute(t, attrs, "langfuse.observation.input", wantInput)
	assertStringAttribute(t, attrs, "langfuse.observation.output",
		`{"role":"assistant","content":"answer [redacted]","tool_calls":[{"id":"call-2","name":"lookup"}]}`)
	if !attrs["gai.gen_ai.input.redaction_applied"].AsBool() || !attrs["gai.gen_ai.output.redaction_applied"].AsBool() {
		t.Fatalf("redaction metadata missing: %#v", attrs)
	}
}

// TestGenerationObservationCapturesStreamedOutput checks that a streaming
// generation exports the text and tool calls it observed.
func TestGenerationObservationCapturesStreamedOutput(t *testing.T) {
	ctx := gai.WithContentCapturePolicy(context.Background(), gai.ContentCapturePolicy{Completion: gai.CaptureEnabled})
	attrs := finishedGenerationAttributes(t, ctx, contentCaptureRequest(), true, func(o *GenerationObservation) {
		o.ObserveToken(Token{Part: &ContentPart{Kind: ContentReasoning, Text: "thinking"}})
		o.ObserveToken(Token{Part: &ContentPart{Kind: ContentText, Text: "Hello, "}})
		o.ObserveToken(Token{Part: &ContentPart{Kind: ContentText, Text: "world"}})
		o.ObserveToken(Token{Part: &ContentPart{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "call-3", Name: "search", Args: json.RawMessage(`{}`)}}})
	}, GenerationResult{})

	assertStringAttribute(t, attrs, "langfuse.observation.output",
		`{"role":"assistant","content":"Hello, world","tool_calls":[{"id":"call-3","name":"search"}]}`)
	if _, ok := attrs["langfuse.observation.input"]; ok {
		t.Fatalf("disabled prompt capture exported the input: %#v", attrs)
	}
}

// TestGenerationObservationContentRespectsPolicy checks the kind toggles and
// the byte limit on generation spans.
func TestGenerationObservationContentRespectsPolicy(t *testing.T) {
	output := TextMessage(RoleAssistant, "a reply that is longer than the limit")

	t.Run("prompt only", func(t *testing.T) {
		ctx := gai.WithContentCapturePolicy(context.Background(), gai.ContentCapturePolicy{Prompt: gai.CaptureEnabled})
		attrs := finishedGenerationAttributes(t, ctx, contentCaptureRequest(), false, nil, GenerationResult{Output: &output})
		if _, ok := attrs["langfuse.observation.input"]; !ok {
			t.Fatal("prompt capture did not export the input")
		}
		if _, ok := attrs["langfuse.observation.output"]; ok {
			t.Fatalf("disabled completion capture exported the output: %#v", attrs)
		}
	})

	t.Run("max bytes", func(t *testing.T) {
		ctx := gai.WithContentCapturePolicy(context.Background(), gai.ContentCapturePolicy{
			Prompt: gai.CaptureEnabled, Completion: gai.CaptureEnabled, MaxBytes: 16,
		})
		attrs := finishedGenerationAttributes(t, ctx, contentCaptureRequest(), false, nil, GenerationResult{Output: &output})
		for _, prefix := range []string{"input", "output"} {
			if got := attrs["langfuse.observation."+prefix].AsString(); len(got) > 16 {
				t.Fatalf("langfuse.observation.%s = %q, want at most 16 bytes", prefix, got)
			}
			if !attrs["gai.gen_ai."+prefix+".truncated"].AsBool() {
				t.Fatalf("gai.gen_ai.%s.truncated missing: %#v", prefix, attrs)
			}
		}
	})

	t.Run("redactor failure", func(t *testing.T) {
		ctx := gai.WithContentCapturePolicy(context.Background(), gai.ContentCapturePolicy{
			Prompt: gai.CaptureEnabled, Completion: gai.CaptureEnabled,
			Redact: func(context.Context, gai.ContentKind, []byte) ([]byte, error) { panic("redactor") },
		})
		attrs := finishedGenerationAttributes(t, ctx, contentCaptureRequest(), false, nil, GenerationResult{Output: &output})
		for _, key := range []string{"langfuse.observation.input", "langfuse.observation.output"} {
			if _, ok := attrs[key]; ok {
				t.Fatalf("failed redaction exported %q", key)
			}
		}
	})
}

// TestGenerationObservationOmitsContentByDefault checks that no content
// reaches a generation span without a ContentCapturePolicy.
func TestGenerationObservationOmitsContentByDefault(t *testing.T) {
	output := TextMessage(RoleAssistant, "answer secret")
	attrs := finishedGenerationAttributes(t, context.Background(), contentCaptureRequest(), true, func(o *GenerationObservation) {
		o.ObserveToken(Token{Part: &ContentPart{Kind: ContentText, Text: "streamed secret"}})
	}, GenerationResult{Output: &output})
	for key, value := range attrs {
		if strings.HasPrefix(key, "langfuse.observation.input") || strings.HasPrefix(key, "langfuse.observation.output") ||
			strings.HasPrefix(key, "gai.gen_ai.input") || strings.HasPrefix(key, "gai.gen_ai.output") {
			t.Fatalf("content attribute %q was exported without a capture policy", key)
		}
		if strings.Contains(value.String(), "secret") {
			t.Fatalf("attribute %q exported content: %q", key, value.String())
		}
	}
}
