package mistral

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/obstest"
)

// TestModelGenerateStreamCapturesDetectedOutput checks that a streamed,
// text-encoded tool call reaches the generation span the way the caller sees
// it: as a tool call with its name, never as completion text carrying its
// arguments, when completion capture is on and tool-input capture is off.
func TestModelGenerateStreamCapturesDetectedOutput(t *testing.T) {
	recorder := obstest.Install(t)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, chunk := range []string{
			`data: {"choices":[{"delta":{"content":"Calling a tool.\n\n"}}]}`,
			`data: {"choices":[{"delta":{"content":"{\"type\":\"function\",\"name\":\"echo\","}}]}`,
			`data: {"choices":[{"delta":{"content":"\"arguments\":{\"text\":\"argument secret\"}}"}}]}`,
			"data: [DONE]",
		} {
			_, _ = fmt.Fprint(w, chunk+"\n\n")
		}
	}))
	defer ts.Close()
	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatal(err)
	}
	ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{Completion: gai.CaptureEnabled})

	var toolCalls int
	for token := range m.GenerateStream(ctx, ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "call a tool")}}) {
		if token.Err != nil {
			t.Fatalf("stream error: %v", token.Err)
		}
		if token.Type() == ai.TokenTypeToolCall {
			toolCalls++
		}
	}
	if toolCalls != 1 {
		t.Fatalf("detected tool calls = %d, want 1", toolCalls)
	}

	// The span must have ended before the caller saw the stream close.
	attrs := obstest.Attributes(obstest.RequireGenerationSpans(t, recorder, 1)[0])
	output := attrs["langfuse.observation.output"].AsString()
	if strings.Contains(output, "argument secret") || strings.Contains(output, "arguments") {
		t.Fatalf("output exported tool arguments: %s", output)
	}
	if !strings.Contains(output, `"tool_calls":[{"id":"call_echo_`) || !strings.Contains(output, `"name":"echo"`) {
		t.Fatalf("output does not list the detected tool call: %s", output)
	}
	if !strings.HasPrefix(output, `{"role":"assistant","content":"Calling a tool.\n\n"`) {
		t.Fatalf("output text = %s", output)
	}
	if got := attrs["gai.gen_ai.response.tool_call_count"].AsInt64(); got != 1 {
		t.Fatalf("tool_call_count = %d, want the detected call counted", got)
	}
}
