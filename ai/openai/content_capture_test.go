package openai

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/obstest"
)

// TestModelGenerateExportsCapturedContentOnGenerationSpan checks that a
// direct Generate call, whose generation span may be a trace's root, carries
// the prompt and completion when the capture policy enables them.
func TestModelGenerateExportsCapturedContentOnGenerationSpan(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl_1","model":"gpt-4o","choices":[{"finish_reason":"stop","message":{"content":"an estimate"}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`))
	}))
	defer ts.Close()
	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model("gpt-4o")
	if err != nil {
		t.Fatal(err)
	}
	req := ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "estimate this")}}

	for _, tc := range []struct {
		name    string
		policy  *gai.ContentCapturePolicy
		capture bool
	}{
		{name: "enabled", policy: &gai.ContentCapturePolicy{Prompt: gai.CaptureEnabled, Completion: gai.CaptureEnabled}, capture: true},
		{name: "default"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := obstest.Install(t)
			ctx := t.Context()
			if tc.policy != nil {
				ctx = gai.WithContentCapturePolicy(ctx, *tc.policy)
			}
			if _, err := any(m).(ai.ModelGenerator).Generate(ctx, req); err != nil {
				t.Fatalf("Generate: %v", err)
			}
			attrs := obstest.Attributes(obstest.RequireGenerationSpans(t, recorder, 1)[0])
			want := map[string]string{
				"langfuse.observation.input":  `[{"role":"user","content":"estimate this"}]`,
				"langfuse.observation.output": `{"role":"assistant","content":"an estimate"}`,
			}
			for key, value := range want {
				got, ok := attrs[key]
				if !tc.capture {
					if ok {
						t.Fatalf("%s exported without a capture policy: %q", key, got.AsString())
					}
					continue
				}
				if got.AsString() != value {
					t.Fatalf("%s = %q, want %q", key, got.AsString(), value)
				}
			}
		})
	}
}
