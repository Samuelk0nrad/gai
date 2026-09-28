package summary_test

import (
	"context"
	"strings"
	"testing"

	"github.com/lace-ai/gai/agent/summary"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestSummarizerRunsSummaryAgentThroughLoop(t *testing.T) {
	t.Parallel()

	model := &recordingModel{response: "short summary"}
	summarizer := summary.New(model)

	got, err := summarizer.Summarize(context.Background(), summary.Request{
		ID:        "history",
		Text:      "long input",
		MaxTokens: 7,
	})
	if err != nil {
		t.Fatalf("Summarize failed: %v", err)
	}
	if got != "short summary" {
		t.Fatalf("unexpected summary: %q", got)
	}
	if model.request.MaxTokens != 7 {
		t.Fatalf("expected summary max tokens on loop request, got %d", model.request.MaxTokens)
	}
	if !strings.Contains(model.request.Prompt, "Summarize the provided context") {
		t.Fatalf("expected embedded summary system prompt: %q", model.request.Prompt)
	}
	if !strings.Contains(model.request.Prompt, "long input") {
		t.Fatalf("expected summary input in user prompt: %q", model.request.Prompt)
	}
}

func TestDefinitionAllowsSystemPromptOverride(t *testing.T) {
	t.Parallel()

	model := &recordingModel{response: "short summary"}
	summarizer := summary.Summarizer{
		Definition: summary.Definition(model, summary.WithSystemPrompt("custom summary system")),
	}

	_, err := summarizer.Summarize(context.Background(), summary.Request{Text: "input"})
	if err != nil {
		t.Fatalf("Summarize failed: %v", err)
	}
	if !strings.Contains(model.request.Prompt, "custom summary system") {
		t.Fatalf("expected custom system prompt: %q", model.request.Prompt)
	}
}

func TestDefinitionConfiguresRetryPolicy(t *testing.T) {
	t.Parallel()

	def := summary.Definition(&recordingModel{}, summary.WithRetryPolicy(loop.RetryPolicy{MaxRetries: 2}))
	if def.RetryPolicy == nil || def.RetryPolicy.MaxRetries != 2 {
		t.Fatalf("retry policy = %#v, want MaxRetries 2", def.RetryPolicy)
	}
}

func TestSummaryRequestPreservesNonpositiveLimitInheritance(t *testing.T) {
	for _, limit := range []int{0, -1, 7} {
		model := &recordingModel{response: "summary"}
		summarizer := summary.New(model)
		summarizer.Definition.Limits.MaxTokens = 19
		if _, err := summarizer.Summarize(t.Context(), summary.Request{Text: "input", MaxTokens: limit}); err != nil {
			t.Fatal(err)
		}
		want := 19
		if limit > 0 {
			want = limit
		}
		if model.request.MaxTokens != want {
			t.Fatalf("request limit %d: got %d, want %d", limit, model.request.MaxTokens, want)
		}
	}
}

type recordingModel struct {
	response string
	request  ai.AIRequest
}

func (m *recordingModel) Name() string {
	return "recording"
}

func (m *recordingModel) Generate(ctx context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
	m.request = req
	return &ai.AIResponse{Text: m.response}, nil
}

func (m *recordingModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)
	go func() {
		defer close(out)
		m.request = req
		out <- ai.Token{Type: ai.TokenTypeText, Text: m.response}
	}()
	return out
}

func (m *recordingModel) Close() error {
	return nil
}

func (m *recordingModel) Tokenizer() ai.Tokenizer {
	return &mocks.MockTokenizer{}
}
