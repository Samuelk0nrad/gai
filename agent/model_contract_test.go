package agent_test

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/ai/anthropic"
	"github.com/lace-ai/gai/ai/gemini"
	"github.com/lace-ai/gai/ai/mistral"
	"github.com/lace-ai/gai/ai/openai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/history"
)

// Deliberately offers only the single runtime method, with no embedded model.
type streamOnlyModel struct{ request ai.AIRequest }

func (m *streamOnlyModel) GenerateStream(_ context.Context, req ai.AIRequest) <-chan ai.Token {
	m.request = req
	out := make(chan ai.Token, 1)
	out <- ai.Token{Type: ai.TokenTypeText, Text: "answer"}
	close(out)
	return out
}

var _ ai.Model = (*streamOnlyModel)(nil)

func TestStreamOnlyModelRunsWithGenericLocalCounter(t *testing.T) {
	model := &streamOnlyModel{}
	var builder *gaictx.Builder
	a := agent.New(agent.Definition{Model: model, Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
		builder = gaictx.New(gaictx.Definition{TokenBudget: 100, SystemInstructions: []gaictx.Part{gaictx.NewTextPart("instructions")}})
		return builder, nil
	}})
	workflow, err := a.NewRun(t.Context(), agent.RunInput{Prompt: gaictx.PromptInput{User: ai.TextParts("question")}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := workflow.Run(t.Context())
	if err != nil || result.Text != "answer" {
		t.Fatalf("run = %+v, %v", result, err)
	}
	if !strings.Contains(model.request.Prompt, "question") || ai.ModelName(model) != "" {
		t.Fatalf("request/name = %+v / %q", model.request, ai.ModelName(model))
	}
	if _, ok := builder.TokenCounter().(ai.TextTokenEstimator); !ok {
		t.Fatalf("automatic counter = %T", builder.TokenCounter())
	}
}

type countOnlyCounter struct {
	err   error
	calls int
}

func (*countOnlyCounter) ID() string                      { return "test/count-only-v1" }
func (*countOnlyCounter) Fidelity() ai.TokenCountFidelity { return ai.TokenCountEstimated }
func (c *countOnlyCounter) CountTokens(context.Context, string) (int, error) {
	c.calls++
	return 3, c.err
}

func TestCountOnlyOverrideAndCountingFailureReachCaller(t *testing.T) {
	sentinel := errors.New("count failed")
	for _, failure := range []error{nil, sentinel} {
		counter := &countOnlyCounter{err: failure}
		a := agent.New(agent.Definition{Model: &streamOnlyModel{}, Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{TokenBudget: 100, SystemInstructions: []gaictx.Part{gaictx.NewTextPart("instructions")}}), nil
		}})
		workflow, err := a.NewRun(t.Context(), agent.RunInput{Execution: &agent.ExecutionOverrides{TokenCounter: agent.Optional[ai.TokenCounter]{Set: true, Value: counter}}})
		if err != nil {
			t.Fatal(err)
		}
		_, err = workflow.Run(t.Context())
		if !errors.Is(err, failure) || counter.calls == 0 {
			t.Fatalf("run error=%v, want %v, counter calls=%d", err, failure, counter.calls)
		}
	}
}

type countingTransport struct{ calls atomic.Int32 }

func (r *countingTransport) RoundTrip(*http.Request) (*http.Response, error) {
	r.calls.Add(1)
	return nil, errors.New("unexpected counting network request")
}

func TestAutomaticProviderCountersBuildPromptsWithoutNetwork(t *testing.T) {
	// Includes first use: a cold SDK tokenizer asset cache must not trigger downloads.
	transport := &countingTransport{}
	previous := http.DefaultTransport
	http.DefaultTransport = transport
	t.Cleanup(func() { http.DefaultTransport = previous })
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	for _, tc := range []struct {
		name     string
		provider ai.Provider
		model    string
		fidelity ai.TokenCountFidelity
	}{
		{"openai", openai.New("test", nil), openai.GPT41, ai.TokenCountEstimated},
		{"openai unknown", openai.New("test", nil), "future-model", ai.TokenCountEstimated},
		{"anthropic", anthropic.New("test", nil), anthropic.ClaudeSonnet4_6, ai.TokenCountEstimated},
		{"gemini", gemini.New("test", nil), "gemini-3-flash-preview", ai.TokenCountEstimated},
		{"mistral", mistral.New("test", nil), mistral.MistralSmallLatest, ai.TokenCountEstimated},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model, err := tc.provider.Model(tc.model)
			if err != nil {
				t.Fatal(err)
			}
			var builder *gaictx.Builder
			a := agent.New(agent.Definition{Model: model, Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
				builder = gaictx.New(gaictx.Definition{
					TokenBudget:        100,
					SystemInstructions: []gaictx.Part{gaictx.NewTextPart("instructions")},
					ContextSources:     []gaictx.ContextSource{history.NewHistory("test", localCountHistoryStore{})},
				})
				return builder, nil
			}})
			_, err = a.NewRun(t.Context(), agent.RunInput{Prompt: gaictx.PromptInput{Context: []gaictx.Part{gaictx.NewTextPart("context")}}})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := builder.BuildContext(t.Context()); err != nil {
				t.Fatal(err)
			}
			if prompt, err := builder.BuildPrompt(t.Context(), nil); err != nil || !strings.Contains(prompt, "earlier question") {
				t.Fatalf("history prompt = %q, %v", prompt, err)
			}
			counter := builder.TokenCounter()
			if counter == nil || counter.Fidelity() != tc.fidelity {
				t.Fatalf("counter=%T, want fidelity %v", counter, tc.fidelity)
			}
			if calls := transport.calls.Load(); calls != 0 {
				t.Fatalf("automatic counting made %d HTTP requests", calls)
			}
		})
	}
}

type legacyTokenizerModel struct{ streamOnlyModel }

func (*legacyTokenizerModel) Tokenizer() ai.Tokenizer {
	panic("legacy tokenizer must be explicitly requested")
}

func TestAutomaticCounterDoesNotConsultLegacyTokenizer(t *testing.T) {
	a := agent.New(agent.Definition{Model: &legacyTokenizerModel{}, Prompt: executionPrompt})
	workflow, err := a.NewRun(t.Context(), agent.RunInput{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := workflow.Run(t.Context()); err != nil {
		t.Fatal(err)
	}
}

type localCountHistoryStore struct{}

func (localCountHistoryStore) GetLastHistoryState(context.Context, string) (*history.HistoryState, error) {
	return &history.HistoryState{Turns: []gaictx.Turn{{ID: "turn", Count: 1, UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "earlier question")}}}}, nil
}
func (localCountHistoryStore) SaveHistoryState(context.Context, string, *history.HistoryState) error {
	return nil
}
func (localCountHistoryStore) UpdateTurnTokens(context.Context, string, string, int) error {
	return nil
}
