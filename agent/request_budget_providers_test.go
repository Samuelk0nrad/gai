package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/ai/anthropic"
	"github.com/lace-ai/gai/ai/gemini"
	"github.com/lace-ai/gai/ai/mistral"
	"github.com/lace-ai/gai/ai/openai"
	gaictx "github.com/lace-ai/gai/context"
)

type budgetHTTPRecorder struct {
	mu       sync.Mutex
	requests []string
}

func (r *budgetHTTPRecorder) RoundTrip(req *http.Request) (*http.Response, error) {
	r.mu.Lock()
	r.requests = append(r.requests, req.Method+" "+req.URL.String())
	r.mu.Unlock()
	return nil, errors.New("unexpected HTTP request during unsupported accurate counting")
}

func (r *budgetHTTPRecorder) snapshot() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.requests...)
}

func TestAgentRequestBudgetUnsupportedProvidersDoNotUseHTTP(t *testing.T) {
	// Catch both provider requests and tokenizer asset downloads on a cold cache.
	assets := &budgetHTTPRecorder{}
	previous := http.DefaultTransport
	http.DefaultTransport = assets
	t.Cleanup(func() { http.DefaultTransport = previous })
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	for _, tc := range []struct {
		name  string
		model func(*http.Client) (ai.Model, error)
	}{
		{"openai", func(client *http.Client) (ai.Model, error) {
			return openai.New("test", nil, openai.WithHTTPClient(client), openai.WithBaseURL("https://provider.invalid/v1")).TypedModel(openai.GPT41)
		}},
		{"gemini", func(client *http.Client) (ai.Model, error) {
			return gemini.New("test", nil, gemini.WithHTTPClient(client), gemini.WithBaseURL("https://provider.invalid")).TypedModel("gemini-3-flash-preview")
		}},
		{"mistral", func(client *http.Client) (ai.Model, error) {
			return mistral.New("test", nil, mistral.WithHTTPClient(client), mistral.WithBaseURL("https://provider.invalid/v1")).TypedModel(mistral.MistralSmallLatest)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			transport := &budgetHTTPRecorder{}
			model, err := tc.model(&http.Client{Transport: transport})
			if err != nil {
				t.Fatal(err)
			}
			a := agent.New(agent.Definition{
				Model: model,
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
					return gaictx.New(gaictx.Definition{TokenBudget: 10000, SystemInstructions: []gaictx.Part{gaictx.NewTextPart("instructions")}}), nil
				},
				RequestBudget: &ai.RequestBudgetConfig{Limit: 10000, Mode: ai.RequestCountAccurate},
			})
			input := textRunInput("question")
			input.Prompt.Context = []gaictx.Part{gaictx.NewTextPart("context details")}
			workflow, err := a.NewRun(t.Context(), input)
			if err != nil {
				t.Fatal(err)
			}
			result, err := workflow.Run(t.Context())
			var unsupported *ai.InputTokenCountUnsupportedError
			if !errors.Is(err, ai.ErrInputTokenCountUnsupported) || !errors.As(err, &unsupported) || unsupported.Model != ai.ModelName(model) {
				t.Fatalf("accurate request count error = %v, want unsupported for %q", err, ai.ModelName(model))
			}
			if !result.Complete || len(result.Primary.Iterations) != 0 || len(result.Output) != 0 || len(result.Errors) != 1 || !errors.Is(result.Errors[0], ai.ErrInputTokenCountUnsupported) {
				t.Fatalf("unsupported provider result = %+v", result)
			}
			if requests := transport.snapshot(); len(requests) != 0 {
				t.Fatalf("unsupported provider made HTTP requests: %v", requests)
			}
			if requests := assets.snapshot(); len(requests) != 0 {
				t.Fatalf("accurate counting downloaded assets or used default HTTP transport: %v", requests)
			}
		})
	}
}

func TestAgentRequestBudgetAnthropicDefaultOnlyGenerates(t *testing.T) {
	type textBlock struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	type wireRequest struct {
		Model     string      `json:"model"`
		Stream    bool        `json:"stream"`
		MaxTokens int         `json:"max_tokens"`
		System    []textBlock `json:"system"`
		Messages  []struct {
			Role    string      `json:"role"`
			Content []textBlock `json:"content"`
		} `json:"messages"`
	}
	type capturedRequest struct {
		method string
		path   string
		body   wireRequest
		err    error
	}
	var mu sync.Mutex
	var requests []capturedRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured := capturedRequest{method: r.Method, path: r.URL.Path}
		captured.err = json.NewDecoder(r.Body).Decode(&captured.body)
		mu.Lock()
		requests = append(requests, captured)
		mu.Unlock()
		if captured.err != nil || r.Method != http.MethodPost || r.URL.Path != "/v1/messages" {
			http.Error(w, "unexpected generation/count request", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"answer"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`)
	}))
	t.Cleanup(server.Close)
	model, err := anthropic.New("test", nil, anthropic.WithHTTPClient(server.Client()), anthropic.WithBaseURL(server.URL)).TypedModel(anthropic.ClaudeSonnet4_6)
	if err != nil {
		t.Fatal(err)
	}
	a := agent.New(agent.Definition{
		Model:  model,
		Limits: agent.Limits{MaxTokens: 64},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{TokenBudget: 10000, SystemInstructions: []gaictx.Part{gaictx.NewTextPart("instructions")}}), nil
		},
	})
	input := textRunInput("question")
	input.Prompt.Context = []gaictx.Part{gaictx.NewTextPart("context details")}
	workflow, err := a.NewRun(t.Context(), input)
	if err != nil {
		t.Fatal(err)
	}
	result, err := workflow.Run(t.Context())
	if err != nil || !result.Complete || result.Text != "answer" || len(result.Errors) != 0 || len(result.Primary.Iterations) != 1 {
		t.Fatalf("default Anthropic workflow = %+v, %v", result, err)
	}
	iteration := result.Primary.Iterations[0]
	budget := iteration.RequestBudget
	if budget == nil || budget.Method != "local_estimate" || budget.Fidelity != ai.TokenCountEstimated || budget.Limit != 10000 || budget.InputTokens <= 0 || budget.OutputReserve != 64 {
		t.Fatalf("default finalized request budget = %+v", budget)
	}
	if !iteration.UsageReported || iteration.Usage.InputTokens != 12 || iteration.Usage.OutputTokens != 1 {
		t.Fatalf("generation usage = %+v, reported %v", iteration.Usage, iteration.UsageReported)
	}
	mu.Lock()
	captured := append([]capturedRequest(nil), requests...)
	mu.Unlock()
	if len(captured) != 1 || captured[0].method != http.MethodPost || captured[0].path != "/v1/messages" || captured[0].err != nil {
		t.Fatalf("default budgeting must only generate once, HTTP requests = %+v", captured)
	}
	body := captured[0].body
	if body.Model != anthropic.ClaudeSonnet4_6 || !body.Stream || body.MaxTokens != 64 || len(body.System) != 1 || !strings.Contains(body.System[0].Text, "instructions") || len(body.Messages) != 2 {
		t.Fatalf("generation request = %+v", body)
	}
	for i, want := range []string{"context details", "question"} {
		message := body.Messages[i]
		if message.Role != "user" || len(message.Content) != 1 || message.Content[0].Type != "text" || !strings.Contains(message.Content[0].Text, want) {
			t.Fatalf("generation message %d = %+v, want user content %q", i, message, want)
		}
	}
}
