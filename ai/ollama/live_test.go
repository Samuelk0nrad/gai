package ollama_test

import (
	"context"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/ai/ollama"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

func TestLiveToolCalling(t *testing.T) {
	modelName := strings.TrimSpace(os.Getenv("OLLAMA_SMOKE_MODEL"))
	if modelName == "" {
		t.Skip("set OLLAMA_SMOKE_MODEL to a locally installed tool-capable model")
	}
	baseURL := strings.TrimSpace(os.Getenv("OLLAMA_BASE_URL"))
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	numCtx, temperature := 512, 0.0
	model, err := ollama.New(nil, ollama.WithBaseURL(baseURL)).TypedModel(
		modelName,
		ollama.WithToolSupport(ai.FeatureSupportSupported),
		ollama.WithOptions(ollama.Options{NumCtx: &numCtx, Temperature: &temperature}),
	)
	if err != nil {
		t.Fatal(err)
	}
	var called atomic.Bool
	weather, err := loop.NewTool("get_weather", "Get the current weather for a city", ai.ToolParameters{
		Properties: []ai.ToolParameter{{Name: "city", Type: ai.ToolParameterString, Description: "City name", Required: true}},
	}, func(_ context.Context, call ai.ToolCall) (string, error) {
		var args struct {
			City string `json:"city"`
		}
		if err := loop.DecodeToolArgs(call, &args); err != nil {
			return "", err
		}
		if !strings.EqualFold(args.City, "Tokyo") {
			t.Errorf("tool city = %q", args.City)
		}
		called.Store(true)
		return `{"city":"Tokyo","celsius":21}`, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	assistant := agent.New(agent.Definition{
		Name:  "ollama-smoke",
		Model: model,
		Tools: []loop.Tool{weather},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{SystemInstructions: []gaictx.Part{
				gaictx.NewTextPart("Use get_weather for every weather question, then answer from its result."),
			}}), nil
		},
		Limits: agent.Limits{MaxLoopIterations: 3, MaxTokens: 64},
	})
	workflow, err := assistant.NewRun(ctx, agent.RunInput{Prompt: gaictx.PromptInput{User: ai.TextParts("What is the weather in Tokyo?")}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := workflow.Run(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !called.Load() {
		t.Fatal("model did not call get_weather")
	}
	if strings.TrimSpace(result.Text) == "" {
		t.Fatal("model returned no final streamed text")
	}
	if result.Usage.InputTokens == 0 || result.Usage.OutputTokens == 0 {
		t.Fatalf("missing terminal usage: %+v", result.Usage)
	}
	if err := (ai.AIRequest{Messages: result.Primary.Messages}).ValidateMessages(); err != nil {
		t.Fatalf("invalid replay transcript: %v", err)
	}
	t.Logf("model=%s output=%q usage=%+v", modelName, result.Text, result.Usage)
}
