package ollama_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/ai/ollama"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

func TestAgentLoopReplaysGeneratedToolCallIdentity(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Messages []struct {
				Role       string `json:"role"`
				ToolCallID string `json:"tool_call_id"`
				ToolName   string `json:"tool_name"`
				ToolCalls  []struct {
					ID string `json:"id"`
				} `json:"tool_calls"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		switch requests.Add(1) {
		case 1:
			fmt.Fprintln(w, `{"model":"fixture","message":{"role":"assistant","content":"I will "},"done":false}`)
			fmt.Fprintln(w, `{"model":"fixture","message":{"role":"assistant","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Tokyo"}}}]},"done":false}`)
			fmt.Fprintln(w, `{"model":"fixture","message":{"role":"assistant","content":"check."},"done":false}`)
			fmt.Fprintln(w, `{"model":"fixture","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":3}`)
		case 2:
			if len(payload.Messages) < 3 {
				t.Errorf("second request messages = %#v", payload.Messages)
				return
			}
			assistantMessage := payload.Messages[len(payload.Messages)-2]
			resultMessage := payload.Messages[len(payload.Messages)-1]
			if len(assistantMessage.ToolCalls) != 1 || assistantMessage.ToolCalls[0].ID == "" {
				t.Errorf("assistant replay = %#v", assistantMessage)
			}
			if resultMessage.ToolCallID != assistantMessage.ToolCalls[0].ID || resultMessage.ToolName != "get_weather" {
				t.Errorf("tool result replay = %#v, assistant = %#v", resultMessage, assistantMessage)
			}
			fmt.Fprintln(w, `{"model":"fixture","message":{"role":"assistant","content":"Tokyo is 21 C."},"done":true,"done_reason":"stop","prompt_eval_count":20,"eval_count":5}`)
		default:
			t.Errorf("unexpected request")
		}
	}))
	defer server.Close()

	model, err := ollama.New(nil, ollama.WithBaseURL(server.URL)).TypedModel("fixture", ollama.WithToolSupport(ai.FeatureSupportSupported))
	if err != nil {
		t.Fatal(err)
	}
	weather, err := loop.NewTool("get_weather", "Get weather", ai.ToolParameters{Properties: []ai.ToolParameter{{Name: "city", Type: ai.ToolParameterString, Required: true}}}, func(context.Context, ai.ToolCall) (string, error) {
		return `{"celsius":21}`, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	assistant := agent.New(agent.Definition{
		Model: model,
		Tools: []loop.Tool{weather},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{}), nil
		},
	})
	workflow, err := assistant.NewRun(context.Background(), agent.RunInput{Prompt: gaictx.PromptInput{User: ai.TextParts("Weather in Tokyo?")}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := workflow.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 2 || !strings.Contains(result.Text, "21") {
		t.Fatalf("requests=%d text=%q", requests.Load(), result.Text)
	}
}
