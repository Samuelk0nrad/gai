package agent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/Samuelk0nrad/gai/agent"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/ai/anthropic"
	"github.com/Samuelk0nrad/gai/loop"
	"github.com/Samuelk0nrad/gai/modelinput"
)

// TestAnthropicPromptCachingAcrossLoopIterations checks that every loop
// iteration asks for automatic caching and that the second iteration reports
// the cached prefix.
func TestAnthropicPromptCachingAcrossLoopIterations(t *testing.T) {
	var mu sync.Mutex
	var cacheControls []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		mu.Lock()
		control, _ := body["cache_control"].(map[string]any)
		cacheControls = append(cacheControls, control)
		iteration := len(cacheControls)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		if iteration == 1 {
			_, _ = fmt.Fprint(w, `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"usage":{"input_tokens":20,"cache_creation_input_tokens":3000,"cache_read_input_tokens":0,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`)
			return
		}
		_, _ = fmt.Fprint(w, `event: message_start
data: {"type":"message_start","message":{"id":"msg_2","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"usage":{"input_tokens":15,"cache_creation_input_tokens":40,"cache_read_input_tokens":3000,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"answer"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`)
	}))
	t.Cleanup(server.Close)
	model, err := anthropic.New("test", nil, anthropic.WithHTTPClient(server.Client()), anthropic.WithBaseURL(server.URL), anthropic.WithPromptCaching(anthropic.PromptCacheTTL5m)).TypedModel(anthropic.ClaudeOpus5_5)
	if err != nil {
		t.Fatal(err)
	}
	tool, err := loop.NewTool("lookup", "Look something up", ai.ToolParameters{}, func(context.Context, ai.ToolCall) (string, error) { return "found", nil })
	if err != nil {
		t.Fatal(err)
	}
	a := agent.New(agent.Definition{
		Model:  model,
		Tools:  []loop.Tool{tool},
		Limits: agent.Limits{MaxTokens: 64},
		Prompt: func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
			return modelinput.New(modelinput.Definition{TokenBudget: 10000, SystemInstructions: []modelinput.Part{modelinput.NewTextPart("instructions")}}), nil
		},
	})
	workflow, err := a.NewRun(t.Context(), textRunInput("question"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := workflow.Run(t.Context())
	if err != nil || !result.Complete || result.Text != "answer" || len(result.Primary.Iterations) != 2 {
		t.Fatalf("workflow = %+v, %v", result, err)
	}
	mu.Lock()
	controls := append([]map[string]any(nil), cacheControls...)
	mu.Unlock()
	if len(controls) != 2 {
		t.Fatalf("requests = %d, want 2", len(controls))
	}
	for i, control := range controls {
		if control["type"] != "ephemeral" || control["ttl"] != "5m" {
			t.Fatalf("request %d cache_control = %#v", i+1, control)
		}
	}
	first, second := result.Primary.Iterations[0].Usage, result.Primary.Iterations[1].Usage
	if first.CacheCreationTokens != 3000 || first.CachedTokens != 0 {
		t.Fatalf("first iteration usage = %+v", first)
	}
	if second.CachedTokens != 3000 || second.InputTokens != 3055 {
		t.Fatalf("second iteration usage = %+v, want 3000 cached input tokens", second)
	}
}
