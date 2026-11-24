package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/Samuelk0nrad/gai/agent"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/loop"
	"github.com/Samuelk0nrad/gai/modelinput"
)

// TestLoopResultToolMediaReachesResponsesRequest runs an agent whose tools
// return an image and plain text, and checks the next Responses request: the
// media result is a function_call_output array, the text result a string.
func TestLoopResultToolMediaReachesResponsesRequest(t *testing.T) {
	var mu sync.Mutex
	var bodies [][]byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, body)
		first := len(bodies) == 1
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		if first {
			for _, call := range []string{`"call_id":"call_1","name":"open_file"`, `"call_id":"call_2","name":"lookup"`} {
				_, _ = w.Write([]byte(`data: {"type":"response.output_item.done","item":{"type":"function_call",` + call + `,"arguments":"{}","status":"completed"}}` + "\n\n"))
			}
			_, _ = w.Write([]byte(`data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-test","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1}}}` + "\n\n"))
			return
		}
		_, _ = w.Write([]byte(`data: {"type":"response.output_text.delta","delta":"a receipt"}` + "\n\n"))
		_, _ = w.Write([]byte(`data: {"type":"response.completed","response":{"id":"resp_2","model":"gpt-test","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1}}}` + "\n\n"))
	}))
	defer server.Close()
	provider := New("test-key", nil, WithResponsesTransport())
	provider.baseURL = server.URL
	model, err := provider.Model("gpt-test")
	if err != nil {
		t.Fatal(err)
	}
	openFile, err := loop.NewResultTool("open_file", "Open a stored file.", ai.ToolParameters{}, func(context.Context, ai.ToolCall) (loop.ToolResult, error) {
		return loop.ToolResult{Text: "receipt.png", Media: []ai.MediaPart{{MIMEType: "image/png", Data: []byte("png"), Name: "receipt.png"}}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	lookup, err := loop.NewTool("lookup", "Look up.", ai.ToolParameters{}, func(context.Context, ai.ToolCall) (string, error) { return "plain", nil })
	if err != nil {
		t.Fatal(err)
	}
	def := agent.Definition{Name: "main", Model: model, Tools: []loop.Tool{openFile, lookup}, Prompt: func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
		return modelinput.New(modelinput.Definition{}), nil
	}}
	run, err := agent.New(def).NewRun(t.Context(), agent.RunInput{Prompt: modelinput.PromptInput{User: ai.TextParts("what is this?")}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := run.Run(t.Context())
	if err != nil || result.Text != "a receipt" {
		t.Fatalf("result = %+v, error = %v", result, err)
	}
	if len(bodies) != 2 {
		t.Fatalf("requests = %d", len(bodies))
	}
	var sent struct {
		Input []map[string]json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(bodies[1], &sent); err != nil {
		t.Fatal(err)
	}
	outputs := map[string]string{}
	for _, item := range sent.Input {
		if string(item["type"]) == `"function_call_output"` {
			var id string
			_ = json.Unmarshal(item["call_id"], &id)
			outputs[id] = string(item["output"])
		}
	}
	want := map[string]string{
		"call_1": `[{"text":"receipt.png","type":"input_text"},{"detail":"auto","image_url":"data:image/png;base64,cG5n","type":"input_image"}]`,
		"call_2": `"plain"`,
	}
	for id, output := range want {
		if outputs[id] != output {
			t.Fatalf("%s output = %s, want %s\nrequest: %s", id, outputs[id], output, bodies[1])
		}
	}
}
