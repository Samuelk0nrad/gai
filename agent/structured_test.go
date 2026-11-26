package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"

	"github.com/Samuelk0nrad/gai/agent"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput"
	"github.com/Samuelk0nrad/gai/structured"
)

func TestRunStructuredRetriesBeforeMiddlewareAndReturnsTypedResult(t *testing.T) {
	schemaJSON := json.RawMessage(`{"type":"object","properties":{"count":{"type":"integer","minimum":1}},"required":["count"],"additionalProperties":false}`)
	schema, err := structured.Raw("count_answer", schemaJSON)
	if err != nil {
		t.Fatal(err)
	}
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{
		{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"count":0}`}}},
		{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: " {\n  \"count\": 2\n} "}}},
	}}
	var mu sync.Mutex
	var visible string
	middleware := agent.MiddlewareFunc(func(ctx context.Context, _ *agent.MiddlewareContext, upstream <-chan agent.Event) <-chan agent.Event {
		out := make(chan agent.Event)
		go func() {
			defer close(out)
			for event := range upstream {
				if event.Type == agent.EventOutput && event.Output != nil && event.Output.Kind == agent.OutputText {
					mu.Lock()
					visible += event.Output.Text
					mu.Unlock()
				}
				select {
				case out <- event:
				case <-ctx.Done():
					return
				}
			}
		}()
		return out
	})
	assistant := agent.New(agent.Definition{
		Name:       "structured",
		Model:      model,
		Middleware: []agent.Middleware{middleware},
		Prompt: func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
			return modelinput.New(modelinput.Definition{Renderer: &modelinput.SimpleRenderer{}}), nil
		},
	})
	input := textRunInput("count")

	result, err := agent.RunStructured(t.Context(), assistant, input, structured.Spec[struct {
		Count int `json:"count"`
	}]{Schema: schema, Retry: structured.ValidationRetryPolicy{MaxRetries: 1}})
	if err != nil {
		t.Fatalf("RunStructured: %v", err)
	}
	if result.Value.Count != 2 || string(result.JSON) != " {\n  \"count\": 2\n} " {
		t.Fatalf("result = %#v", result)
	}
	mu.Lock()
	gotVisible := visible
	mu.Unlock()
	if gotVisible != string(result.JSON) {
		t.Fatalf("middleware visible output = %q, want accepted JSON %q", gotVisible, result.JSON)
	}
	requests := model.Requests()
	if len(requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(requests))
	}
	for i, request := range requests {
		if request.ResponseFormat.Type != ai.ResponseFormatJSONSchema || request.ResponseFormat.Name != schema.Name || string(request.ResponseFormat.Schema) != string(schemaJSON) {
			t.Fatalf("request %d response format = %#v", i, request.ResponseFormat)
		}
	}
	if len(requests[1].Messages) != len(requests[0].Messages)+1 {
		t.Fatalf("retry messages = %d, want %d", len(requests[1].Messages), len(requests[0].Messages)+1)
	}
	if input.Execution != nil {
		t.Fatalf("caller input was mutated: %#v", input.Execution)
	}
}

func TestRunStructuredReturnsPathAwareErrorWithoutRetryByDefault(t *testing.T) {
	schema, err := structured.Raw("answer", json.RawMessage(`{"type":"object","properties":{"items":{"type":"array","items":{"type":"integer","minimum":1}}},"required":["items"]}`))
	if err != nil {
		t.Fatal(err)
	}
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"items":[0]}`}}}}}
	assistant := agent.New(agent.Definition{
		Model: model,
		Prompt: func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
			return modelinput.New(modelinput.Definition{Renderer: &modelinput.SimpleRenderer{}}), nil
		},
	})

	_, err = agent.RunStructured(t.Context(), assistant, textRunInput("answer"), structured.Spec[struct {
		Items []int `json:"items"`
	}]{Schema: schema})
	var validation *structured.ValidationError
	if !errors.As(err, &validation) {
		t.Fatalf("error = %T %v, want ValidationError", err, err)
	}
	if validation.Path != "/items/0" || string(validation.JSON) != `{"items":[0]}` {
		t.Fatalf("validation error = %#v", validation)
	}
	if len(model.Requests()) != 1 {
		t.Fatalf("requests = %d, want 1", len(model.Requests()))
	}
}
