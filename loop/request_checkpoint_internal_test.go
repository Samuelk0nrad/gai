package loop

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"testing"

	"github.com/lace-ai/gai/ai"
)

type checkpointTestModel struct{ name string }

func (m *checkpointTestModel) Name() string { return m.name }
func (*checkpointTestModel) GenerateStream(context.Context, ai.AIRequest) <-chan ai.Token {
	ch := make(chan ai.Token)
	close(ch)
	return ch
}

type checkpointSliceModel []string

func (checkpointSliceModel) GenerateStream(context.Context, ai.AIRequest) <-chan ai.Token {
	ch := make(chan ai.Token)
	close(ch)
	return ch
}

type checkpointValueModel struct{ identity any }

func (checkpointValueModel) Name() string { return "value-model" }
func (checkpointValueModel) GenerateStream(context.Context, ai.AIRequest) <-chan ai.Token {
	ch := make(chan ai.Token)
	close(ch)
	return ch
}

type checkpointTransportModel struct{ transport http.RoundTripper }

func (checkpointTransportModel) Name() string { return "transport-value-model" }
func (checkpointTransportModel) GenerateStream(context.Context, ai.AIRequest) <-chan ai.Token {
	ch := make(chan ai.Token)
	close(ch)
	return ch
}

type checkpointRoundTripFunc func(*http.Request) (*http.Response, error)

func (f checkpointRoundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

type checkpointTestCounter struct {
	id     string
	inputs []string
}

func (c *checkpointTestCounter) ID() string                    { return c.id }
func (*checkpointTestCounter) Fidelity() ai.TokenCountFidelity { return ai.TokenCountExact }
func (c *checkpointTestCounter) CountTokens(_ context.Context, stringValue string) (int, error) {
	c.inputs = append(c.inputs, stringValue)
	return 1, nil
}

func checkpointTestRequest() ai.AIRequest {
	return ai.AIRequest{
		Messages: []ai.Message{
			ai.TextMessage(ai.RoleSystem, "system instructions"),
			ai.TextMessage(ai.RoleUser, "selected context"),
			{Role: ai.RoleUser, Parts: []ai.ContentPart{
				{Kind: ai.ContentText, Text: "current user", Extensions: []ai.Extension{{Namespace: "test", Type: "opaque", Data: json.RawMessage(`"signed"`)}}},
				{Kind: ai.ContentJSON, JSON: json.RawMessage(`{"key":1}`)},
			}},
		},
		MaxTokens:      9,
		Tools:          []ai.ToolDefinition{{Name: "echo", Description: "echo text", Parameters: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice:     ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"echo"}},
		ResponseFormat: ai.ResponseFormat{Type: ai.ResponseFormatJSONSchema, Name: "answer", Schema: json.RawMessage(`{"type":"object"}`)},
		Reasoning:      ai.ReasoningConfig{Effort: ai.ReasoningEffortLow},
	}
}

func TestRequestCheckpointRequiresIdenticalBaseAndConversationPrefix(t *testing.T) {
	mutations := []struct {
		name   string
		mutate func(*ai.AIRequest)
	}{
		{"system", func(r *ai.AIRequest) { r.Messages[0].Parts[0].Text = "other system" }},
		{"context", func(r *ai.AIRequest) { r.Messages[1].Parts[0].Text = "refreshed context" }},
		{"user", func(r *ai.AIRequest) { r.Messages[2].Parts[0].Text = "new user input" }},
		{"message role", func(r *ai.AIRequest) { r.Messages[1].Role = ai.RoleAssistant }},
		{"part order", func(r *ai.AIRequest) {
			r.Messages[2].Parts[0], r.Messages[2].Parts[1] = r.Messages[2].Parts[1], r.Messages[2].Parts[0]
		}},
		{"opaque part bytes", func(r *ai.AIRequest) { r.Messages[2].Parts[0].Extensions[0].Data[1] = 'X' }},
		{"structured JSON", func(r *ai.AIRequest) { r.Messages[2].Parts[1].JSON[7] = '2' }},
		{"prefix deletion", func(r *ai.AIRequest) { r.Messages = r.Messages[1:] }},
		{"prefix reorder", func(r *ai.AIRequest) { r.Messages[0], r.Messages[1] = r.Messages[1], r.Messages[0] }},
		{"tool name", func(r *ai.AIRequest) { r.Tools[0].Name = "other" }},
		{"tool description", func(r *ai.AIRequest) { r.Tools[0].Description = "new description" }},
		{"tool schema", func(r *ai.AIRequest) { r.Tools[0].Parameters = json.RawMessage(`{"type":"string"}`) }},
		{"tool choice", func(r *ai.AIRequest) { r.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceAuto} }},
		{"named choice", func(r *ai.AIRequest) { r.ToolChoice.Names[0] = "other" }},
		{"max output", func(r *ai.AIRequest) { r.MaxTokens++ }},
		{"response format", func(r *ai.AIRequest) { r.ResponseFormat.Type = ai.ResponseFormatJSONObject }},
		{"response name", func(r *ai.AIRequest) { r.ResponseFormat.Name = "other" }},
		{"response schema", func(r *ai.AIRequest) { r.ResponseFormat.Schema = json.RawMessage(`{"type":"array"}`) }},
		{"reasoning effort", func(r *ai.AIRequest) { r.Reasoning.Effort = ai.ReasoningEffortHigh }},
		{"reasoning budget", func(r *ai.AIRequest) { r.Reasoning.BudgetTokens = 1000 }},
	}
	for _, mutation := range mutations {
		t.Run(mutation.name, func(t *testing.T) {
			original := checkpointTestRequest()
			model := &checkpointTestModel{name: "test-model"}
			counter := &checkpointTestCounter{id: "test/counter-v1"}
			checkpoint := &requestCheckpoint{request: original.Copy(), model: model, modelName: model.Name(), counterID: ai.RequestEstimateID + ":" + counter.ID(), input: 40}
			current := original.Copy()
			current.Messages = append(current.Messages, ai.TextMessage(ai.RoleAssistant, "new output"))
			if !checkpoint.matches(current, model, counter) {
				t.Fatal("unchanged prefix with additions cannot reuse checkpoint")
			}
			mutation.mutate(&current)
			if checkpoint.matches(current, model, counter) {
				t.Fatal("changed request base reused checkpoint")
			}
			if !reflect.DeepEqual(checkpoint.request, original) {
				t.Fatal("mutating current request altered stored identity")
			}
		})
	}
}

func TestRequestCheckpointRequiresIdenticalModelAndCounterIdentity(t *testing.T) {
	request := checkpointTestRequest()
	model := &checkpointTestModel{name: "provider/model-v1"}
	counter := &checkpointTestCounter{id: "test/counter-v1"}
	checkpoint := &requestCheckpoint{request: request.Copy(), model: model, modelName: model.Name(), counterID: ai.RequestEstimateID + ":" + counter.ID(), input: 40}
	if !checkpoint.matches(request, model, counter) {
		t.Fatal("exact request must reuse checkpoint")
	}
	if checkpoint.matches(request, &checkpointTestModel{name: model.Name()}, counter) {
		t.Fatal("different model instance reused checkpoint")
	}
	model.name = "provider/model-v2"
	if checkpoint.matches(request, model, counter) {
		t.Fatal("changed model name reused checkpoint")
	}
	model.name = checkpoint.modelName
	counter.id = "test/counter-v2"
	if checkpoint.matches(request, model, counter) {
		t.Fatal("changed counter version reused checkpoint")
	}
	if checkpoint.matches(request, model, nil) {
		t.Fatal("missing counter reused checkpoint")
	}
	var absent *requestCheckpoint
	if absent.matches(request, model, counter) {
		t.Fatal("absent checkpoint matched")
	}
	if sameModel(checkpointSliceModel{"one"}, checkpointSliceModel{"one"}) {
		t.Fatal("non-comparable model identity matched")
	}
}

func TestRequestCheckpointCountsOnlyAddedMessagesAndRecountsInvalidatedBase(t *testing.T) {
	request := checkpointTestRequest()
	model := &checkpointTestModel{name: "model"}
	counter := &checkpointTestCounter{id: "test/counter-v1"}
	checkpoint := &requestCheckpoint{request: request.Copy(), model: model, modelName: model.Name(), counterID: ai.RequestEstimateID + ":" + counter.ID(), input: 40}
	current := request.Copy()
	additions := []ai.Message{ai.TextMessage(ai.RoleAssistant, "new assistant output"), ai.TextMessage(ai.RoleUser, "new tool-like context")}
	current.Messages = append(current.Messages, additions...)
	run := &runExecution{owner: &Loop{Model: model}, ctx: t.Context(), counter: counter, checkpoint: checkpoint, budget: ai.RequestBudgetConfig{Limit: 100, OutputReserve: 4, SafetyMargin: 3}}
	result, err := run.checkRequestBudget(t.Context(), current)
	if err != nil {
		t.Fatal(err)
	}
	if result.Method != "usage_checkpoint" || result.InputTokens != 50 || result.MessageTokens != 10 || result.CheckpointTokens != 40 || result.TotalTokens != 62 || result.FramingTokens != 0 || result.ToolTokens != 0 || result.OptionTokens != 0 || result.Fidelity != ai.TokenCountEstimated {
		t.Fatalf("checkpoint result=%+v", result)
	}
	if len(counter.inputs) != 2 {
		t.Fatalf("counter calls=%d, want only two additions", len(counter.inputs))
	}
	for i, message := range additions {
		if counter.inputs[i] != message.Text() {
			t.Fatalf("counted old base instead of addition %d: %s", i, counter.inputs[i])
		}
	}
	counter.inputs = nil
	current.Messages[1].Parts[0].Text = "refreshed source content"
	result, err = run.checkRequestBudget(t.Context(), current)
	if err != nil {
		t.Fatal(err)
	}
	if result.Method != "local_estimate" || result.CheckpointTokens != 0 || result.InputTokens != 26 || len(counter.inputs) != 7 {
		t.Fatalf("invalidated base was not fully recounted: result=%+v calls=%d", result, len(counter.inputs))
	}
}

func TestRequestCheckpointBudgetRejectsBoundaryAndOverflow(t *testing.T) {
	request := checkpointTestRequest()
	model := &checkpointTestModel{name: "model"}
	counter := &checkpointTestCounter{id: "test/counter-v1"}
	checkpoint := &requestCheckpoint{request: request.Copy(), model: model, modelName: model.Name(), counterID: ai.RequestEstimateID + ":" + counter.ID(), input: 40}
	request.Messages = append(request.Messages, ai.TextMessage(ai.RoleAssistant, "addition"))
	run := &runExecution{owner: &Loop{Model: model}, ctx: t.Context(), counter: counter, checkpoint: checkpoint, budget: ai.RequestBudgetConfig{Limit: 53}}
	result, err := run.checkRequestBudget(t.Context(), request)
	if !errors.Is(err, ai.ErrRequestBudgetExceeded) || result.TotalTokens != 54 {
		t.Fatalf("result=%+v error=%v", result, err)
	}
	run.budget.Limit = 54
	if _, err = run.checkRequestBudget(t.Context(), request); err != nil {
		t.Fatalf("exact boundary rejected: %v", err)
	}
	checkpoint.input = int(^uint(0) >> 1)
	if _, err = run.checkRequestBudget(t.Context(), request); !errors.Is(err, ai.ErrRequestCountFailed) {
		t.Fatalf("overflow error=%v", err)
	}
}

func TestRequestCheckpointUncomparableInterfaceFieldsUseFullEstimate(t *testing.T) {
	for _, tc := range []struct {
		name  string
		model ai.Model
	}{
		{"slice", checkpointValueModel{identity: []string{"value"}}},
		{"map", checkpointValueModel{identity: map[string]int{"value": 1}}},
		{"function", checkpointValueModel{identity: func() {}}},
		{"nested interface array", checkpointValueModel{identity: [1]any{[]string{"value"}}}},
		{"round trip function adapter", checkpointTransportModel{transport: checkpointRoundTripFunc(func(*http.Request) (*http.Response, error) { return nil, nil })}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !reflect.TypeOf(tc.model).Comparable() {
				t.Fatal("fixture must have a statically comparable model type")
			}
			request := ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "question")}}
			counter := &checkpointTestCounter{id: "test/counter-v1"}
			checkpoint := &requestCheckpoint{request: request.Copy(), model: tc.model, modelName: ai.ModelName(tc.model), counterID: ai.RequestEstimateID + ":" + counter.ID(), input: 40}
			if checkpoint.matches(request, tc.model, counter) {
				t.Fatal("uncomparable model value reused checkpoint")
			}
			run := &runExecution{owner: &Loop{Model: tc.model}, ctx: t.Context(), counter: counter, checkpoint: checkpoint, budget: ai.RequestBudgetConfig{Limit: 100}}
			result, err := run.checkRequestBudget(t.Context(), request)
			if err != nil {
				t.Fatal(err)
			}
			if result.Method != "local_estimate" || result.CheckpointTokens != 0 || result.InputTokens != 8 || len(counter.inputs) != 1 {
				t.Fatalf("unsafe equality did not fall back to full estimation: %+v", result)
			}
		})
	}
}

func TestRequestCheckpointComparableValueIdentityStillMatches(t *testing.T) {
	for _, identity := range []any{"stable", 17, [1]any{"stable"}, (*http.Transport)(nil)} {
		model := checkpointValueModel{identity: identity}
		request := ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "question")}}
		counter := &checkpointTestCounter{id: "test/counter-v1"}
		checkpoint := &requestCheckpoint{request: request.Copy(), model: model, modelName: model.Name(), counterID: ai.RequestEstimateID + ":" + counter.ID(), input: 40}
		equivalent := checkpointValueModel{identity: identity}
		if !checkpoint.matches(request, equivalent, counter) {
			t.Fatalf("safe comparable value lost identity: %#v", identity)
		}
		run := &runExecution{owner: &Loop{Model: equivalent}, ctx: t.Context(), counter: counter, checkpoint: checkpoint, budget: ai.RequestBudgetConfig{Limit: 100}}
		result, err := run.checkRequestBudget(t.Context(), request)
		if err != nil {
			t.Fatal(err)
		}
		if result.Method != "usage_checkpoint" || result.InputTokens != 40 || len(counter.inputs) != 0 {
			t.Fatalf("equal comparable model was fully recounted: %+v", result)
		}
	}
}
