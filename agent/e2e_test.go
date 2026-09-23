package agent_test

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/testutil/mocks"
)

type scriptedWorkflowModel struct {
	mu       sync.Mutex
	requests []ai.AIRequest
	scripts  [][]ai.Token
}

func (m *scriptedWorkflowModel) Name() string {
	return "scripted-workflow-model"
}

func (m *scriptedWorkflowModel) Generate(ctx context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
	return &ai.AIResponse{}, nil
}

func (m *scriptedWorkflowModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)
	m.mu.Lock()
	call := len(m.requests)
	m.requests = append(m.requests, req)
	var script []ai.Token
	if call < len(m.scripts) {
		script = append([]ai.Token(nil), m.scripts[call]...)
	}
	m.mu.Unlock()

	go func() {
		defer close(out)
		for _, token := range script {
			select {
			case out <- token:
			case <-ctx.Done():
				select {
				case out <- ai.Token{Type: ai.TokenTypeErr, Err: ctx.Err()}:
				default:
				}
				return
			}
		}
	}()
	return out
}

func (m *scriptedWorkflowModel) Close() error {
	return nil
}

func (m *scriptedWorkflowModel) Tokenizer() ai.Tokenizer {
	return &mocks.MockTokenizer{}
}

func (m *scriptedWorkflowModel) Requests() []ai.AIRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]ai.AIRequest(nil), m.requests...)
}

func TestAgentWorkflowEndToEndWithToolCall(t *testing.T) {
	model := &scriptedWorkflowModel{
		scripts: [][]ai.Token{
			{
				{Type: ai.TokenTypeThought, Text: "checking tool"},
				{
					Type: ai.TokenTypeToolCall,
					ToolCall: &ai.ToolCall{
						ID:   "call_1",
						Type: "function",
						Name: "echo",
						Args: []byte(`{"text":"tool says hi"}`),
					},
				},
			},
			{{Type: ai.TokenTypeText, Text: "final answer"}},
		},
	}
	assistant := agent.New(agent.Definition{
		Name:  "e2e",
		Model: model,
		Tools: []loop.Tool{loop.NewEchoTool()},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{
				Renderer: &gaictx.SimpleRenderer{},
			}), nil
		},
		Limits: agent.Limits{
			MaxLoopIterations: 3,
			MaxTokens:         64,
		},
	})

	input := textRunInput("use echo")
	input.ResponseFormat = ai.ResponseFormat{
		Type:   ai.ResponseFormatJSONSchema,
		Name:   "answer",
		Schema: []byte(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`),
	}
	expectedSchema := string(input.ResponseFormat.Schema)
	workflow, err := assistant.NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	input.ResponseFormat.Schema[0] = '['
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 0 {
		t.Fatalf("unexpected workflow errors: %v", consumed.errs)
	}
	if got := tokensText(consumed.tokens); got != "final answer" {
		t.Fatalf("unexpected final token text: %q", got)
	}
	if len(consumed.statuses) != 2 {
		t.Fatalf("expected two loop statuses, got %#v", consumed.statuses)
	}
	var toolStart, toolResult bool
	for _, event := range consumed.events {
		if event.Type == agent.EventToolStart && event.Source.Kind == agent.SourcePrimary && event.ToolCall != nil && event.ToolCall.Name == "echo" {
			toolStart = true
		}
		if event.Type == agent.EventToolResult && event.Source.Kind == agent.SourcePrimary && event.ToolResponse != nil && event.ToolResponse.TextValue() == "tool says hi" {
			toolResult = true
		}
	}
	if !toolStart || !toolResult {
		t.Fatalf("tool lifecycle missing from workflow events: start=%v result=%v events=%#v", toolStart, toolResult, consumed.events)
	}

	result := workflow.Result()
	if !result.Complete {
		t.Fatalf("workflow result was not marked complete: %+v", result)
	}
	if result.Text != "final answer" || result.Primary.Text != "final answer" {
		t.Fatalf("unexpected workflow text: %+v", result)
	}
	if result.Reasoning != "checking tool" || result.Primary.Reasoning != "checking tool" {
		t.Fatalf("unexpected reasoning capture: %+v", result)
	}
	if len(result.Primary.Iterations) != 2 {
		t.Fatalf("expected two iterations, got %+v", result.Primary.Iterations)
	}
	first := result.Primary.Iterations[0]
	var toolPart *loop.IterationPart
	for i := range first.Parts {
		if first.Parts[i].ToolReq != nil {
			toolPart = &first.Parts[i]
			break
		}
	}
	if toolPart == nil || toolPart.ToolResp == nil {
		t.Fatalf("expected first iteration to contain executed tool call, got %+v", first.Parts)
	}
	if toolPart.ToolResp.TextValue() != "tool says hi" {
		t.Fatalf("unexpected tool response: %+v", toolPart.ToolResp)
	}
	if len(result.Primary.Messages) != 4 {
		t.Fatalf("expected user, tool call, tool result, and final assistant messages, got %+v", result.Primary.Messages)
	}
	if result.Primary.Messages[1].Content.Type() != gaictx.ContentTypeToolCall ||
		result.Primary.Messages[2].Content.Type() != gaictx.ContentTypeToolResult ||
		result.Primary.Messages[3].Content.String() != "final answer" {
		t.Fatalf("unexpected reconstructed messages: %+v", result.Primary.Messages)
	}

	requests := model.Requests()
	if len(requests) != 2 {
		t.Fatalf("expected two model requests, got %d", len(requests))
	}
	if requests[0].MaxTokens != 64 || len(requests[0].Tools) != 0 {
		t.Fatalf("first request did not preserve limits or text tool transport: %+v", requests[0])
	}
	if !strings.Contains(requests[0].Prompt, `{"type":"function","name":"<tool-name>","arguments":{...}}`) {
		t.Fatalf("first request did not include the text tool protocol:\n%s", requests[0].Prompt)
	}
	for index, request := range requests {
		if request.ResponseFormat.Type != ai.ResponseFormatJSONSchema || request.ResponseFormat.Name != "answer" || string(request.ResponseFormat.Schema) != expectedSchema {
			t.Fatalf("request %d did not preserve response format: %+v", index, request.ResponseFormat)
		}
		if len(request.Tools) != 0 {
			t.Fatalf("request %d sent provider-native tools during text transport: %+v", index, request.Tools)
		}
	}
	if !strings.Contains(requests[1].Prompt, "tool res: tool says hi") {
		t.Fatalf("second prompt did not include tool result:\n%s", requests[1].Prompt)
	}
}

func TestAgentWorkflowMarksTerminalFailedAttemptDiscardable(t *testing.T) {
	fatalErr := errors.New("fatal stream error")
	model := &scriptedWorkflowModel{
		scripts: [][]ai.Token{
			{
				{Type: ai.TokenTypeText, Text: "partial"},
				{Err: fatalErr},
			},
		},
	}
	assistant := agent.New(agent.Definition{
		Name:  "terminal-failure",
		Model: model,
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{
				Renderer: &gaictx.SimpleRenderer{},
			}), nil
		},
		Limits: agent.Limits{
			MaxLoopIterations: 1,
		},
	})

	workflow, err := assistant.NewRun(context.Background(), textRunInput("fail"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 1 {
		t.Fatalf("expected one workflow error, got %d", len(consumed.errs))
	}
	if !errors.Is(consumed.errs[0], fatalErr) || errors.Is(consumed.errs[0], loop.ErrMaxRetries) {
		t.Fatalf("error = %v, want original non-retry error", consumed.errs[0])
	}
	if got := tokensText(consumed.tokens); got != "partial" {
		t.Fatalf("expected partial token to stream before failure, got %q", got)
	}
	if len(consumed.statuses) != 1 {
		t.Fatalf("expected one discard status, got %#v", consumed.statuses)
	}
	status := consumed.statuses[0]
	if !status.DiscardIteration || status.Retrying {
		t.Fatalf("expected terminal failed attempt to be discardable without retrying, got %#v", status)
	}
	if status.IterationCount != 1 || status.AttemptID != 1 || status.PartCount != 1 {
		t.Fatalf("expected failed attempt metadata, got %#v", status)
	}
	if got := status.Iteration.Parts[0].Response.Text; got != "partial" {
		t.Fatalf("expected discard status to carry partial attempt text, got %q", got)
	}
}

func TestAgentWorkflowStreamsRetriedAttemptTokens(t *testing.T) {
	model := &scriptedWorkflowModel{
		scripts: [][]ai.Token{
			{
				{Type: ai.TokenTypeText, Text: "partial"},
				{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{Usage: ai.Usage{InputTokens: 3, OutputTokens: 2}}},
				{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{Usage: ai.Usage{InputTokens: 4, OutputTokens: 3}}},
				{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("retriable stream error")}},
			},
			{
				{Type: ai.TokenTypeText, Text: "final"},
				{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{Usage: ai.Usage{InputTokens: 5, OutputTokens: 4}}},
				{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{Usage: ai.Usage{InputTokens: 6, OutputTokens: 5}}},
			},
		},
	}
	assistant := agent.New(agent.Definition{
		Name:  "retry-discard",
		Model: model,
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}}), nil
		},
		Limits:      agent.Limits{MaxLoopIterations: 1},
		RetryPolicy: &loop.RetryPolicy{MaxRetries: 1},
	})

	workflow, err := assistant.NewRun(context.Background(), textRunInput("retry"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 0 {
		t.Fatalf("unexpected workflow errors: %v", consumed.errs)
	}
	if got := tokensText(consumed.tokens); got != "partialfinal" {
		t.Fatalf("unexpected real-time stream text: %q", got)
	}
	if len(consumed.statuses) != 2 || !consumed.statuses[0].Retrying || !consumed.statuses[0].DiscardIteration {
		t.Fatalf("expected a discardable retry status, got %#v", consumed.statuses)
	}
	result := workflow.Result()
	if got := result.Primary.Text; got != "final" {
		t.Fatalf("unexpected canonical primary result text: %q", got)
	}
	if got := result.Text; got != "final" {
		t.Fatalf("unexpected canonical workflow result text: %q", got)
	}
	if got := result.Primary.AttemptedText; got != "partialfinal" {
		t.Fatalf("unexpected primary diagnostic text: %q", got)
	}
	if got := result.AttemptedText; got != "partialfinal" {
		t.Fatalf("unexpected workflow diagnostic text: %q", got)
	}
	if result.Usage != (ai.Usage{InputTokens: 6, OutputTokens: 5}) {
		t.Fatalf("accepted usage = %#v", result.Usage)
	}
	if result.BilledUsage != (ai.Usage{InputTokens: 10, OutputTokens: 8}) {
		t.Fatalf("billed usage = %#v", result.BilledUsage)
	}
}

func TestAgentWorkflowBillsRejectedRequiredToolAttemptWithoutExposingIt(t *testing.T) {
	model := &scriptedWorkflowModel{
		scripts: [][]ai.Token{
			{
				{Type: ai.TokenTypeText, Text: "rejected response"},
				{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{Usage: ai.Usage{InputTokens: 3, OutputTokens: 2}}},
			},
			{
				{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "echo", Args: []byte(`{"text":"payload"}`)}},
				{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{Usage: ai.Usage{InputTokens: 5, OutputTokens: 4}}},
			},
			{
				{Type: ai.TokenTypeText, Text: "final answer"},
				{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{Usage: ai.Usage{InputTokens: 7, OutputTokens: 6}}},
			},
		},
	}
	assistant := agent.New(agent.Definition{
		Name:  "required-tool-billing",
		Model: model,
		Tools: []loop.Tool{loop.NewEchoTool()},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}}), nil
		},
		Limits: agent.Limits{MaxLoopIterations: 3},
	})

	input := textRunInput("use echo")
	input.Execution.ToolChoice = &ai.ToolChoice{Mode: ai.ToolChoiceRequired}
	workflow, err := assistant.NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 0 {
		t.Fatalf("unexpected workflow errors: %v", consumed.errs)
	}
	if got := tokensText(consumed.tokens); got != "final answer" {
		t.Fatalf("visible tokens = %q, want only the accepted final response", got)
	}
	if len(consumed.statuses) != 3 {
		t.Fatalf("statuses = %#v, want discarded attempt plus two accepted iterations", consumed.statuses)
	}
	discarded := consumed.statuses[0]
	if !discarded.DiscardIteration || discarded.Retrying || discarded.Iteration.Usage != (ai.Usage{InputTokens: 3, OutputTokens: 2}) || len(discarded.Iteration.Parts) != 0 {
		t.Fatalf("discarded status = %#v, want usage metadata without response content", discarded)
	}
	result := workflow.Result()
	if got := result.Text; got != "final answer" {
		t.Fatalf("result text = %q, want final answer", got)
	}
	if result.Usage != (ai.Usage{InputTokens: 12, OutputTokens: 10}) {
		t.Fatalf("accepted usage = %#v", result.Usage)
	}
	if result.BilledUsage != (ai.Usage{InputTokens: 15, OutputTokens: 12}) {
		t.Fatalf("billed usage = %#v", result.BilledUsage)
	}
}

func TestAgentWorkflowRunEventsBillsRejectedRequiredToolAttempt(t *testing.T) {
	model := &scriptedWorkflowModel{
		scripts: [][]ai.Token{
			{
				{Type: ai.TokenTypeText, Text: "rejected response"},
				{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{Usage: ai.Usage{InputTokens: 3, OutputTokens: 2}}},
			},
			{
				{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "echo", Args: []byte(`{"text":"payload"}`)}},
				{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{Usage: ai.Usage{InputTokens: 5, OutputTokens: 4}}},
			},
			{
				{Type: ai.TokenTypeText, Text: "final answer"},
				{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{Usage: ai.Usage{InputTokens: 7, OutputTokens: 6}}},
			},
		},
	}
	assistant := agent.New(agent.Definition{
		Name:  "required-tool-events-billing",
		Model: model,
		Tools: []loop.Tool{loop.NewEchoTool()},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}}), nil
		},
		Limits: agent.Limits{MaxLoopIterations: 3},
	})

	input := textRunInput("use echo")
	input.Execution.ToolChoice = &ai.ToolChoice{Mode: ai.ToolChoiceRequired}
	workflow, err := assistant.NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}

	var discarded agent.Event
	for event := range workflow.RunEvents(context.Background()) {
		if event.Type == agent.EventDiscard {
			discarded = event
		}
	}
	if discarded.Iteration == nil || discarded.Iteration.Usage != (ai.Usage{InputTokens: 3, OutputTokens: 2}) || len(discarded.Iteration.Parts) != 0 {
		t.Fatalf("discarded event = %#v, want usage metadata without response content", discarded)
	}
	result := workflow.Result()
	if result.Usage != (ai.Usage{InputTokens: 12, OutputTokens: 10}) {
		t.Fatalf("accepted usage = %#v", result.Usage)
	}
	if result.BilledUsage != (ai.Usage{InputTokens: 15, OutputTokens: 12}) {
		t.Fatalf("billed usage = %#v", result.BilledUsage)
	}
}

func TestAgentWorkflowRunEventsBillsTerminalErrorAttempt(t *testing.T) {
	model := &scriptedWorkflowModel{
		scripts: [][]ai.Token{{
			{Type: ai.TokenTypeText, Text: "partial response"},
			{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{Usage: ai.Usage{InputTokens: 3, OutputTokens: 2}}},
			{Err: errors.New("terminal stream error")},
		}},
	}
	assistant := agent.New(agent.Definition{
		Name:  "terminal-error-events-billing",
		Model: model,
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}}), nil
		},
	})

	workflow, err := assistant.NewRun(context.Background(), textRunInput("fail"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}

	var terminal agent.Event
	for event := range workflow.RunEvents(context.Background()) {
		if event.Type == agent.EventStageFinish && event.StageOutcome == agent.StageFailed {
			terminal = event
		}
	}
	if terminal.Iteration == nil || terminal.Iteration.Usage != (ai.Usage{InputTokens: 3, OutputTokens: 2}) {
		t.Fatalf("terminal error event = %#v, want usage metadata", terminal)
	}
	result := workflow.Result()
	if result.BilledUsage != (ai.Usage{InputTokens: 3, OutputTokens: 2}) {
		t.Fatalf("billed usage = %#v", result.BilledUsage)
	}
}

func TestAgentWorkflowReportsCancellationWithoutError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	assistant := agent.New(agent.Definition{
		Name:  "canceled",
		Model: &scriptedWorkflowModel{},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}}), nil
		},
	})

	workflow, err := assistant.NewRun(context.Background(), textRunInput("stop"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	var gotStatuses []consumedStatus
	for event := range workflow.RunEvents(ctx) {
		if event.Type == agent.EventCanceled {
			gotStatuses = append(gotStatuses, consumedStatus{Canceled: true, DiscardIteration: true, CancellationErr: event.Err})
		}
	}
	if len(gotStatuses) != 1 {
		t.Fatalf("expected one cancellation status, got %#v", gotStatuses)
	}
	status := gotStatuses[0]
	if !status.Canceled || !errors.Is(status.CancellationErr, context.Canceled) || !status.DiscardIteration {
		t.Fatalf("unexpected cancellation status: %#v", status)
	}
	result := workflow.Result()
	if !result.Complete || !result.Canceled || !result.Primary.Canceled || !errors.Is(result.CancellationErr, context.Canceled) {
		t.Fatalf("unexpected canceled workflow result: %#v", result)
	}
}

func TestAgentWorkflowEndToEndWithAppendMiddleware(t *testing.T) {
	main := agent.New(agent.Definition{
		Name:  "main",
		Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Text: "answer"}}}},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}}), nil
		},
		Middleware: []agent.Middleware{
			agent.NewAgentMiddleware(agent.New(agent.Definition{
				Name:  "audit",
				Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Text: " audited"}}}},
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
					return gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}}), nil
				},
			}), agent.AgentMiddlewareConfig{
				Name:   "audit",
				Output: agent.AppendOutput,
			}),
		},
	})

	workflow, err := main.NewRun(context.Background(), textRunInput("question"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 0 {
		t.Fatalf("unexpected workflow errors: %v", consumed.errs)
	}
	if got := tokensText(consumed.tokens); got != "answer audited" {
		t.Fatalf("unexpected transformed output: %q", got)
	}

	result := workflow.Result()
	if !result.Complete || result.Primary.Text != "answer" || result.Text != "answer audited" {
		t.Fatalf("unexpected workflow result: %+v", result)
	}
	if len(result.Stages) != 1 || result.Stages[0].Name != "audit" || result.Stages[0].Result.Text != " audited" {
		t.Fatalf("unexpected middleware stages: %+v", result.Stages)
	}
}

func TestAgentWorkflowRunEventsPreservesRetryOrdering(t *testing.T) {
	model := &scriptedWorkflowModel{
		scripts: [][]ai.Token{
			{
				{Type: ai.TokenTypeText, Text: "partial"},
				{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("retriable stream error")}},
			},
			{{Type: ai.TokenTypeText, Text: "final"}},
		},
	}
	assistant := agent.New(agent.Definition{
		Name:  "retry-events",
		Model: model,
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}}), nil
		},
		Limits:      agent.Limits{MaxLoopIterations: 1},
		RetryPolicy: &loop.RetryPolicy{MaxRetries: 1},
	})

	workflow, err := assistant.NewRun(context.Background(), textRunInput("retry"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}

	var events []agent.Event
	for event := range workflow.RunEvents(context.Background()) {
		events = append(events, event)
	}

	if got, want := eventTypes(events), []agent.EventType{
		agent.EventStageStart,
		agent.EventAttemptStart,
		agent.EventOutput,
		agent.EventRetry,
		agent.EventAttemptStart,
		agent.EventOutput,
		agent.EventIterationDone,
		agent.EventStageFinish,
		agent.EventDone,
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected event order: got %v want %v", got, want)
	}
	if events[2].Output == nil || events[2].Output.Text != "partial" || events[2].AttemptID != 1 {
		t.Fatalf("unexpected first output event: %#v", events[2])
	}
	if events[3].AttemptID != 1 || events[3].RetryCount != 1 || events[3].Iteration == nil {
		t.Fatalf("retry event did not preserve attempt metadata: %#v", events[3])
	}
	if events[5].Output == nil || events[5].Output.Text != "final" || events[5].AttemptID != 2 {
		t.Fatalf("unexpected final output event: %#v", events[5])
	}
	result := workflow.Result()
	if !result.Complete || result.Text != "final" || result.Primary.Text != "final" {
		t.Fatalf("RunEvents did not finalize workflow result: %+v", result)
	}
}

func eventTypes(events []agent.Event) []agent.EventType {
	types := make([]agent.EventType, len(events))
	for i, event := range events {
		types[i] = event.Type
	}
	return types
}
