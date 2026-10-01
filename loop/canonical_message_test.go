package loop_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

func TestLoopCanonicalJSONPartCannotExecuteStaleLegacyToolCall(t *testing.T) {
	tool := &countingTool{}
	model := &scriptedStreamModel{sequences: [][]ai.Token{{{
		Type:     ai.TokenTypeToolCall,
		Part:     &ai.ContentPart{Kind: ai.ContentJSON, JSON: json.RawMessage(`{"ok":true}`)},
		ToolCall: &ai.ToolCall{ID: "stale", Type: "function", Name: tool.Name(), Args: json.RawMessage(`{"text":"must not run"}`)},
		Text:     "stale text", Data: []byte("stale data"),
	}}}}
	l := loop.New(model, []loop.Tool{tool}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	events := collectLoopEvents(t, l, t.Context())
	if err := loopError(events); err != nil {
		t.Fatal(err)
	}
	if tool.calls.Load() != 0 {
		t.Fatal("canonical JSON was executed as a stale legacy tool call")
	}
	for _, event := range events {
		if event.Type == loop.EventToolStart || event.Type == loop.EventToolResult {
			t.Fatalf("unexpected tool event: %#v", event)
		}
		if event.Token != nil && (event.Token.ToolCall != nil || event.Token.Text != "" || len(event.Token.Data) != 0 || event.Token.Type != ai.TokenTypePart) {
			t.Fatalf("stale compatibility fields escaped: %#v", event.Token)
		}
	}
	messages := l.Messages()
	if len(messages) != 2 || len(messages[1].Parts) != 1 || messages[1].Parts[0].Kind != ai.ContentJSON || string(messages[1].Parts[0].JSON) != `{"ok":true}` {
		t.Fatalf("canonical messages = %#v", messages)
	}
}

type canonicalParallelTool struct {
	release    <-chan struct{}
	secondDone chan struct{}
	calls      atomic.Int32
}

func (*canonicalParallelTool) Name() string              { return "echo" }
func (*canonicalParallelTool) Description() string       { return "Returns identified test results." }
func (*canonicalParallelTool) Params() ai.ToolParameters { return loop.NewEchoTool().Params() }
func (tool *canonicalParallelTool) Function(ctx context.Context, call *ai.ToolCall) *loop.ToolResponse {
	tool.calls.Add(1)
	select {
	case <-tool.release:
	case <-ctx.Done():
		return loop.NewToolError(ctx.Err())
	}
	var args struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(call.Args, &args); err != nil {
		return loop.NewToolError(err)
	}
	if call.ID == "second" {
		close(tool.secondDone)
		return loop.NewToolError(errors.New(args.Text + " unavailable"))
	}
	select {
	case <-tool.secondDone:
	case <-ctx.Done():
		return loop.NewToolError(ctx.Err())
	}
	return loop.NewToolSuccess(args.Text)
}

func canonicalLoopCall(id, text string) ai.ContentPart {
	return ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{
		ID: id, Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"` + text + `"}`),
		ThoughtSignature: []byte("signature-" + id),
		Extensions:       []ai.Extension{{Namespace: "future", Type: "state", Data: json.RawMessage(`"opaque"`)}},
	}}
}

func mutateCanonicalCall(call *ai.ToolCall) {
	if call == nil {
		return
	}
	call.ID = "mutated"
	call.Name = "mutated"
	if len(call.Args) > 9 {
		call.Args[9] = 'X'
	}
	if len(call.ThoughtSignature) > 0 {
		call.ThoughtSignature[0] = 'X'
	}
	for i := range call.Extensions {
		if len(call.Extensions[i].Data) > 1 {
			call.Extensions[i].Data[1] = 'X'
		}
	}
}

func mutateCanonicalParts(parts []ai.ContentPart) {
	for i := range parts {
		p := &parts[i]
		p.Text = "mutated"
		for j := range p.Extensions {
			if len(p.Extensions[j].Data) > 1 {
				p.Extensions[j].Data[1] = 'X'
			}
		}
		mutateCanonicalCall(p.ToolCall)
		if p.ToolResult != nil {
			p.ToolResult.Name = "mutated"
			mutateCanonicalParts(p.ToolResult.Parts)
		}
	}
}

func TestLoopCanonicalMixedPartsSurviveRetryParallelToolsAndEventMutation(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	defer unblock()
	tool := &canonicalParallelTool{release: release, secondDone: make(chan struct{})}
	first, second, rejected := canonicalLoopCall("first", "first"), canonicalLoopCall("second", "second"), canonicalLoopCall("rejected", "rejected")
	reasoning := ai.ContentPart{Kind: ai.ContentReasoning, Text: "consider ", Extensions: []ai.Extension{{Namespace: "anthropic", Type: "signature", Data: json.RawMessage(`"reasoning-state"`), Required: true}}}
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Type: ai.TokenTypePart, Part: &rejected}, {Type: ai.TokenTypeErr, Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient}}},
		{
			{Type: ai.TokenTypePart, Part: &reasoning},
			{Type: ai.TokenTypeText, Text: "before"},
			{Type: ai.TokenTypePart, Part: &first},
			{Type: ai.TokenTypeText, Text: "between"},
			{Type: ai.TokenTypePart, Part: &second},
			{Type: ai.TokenTypeText, Text: "after"},
		},
		{{Type: ai.TokenTypeText, Text: "done"}},
	}}
	wantAssistant := ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{
		reasoning, {Kind: ai.ContentText, Text: "before"}, first, {Kind: ai.ContentText, Text: "between"}, second, {Kind: ai.ContentText, Text: "after"},
	}}
	wantDelta := []ai.Message{
		wantAssistant.Clone(),
		{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "first", Name: "echo", Parts: ai.TextParts("first")}}}},
		{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "second", Name: "echo", IsError: true, Parts: ai.TextParts("second unavailable")}}}},
	}
	builder := gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}, PromptInput: gaictx.PromptInput{User: ai.TextParts("question")}})
	l := loop.New(model, []loop.Tool{tool}, builder, nil)
	l.MaxLoopIterations = 2
	l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}
	var starts, retries, accepted int
	for event := range l.Run(ctx) {
		if event.Type == loop.EventError || event.Type == loop.EventCanceled {
			t.Fatalf("unexpected terminal event: %v", event.Err)
		}
		if event.Type == loop.EventToolStart {
			starts++
			mutateCanonicalCall(event.ToolCall)
			if starts == 2 {
				unblock()
			}
		}
		if event.Type == loop.EventRetry {
			retries++
		}
		if event.Type == loop.EventIterationDone {
			accepted++
		}
		if event.Token != nil {
			mutateCanonicalCall(event.Token.ToolCall)
			if event.Token.Part != nil {
				event.Token.Part.Text = "mutated"
				mutateCanonicalParts([]ai.ContentPart{*event.Token.Part})
			}
		}
		if event.ToolResponse != nil && event.ToolResponse.Text != nil {
			*event.ToolResponse.Text = "mutated result"
		}
		if event.Iteration != nil {
			for i := range event.Iteration.Conversation {
				mutateCanonicalParts(event.Iteration.Conversation[i].Parts)
			}
			for i := range event.Iteration.Parts {
				part := &event.Iteration.Parts[i]
				mutateCanonicalCall(part.ToolReq)
				if part.ToolResp != nil && part.ToolResp.Text != nil {
					*part.ToolResp.Text = "mutated stored result"
				}
			}
		}
	}
	if starts != 2 || tool.calls.Load() != 2 || retries != 1 || accepted != 2 {
		t.Fatalf("starts=%d executions=%d retries=%d accepted=%d", starts, tool.calls.Load(), retries, accepted)
	}
	requests := model.Requests()
	if len(requests) != 3 {
		t.Fatalf("requests=%d, want 3", len(requests))
	}
	if !reflect.DeepEqual(requests[0].Messages, requests[1].Messages) {
		t.Fatal("rejected attempt leaked into retry request")
	}
	userCount := 0
	for _, message := range requests[2].Messages {
		if message.Role == ai.RoleUser {
			userCount++
			if message.Text() != "question" {
				t.Fatalf("unexpected user input: %q", message.Text())
			}
		}
	}
	if userCount != 1 {
		t.Fatalf("next request contains user input %d times, want once", userCount)
	}
	if err := requests[2].ValidateMessages(); err != nil {
		t.Fatalf("next request became invalid: %v", err)
	}
	if !reflect.DeepEqual(requests[2].Messages[1:], wantDelta) {
		got, _ := json.Marshal(requests[2].Messages[1:])
		want, _ := json.Marshal(wantDelta)
		t.Fatalf("next request lost ordered canonical data:\ngot %s\nwant %s", got, want)
	}
	if len(l.Iterations) != 2 || !reflect.DeepEqual(l.Iterations[0].DeltaMessages(), wantDelta) {
		t.Fatal("event mutation changed accepted stored messages")
	}
	messages := l.Messages()
	userCount = 0
	for _, message := range messages {
		if message.Role == ai.RoleUser {
			userCount++
			if message.Text() != "question" {
				t.Fatalf("stored input = %q", message.Text())
			}
		}
	}
	if userCount != 1 {
		t.Fatalf("persistence view contains user input %d times, want once", userCount)
	}
	for i := range messages {
		mutateCanonicalParts(messages[i].Parts)
	}
	if !reflect.DeepEqual(l.Iterations[0].DeltaMessages(), wantDelta) {
		t.Fatal("Messages projection aliases accepted history")
	}
}

func TestLoopCanonicalEmptyUserSliceDoesNotCreateInvalidStoredInput(t *testing.T) {
	builder := gaictx.New(gaictx.Definition{
		SystemInstructions: []gaictx.Part{gaictx.NewTextPart("context only")},
		PromptInput:        gaictx.PromptInput{User: []ai.ContentPart{}},
	})
	model := &scriptedStreamModel{sequences: [][]ai.Token{{{Type: ai.TokenTypeText, Text: "answer"}}}}
	l := loop.New(model, nil, builder, nil)
	events := collectLoopEvents(t, l, t.Context())
	if err := loopError(events); err != nil {
		t.Fatal(err)
	}
	messages := l.Messages()
	if len(messages) != 1 || messages[0].Role != ai.RoleAssistant {
		t.Fatalf("conversation = %#v, want only assistant output", messages)
	}
	if _, err := json.Marshal(gaictx.StoredMessage{Message: messages[0]}); err != nil {
		t.Fatalf("accepted output cannot be stored: %v", err)
	}
}
