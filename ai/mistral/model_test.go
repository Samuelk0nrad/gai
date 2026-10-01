package mistral

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/internal/obstest"
)

func TestModelGenerate(t *testing.T) {
	recorder := obstest.Install(t)
	var gotAuth string
	var gotReq chatCompletionRequest

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}

		gotAuth = r.Header.Get("Authorization")
		if err := json.NewDecoder(r.Body).Decode(&gotReq); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	res, err := any(m).(ai.ModelGenerator).Generate(context.Background(), ai.AIRequest{
		Prompt:    "sys\n\nctx\n\nhello",
		MaxTokens: 42,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if gotAuth != "Bearer test-key" {
		t.Fatalf("unexpected auth header: %q", gotAuth)
	}
	if gotReq.Model != MistralSmallLatest {
		t.Fatalf("unexpected model: %q", gotReq.Model)
	}
	if len(gotReq.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(gotReq.Messages))
	}
	if gotReq.Messages[0].Role != "user" {
		t.Fatalf("unexpected role: %q", gotReq.Messages[0].Role)
	}
	if gotReq.MaxTokens == nil || *gotReq.MaxTokens != 42 {
		t.Fatalf("expected max_tokens=42, got %+v", gotReq.MaxTokens)
	}

	if res.Text != "ok" {
		t.Fatalf("unexpected response text: %q", res.Text)
	}
	if res.FinishReason != "stop" {
		t.Fatalf("unexpected finish reason: %q", res.FinishReason)
	}
	if res.InputTokens != 11 || res.OutputTokens != 7 {
		t.Fatalf("unexpected usage mapping: %+v", res)
	}
	attrs := obstest.Attributes(obstest.RequireGenerationSpans(t, recorder, 1)[0])
	if attrs["gen_ai.provider.name"].AsString() != "mistral_ai" || attrs["ai.provider"].AsString() != "mistral" || attrs["gai.gen_ai.streaming"].AsBool() {
		t.Fatalf("generation attrs = %#v", attrs)
	}
}

func TestModelGenerateUsesContentCapturePolicy(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"answer-secret"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}`))
	}))
	defer ts.Close()

	var events []gai.Observation
	p := New("api-key-must-not-appear", gai.ObservationSinkFunc(func(_ context.Context, event gai.Observation) {
		events = append(events, event)
	}))
	p.baseURL = ts.URL
	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}
	ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{
		Prompt: gai.CaptureEnabled, Completion: gai.CaptureEnabled,
		Redact: func(_ context.Context, _ gai.ContentKind, value []byte) ([]byte, error) {
			return []byte(strings.ReplaceAll(string(value), "secret", "[redacted]")), nil
		},
	})
	if _, err := any(m).(ai.ModelGenerator).Generate(ctx, ai.AIRequest{Prompt: "question-secret"}); err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	var request, response gai.Observation
	for _, event := range events {
		switch event.Name {
		case "mistral_generate_request":
			request = event
		case "mistral_generate_response":
			response = event
		}
	}
	if request.Fields["prompt"] != "question-[redacted]" || response.Fields["response_text"] != "answer-[redacted]" {
		t.Fatalf("unexpected capture events: request=%#v response=%#v", request.Fields, response.Fields)
	}
	if _, ok := response.Fields["response"]; ok {
		t.Fatalf("policy exposed aggregate provider response: %#v", response.Fields)
	}
	if strings.Contains(fmt.Sprint(events), "api-key-must-not-appear") || strings.Contains(fmt.Sprint(events), "answer-secret") {
		t.Fatalf("raw content or credentials reached debug events: %#v", events)
	}
}

func TestNativeMessagesMapUserPayload(t *testing.T) {
	messages, err := mapNativeMessages([]ai.Message{{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "initial request"}}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 1 || messages[0].Role != "user" || messages[0].Content != "initial request" {
		t.Fatalf("payload = %#v", messages)
	}
}

func TestNativeMessagesMapToolErrorPayload(t *testing.T) {
	messages, err := mapNativeMessages([]ai.Message{{
		Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{
			ToolCallID: "call_1",
			Name:       "search", Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "upstream unavailable"}}, IsError: true,
		}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 1 || messages[0].Role != "tool" || messages[0].ToolCallID != "call_1" || messages[0].Content != `{"error":"upstream unavailable"}` {
		t.Fatalf("payload = %#v", messages)
	}
}

func TestModelGenerateMapsRequestCapabilities(t *testing.T) {
	var gotReq chatCompletionRequest

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotReq); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":2}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	_, err = any(m).(ai.ModelGenerator).Generate(context.Background(), ai.AIRequest{
		Prompt: "hello",
		Tools: []ai.ToolDefinition{
			{
				Type:        "function",
				Name:        "search",
				Description: "Searches documents.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
			},
		},
		ToolChoice: ai.ToolChoice{
			Mode:  ai.ToolChoiceRequired,
			Names: []string{"search"},
		},
		ResponseFormat: ai.ResponseFormat{
			Type:   ai.ResponseFormatJSONSchema,
			Name:   "answer",
			Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}}}`),
		},
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if len(gotReq.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(gotReq.Tools))
	}
	if gotReq.Tools[0].Type != "function" || gotReq.Tools[0].Function.Name != "search" {
		t.Fatalf("unexpected tool mapping: %#v", gotReq.Tools[0])
	}
	choice, ok := gotReq.ToolChoice.(map[string]any)
	if !ok {
		t.Fatalf("expected named tool choice map, got %#v", gotReq.ToolChoice)
	}
	function, ok := choice["function"].(map[string]any)
	if !ok || function["name"] != "search" {
		t.Fatalf("unexpected tool choice: %#v", gotReq.ToolChoice)
	}
	if gotReq.ResponseFormat == nil || gotReq.ResponseFormat.Type != "json_schema" {
		t.Fatalf("unexpected response format: %#v", gotReq.ResponseFormat)
	}
	if gotReq.ResponseFormat.JSONSchema == nil || gotReq.ResponseFormat.JSONSchema.Name != "answer" {
		t.Fatalf("unexpected response schema: %#v", gotReq.ResponseFormat.JSONSchema)
	}
}

func TestBuildChatCompletionRequestRejectsUnsupportedToolChoiceMode(t *testing.T) {
	_, err := buildChatCompletionRequest(ai.AIRequest{
		Prompt: "hello",
		Tools: []ai.ToolDefinition{
			{
				Type:        "function",
				Name:        "search",
				Description: "Searches documents.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
			},
		},
		ToolChoice: ai.ToolChoice{
			Mode: "sometimes",
		},
	}, MistralSmallLatest, false)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), `unsupported tool choice mode "sometimes"`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestModelGenerateMapsMultipleResponseToolCalls(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"choices":[{
				"message":{
					"content":"",
					"tool_calls":[
						{"id":"call_1","type":"function","function":{"name":"first_tool","arguments":"{\"value\":1}"}},
						{"id":"call_2","type":"function","function":{"name":"second_tool","arguments":"{\"value\":2}"}}
					]
				}
			}],
			"usage":{"prompt_tokens":3,"completion_tokens":4}
		}`))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	res, err := any(m).(ai.ModelGenerator).Generate(context.Background(), ai.AIRequest{Prompt: "call tools"})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if len(res.ToolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %#v", res.ToolCalls)
	}
	if res.ToolCalls[0].ID != "call_1" || res.ToolCalls[0].Name != "first_tool" || string(res.ToolCalls[0].Args) != `{"value":1}` {
		t.Fatalf("unexpected first tool call: %#v", res.ToolCalls[0])
	}
	if res.ToolCalls[1].ID != "call_2" || res.ToolCalls[1].Name != "second_tool" || string(res.ToolCalls[1].Args) != `{"value":2}` {
		t.Fatalf("unexpected second tool call: %#v", res.ToolCalls[1])
	}
}

func TestMapChatResponseToolCallsRejectsMalformedEntries(t *testing.T) {
	tests := []struct {
		name    string
		raw     json.RawMessage
		wantErr string
	}{
		{
			name:    "missing tool name",
			raw:     json.RawMessage(`[{"id":"call_1","type":"function","function":{"arguments":"{\"value\":1}"}}]`),
			wantErr: "map tool_calls[0]: missing tool name",
		},
		{
			name:    "invalid JSON arguments",
			raw:     json.RawMessage(`[{"id":"call_1","type":"function","function":{"name":"search","arguments":"{\"value\""}}]`),
			wantErr: `map tool_calls[0]: invalid JSON arguments for tool "search"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := mapChatResponseToolCalls(tt.raw)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %q", tt.wantErr, err.Error())
			}
		})
	}
}

func TestModelGenerateNoChoices(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	_, err = any(m).(ai.ModelGenerator).Generate(context.Background(), ai.AIRequest{Prompt: "hello"})
	if err != ErrNoChoices {
		t.Fatalf("expected ErrNoChoices, got %v", err)
	}
}

func TestModelTokenizerTokenizeUnsupported(t *testing.T) {
	p := New("test-key", nil)

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	tokens, err := m.(*Model).Tokenizer().Tokenize(context.Background(), "hello")
	if !errors.Is(err, ai.ErrTokenizerUnsupported) {
		t.Fatalf("expected ErrTokenizerUnsupported, got %v", err)
	}
	if tokens != nil {
		t.Fatalf("expected no tokens for unsupported tokenizer, got %#v", tokens)
	}
}

func TestModelTokenizerCountTokens(t *testing.T) {
	var gotAuth string
	var gotReq chatCompletionRequest

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}

		gotAuth = r.Header.Get("Authorization")
		if err := json.NewDecoder(r.Body).Decode(&gotReq); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":0}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	got, err := m.(*Model).Tokenizer().(*Tokenizer).CountTokens(context.Background(), "hello")
	if err != nil {
		t.Fatalf("CountTokens error: %v", err)
	}

	if gotAuth != "Bearer test-key" {
		t.Fatalf("unexpected auth header: %q", gotAuth)
	}
	if gotReq.Model != MistralSmallLatest {
		t.Fatalf("unexpected model: %q", gotReq.Model)
	}
	if len(gotReq.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(gotReq.Messages))
	}
	if gotReq.Messages[0].Content != "hello" {
		t.Fatalf("unexpected content: %q", gotReq.Messages[0].Content)
	}
	if gotReq.MaxTokens == nil || *gotReq.MaxTokens != 1 {
		t.Fatalf("expected max_tokens=1, got %+v", gotReq.MaxTokens)
	}
	if got != 9 {
		t.Fatalf("expected 9 tokens, got %d", got)
	}
}

func TestModelTokenizerID(t *testing.T) {
	p := New("test-key", nil)
	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	tokenizer := m.(*Model).Tokenizer()
	if tokenizer.ID() != "mistral."+MistralSmallLatest {
		t.Fatalf("unexpected tokenizer ID: %q", tokenizer.ID())
	}
}

func TestModelTokenizerCountTokensHTTPError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	_, err = m.(*Model).Tokenizer().(*Tokenizer).CountTokens(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "mistral token count failed (status 400)") {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(err.Error(), "bad request") {
		t.Fatalf("safe error string exposed response content: %v", err)
	}
	var httpErr *HTTPError
	if !errors.As(err, &httpErr) || string(httpErr.ResponseBody()) != "bad request\n" {
		t.Fatalf("typed error did not preserve explicit diagnostics: %#v", err)
	}
}

func TestModelGenerateStream(t *testing.T) {
	var gotReq chatCompletionRequest

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&gotReq); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer is not a flusher")
		}

		for _, chunk := range []string{
			`data: {"choices":[{"delta":{"content":"hel"}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"lo"}}]}` + "\n\n",
			"data: [DONE]\n\n",
		} {
			if _, err := fmt.Fprint(w, chunk); err != nil {
				t.Fatalf("write stream chunk: %v", err)
			}
			flusher.Flush()
		}
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	stream := m.GenerateStream(context.Background(), ai.AIRequest{
		Prompt:    "hello",
		MaxTokens: 55,
	})

	var gotText string
	for tok := range stream {
		if tok.Err != nil {
			t.Fatalf("unexpected stream error: %v", tok.Err)
		}
		if tok.Type != ai.TokenTypeText {
			t.Fatalf("unexpected token type: %s", tok.Type)
		}
		gotText += tok.String()
	}

	if !gotReq.Stream {
		t.Fatalf("expected stream=true in payload")
	}
	if gotReq.MaxTokens == nil || *gotReq.MaxTokens != 55 {
		t.Fatalf("expected max_tokens=55, got %+v", gotReq.MaxTokens)
	}
	if gotText != "hello" {
		t.Fatalf("unexpected streamed text: %q", gotText)
	}
}

func TestModelGenerateStreamEmitsTerminalCompletion(t *testing.T) {
	recorder := obstest.Install(t)
	var gotReq chatCompletionRequest
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotReq); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"cmpl_1\",\"model\":\"mistral-test\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"id\":\"cmpl_1\",\"model\":\"mistral-test\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n")
	}))
	defer ts.Close()
	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatal(err)
	}

	var completion *ai.Completion
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Prompt: "hello"}) {
		if token.Type == ai.TokenTypeCompletion {
			completion = token.Completion
		}
	}
	if gotReq.StreamOptions == nil || !gotReq.StreamOptions.IncludeUsage {
		t.Fatalf("stream options = %#v", gotReq.StreamOptions)
	}
	if completion == nil || completion.Provider != "mistral" || completion.RequestID != "cmpl_1" || completion.Model != "mistral-test" || completion.FinishReason != "stop" || completion.Usage.InputTokens != 7 || completion.Usage.OutputTokens != 3 || !json.Valid(completion.Raw) {
		t.Fatalf("completion = %#v", completion)
	}
	attrs := obstest.Attributes(obstest.RequireGenerationSpans(t, recorder, 1)[0])
	if !attrs["gai.gen_ai.streaming"].AsBool() || attrs["gen_ai.usage.input_tokens"].AsInt64() != 7 {
		t.Fatalf("generation attrs = %#v", attrs)
	}
}

func TestModelGenerateStreamEmitsCompletionForIdentityMetadata(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"cmpl_1\",\"model\":\"mistral-test\",\"choices\":[]}\n\ndata: [DONE]\n\n")
	}))
	defer ts.Close()
	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatal(err)
	}

	var completion *ai.Completion
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Prompt: "hello"}) {
		if token.Err != nil {
			t.Fatalf("unexpected stream error: %v", token.Err)
		}
		if token.Type == ai.TokenTypeCompletion {
			completion = token.Completion
		}
	}
	if completion == nil || completion.Provider != "mistral" || completion.RequestID != "cmpl_1" || completion.Model != "mistral-test" || !json.Valid(completion.Raw) {
		t.Fatalf("completion = %#v", completion)
	}
}

func TestModelGenerateStreamEmitsCompletionOnCleanEOFWithoutDone(t *testing.T) {
	recorder := obstest.Install(t)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"cmpl_eof\",\"model\":\"mistral-eof\",\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4}}\n\n")
	}))
	defer ts.Close()
	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatal(err)
	}

	var completions []*ai.Completion
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Prompt: "hello"}) {
		if token.Type == ai.TokenTypeCompletion {
			completions = append(completions, token.Completion)
		}
	}
	if len(completions) != 1 || completions[0] == nil || completions[0].RequestID != "cmpl_eof" || completions[0].Usage.InputTokens != 9 {
		t.Fatalf("completions = %#v", completions)
	}
	attrs := obstest.Attributes(obstest.RequireGenerationSpans(t, recorder, 1)[0])
	if attrs["gen_ai.response.id"].AsString() != "cmpl_eof" || attrs["gen_ai.usage.output_tokens"].AsInt64() != 4 {
		t.Fatalf("generation attrs = %#v", attrs)
	}
}

func TestModelGenerateStreamPreservesCompletionOnMalformedEvent(t *testing.T) {
	recorder := obstest.Install(t)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"cmpl_error\",\"model\":\"mistral-error\",\"choices\":[],\"usage\":{\"prompt_tokens\":6,\"completion_tokens\":2}}\n\ndata: {not-json}\n\n")
	}))
	defer ts.Close()
	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatal(err)
	}

	var completion *ai.Completion
	var streamError error
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Prompt: "hello"}) {
		switch token.Type {
		case ai.TokenTypeCompletion:
			completion = token.Completion
		case ai.TokenTypeErr:
			streamError = token.Err
		}
	}
	if streamError == nil || completion == nil || completion.RequestID != "cmpl_error" {
		t.Fatalf("stream error = %v, completion = %#v", streamError, completion)
	}
	span := obstest.RequireGenerationSpans(t, recorder, 1)[0]
	attrs := obstest.Attributes(span)
	if span.Status().Code.String() != "Error" || attrs["gen_ai.response.id"].AsString() != "cmpl_error" || attrs["gen_ai.usage.input_tokens"].AsInt64() != 6 {
		t.Fatalf("generation status = %s, attrs = %#v", span.Status().Code, attrs)
	}
}

func TestModelGenerateStreamToolCall(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer is not a flusher")
		}

		toolCallJSON := `[{"id":"call_abc","type":"function","function":{"name":"my_tool","arguments":"{\"param\":\"value\"}"}}]`
		for _, chunk := range []string{
			`data: {"choices":[{"delta":{"tool_calls":` + toolCallJSON + `}}]}` + "\n\n",
			"data: [DONE]\n\n",
		} {
			if _, err := fmt.Fprint(w, chunk); err != nil {
				t.Fatalf("write stream chunk: %v", err)
			}
			flusher.Flush()
		}
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	stream := m.GenerateStream(context.Background(), ai.AIRequest{
		Prompt: "call a tool",
	})

	var tokens []ai.Token
	for tok := range stream {
		if tok.Err != nil {
			t.Fatalf("unexpected stream error: %v", tok.Err)
		}
		tokens = append(tokens, tok)
	}

	if len(tokens) != 1 {
		t.Fatalf("expected 1 token, got %d", len(tokens))
	}
	tok := tokens[0]
	if tok.Type != ai.TokenTypeToolCall {
		t.Fatalf("expected TokenTypeToolCall, got %s", tok.Type)
	}
	if tok.ToolCall == nil {
		t.Fatal("expected ToolCall to be populated, got nil")
	}
	if tok.ToolCall.ID != "call_abc" {
		t.Fatalf("expected provider ToolCall.ID=call_abc, got %q", tok.ToolCall.ID)
	}
	if tok.ToolCall.Type != "function" {
		t.Fatalf("expected ToolCall.Type=function, got %q", tok.ToolCall.Type)
	}
	if tok.ToolCall.Name != "my_tool" {
		t.Fatalf("expected ToolCall.Name=my_tool, got %q", tok.ToolCall.Name)
	}
	wantArgs := `{"param":"value"}`
	if string(tok.ToolCall.Args) != wantArgs {
		t.Fatalf("expected ToolCall.Args=%s, got %s", wantArgs, string(tok.ToolCall.Args))
	}
}

func TestModelGenerateStreamToolCallDeltas(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer is not a flusher")
		}

		for _, chunk := range []string{
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_","type":"function","function":{"name":"my_","arguments":"{\"param\""}}]}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"abc","function":{"name":"tool","arguments":":\"value\"}"}}]}}]}` + "\n\n",
			"data: [DONE]\n\n",
		} {
			if _, err := fmt.Fprint(w, chunk); err != nil {
				t.Fatalf("write stream chunk: %v", err)
			}
			flusher.Flush()
		}
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	stream := m.GenerateStream(context.Background(), ai.AIRequest{
		Prompt: "call a tool",
	})

	var tokens []ai.Token
	for tok := range stream {
		if tok.Err != nil {
			t.Fatalf("unexpected stream error: %v", tok.Err)
		}
		tokens = append(tokens, tok)
	}

	if len(tokens) != 1 {
		t.Fatalf("expected 1 token, got %d", len(tokens))
	}
	tok := tokens[0]
	if tok.Type != ai.TokenTypeToolCall {
		t.Fatalf("expected TokenTypeToolCall, got %s", tok.Type)
	}
	if tok.ToolCall == nil {
		t.Fatal("expected ToolCall to be populated, got nil")
	}
	if tok.ToolCall.ID != "call_abc" {
		t.Fatalf("expected accumulated provider ToolCall.ID=call_abc, got %q", tok.ToolCall.ID)
	}
	if tok.ToolCall.Name != "my_tool" {
		t.Fatalf("expected ToolCall.Name=my_tool, got %q", tok.ToolCall.Name)
	}
	if string(tok.ToolCall.Args) != `{"param":"value"}` {
		t.Fatalf("expected accumulated ToolCall.Args, got %s", string(tok.ToolCall.Args))
	}
}

func TestModelGenerateStreamDetectsTextEncodedToolCall(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer is not a flusher")
		}

		for _, chunk := range []string{
			`data: {"choices":[{"delta":{"content":"\n"}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"\n"}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"{\"id\":\""}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"echo\",\""}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"type\":\"function\",\""}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"name\":\"echo\",\""}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"arguments\":{\""}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"text\":\"try"}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":" the echo tool\"}}"}}]}` + "\n\n",
			"data: [DONE]\n\n",
		} {
			if _, err := fmt.Fprint(w, chunk); err != nil {
				t.Fatalf("write stream chunk: %v", err)
			}
			flusher.Flush()
		}
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	stream := m.GenerateStream(context.Background(), ai.AIRequest{
		Prompt: "call a tool",
	})

	var (
		gotToolCall bool
		gotText     string
	)
	for tok := range stream {
		if tok.Err != nil {
			t.Fatalf("unexpected stream error: %v", tok.Err)
		}

		if tok.Type == ai.TokenTypeToolCall {
			if tok.ToolCall == nil {
				t.Fatal("expected ToolCall to be populated, got nil")
			}
			if !strings.HasPrefix(tok.ToolCall.ID, "call_echo_") {
				t.Fatalf("expected generated ToolCall.ID for echo, got %q", tok.ToolCall.ID)
			}
			if tok.ToolCall.Type != "function" {
				t.Fatalf("expected ToolCall.Type=function, got %q", tok.ToolCall.Type)
			}
			if tok.ToolCall.Name != "echo" {
				t.Fatalf("expected ToolCall.Name=echo, got %q", tok.ToolCall.Name)
			}
			if string(tok.ToolCall.Args) != `{"text":"try the echo tool"}` {
				t.Fatalf("unexpected ToolCall.Args: %s", string(tok.ToolCall.Args))
			}
			gotToolCall = true
			continue
		}

		if tok.Type == ai.TokenTypeText {
			gotText += tok.String()
		}
	}

	if !gotToolCall {
		t.Fatal("expected to detect a tool call from text stream, got none")
	}
	if gotText != "" {
		t.Fatalf("expected no text tokens after tool-call detection, got %q", gotText)
	}
}

func TestModelGenerateStreamClassifiesProviderRateLimit(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("x-request-id", "req_mistral")
		w.Header().Set("Retry-After", "2")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"code":"rate_limit_exceeded"}}`))
	}))
	defer ts.Close()
	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatal(err)
	}
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Prompt: "hello"}) {
		if token.Type != ai.TokenTypeErr {
			continue
		}
		var providerErr *ai.ProviderError
		if !errors.As(token.Err, &providerErr) || providerErr.Kind != ai.ProviderErrorRateLimited || providerErr.StatusCode != http.StatusTooManyRequests || providerErr.Code != "rate_limit_exceeded" || providerErr.RequestID != "req_mistral" || providerErr.RetryAfter != 2*time.Second || errors.Unwrap(providerErr) == nil {
			t.Fatalf("provider error = %#v", token.Err)
		}
		return
	}
	t.Fatal("expected stream error")
}
