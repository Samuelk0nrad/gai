package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/internal/obstest"
)

func TestModelGenerateMapsCapabilitiesAndResponse(t *testing.T) {
	recorder := obstest.Install(t)
	var got map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if gotAuth := r.Header.Get("Authorization"); gotAuth != "Bearer test-key" {
			t.Fatalf("unexpected auth: %q", gotAuth)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl_1","choices":[{"message":{"content":"ok","tool_calls":[{"id":"call_1","type":"function","function":{"name":"search","arguments":"{\"query\":\"go\"}"}}]}}],"usage":{"prompt_tokens":11,"completion_tokens":7,"completion_tokens_details":{"reasoning_tokens":3}}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(O3)
	if err != nil {
		t.Fatalf("Model returned error: %v", err)
	}
	res, err := any(m).(ai.ModelGenerator).Generate(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}, MaxTokens: 42,
		Tools:          []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`)}},
		ToolChoice:     ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"search"}},
		ResponseFormat: ai.ResponseFormat{Type: ai.ResponseFormatJSONSchema, Name: "answer", Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}}}`)},
		Reasoning:      ai.ReasoningConfig{Effort: ai.ReasoningEffortHigh},
	})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if got["model"] != O3 || got["max_completion_tokens"] != float64(42) || got["reasoning_effort"] != "high" {
		t.Fatalf("unexpected request mapping: %#v", got)
	}
	if len(got["tools"].([]any)) != 1 || got["tool_choice"].(map[string]any)["function"].(map[string]any)["name"] != "search" {
		t.Fatalf("unexpected tool mapping: %#v", got)
	}
	if got["response_format"].(map[string]any)["type"] != "json_schema" {
		t.Fatalf("unexpected response format: %#v", got)
	}
	if res.Text() != "ok" || res.InputTokens != 11 || res.OutputTokens != 7 || res.ReasoningTokens != 3 {
		t.Fatalf("unexpected response: %#v", res)
	}
	if len(res.ToolCalls()) != 1 || res.ToolCalls()[0].ID != "call_1" || res.ToolCalls()[0].Name != "search" || string(res.ToolCalls()[0].Args) != `{"query":"go"}` {
		t.Fatalf("unexpected tool calls: %#v", res.ToolCalls())
	}
	attrs := obstest.Attributes(obstest.RequireGenerationSpans(t, recorder, 1)[0])
	if attrs["gen_ai.provider.name"].AsString() != "openai" || attrs["gai.gen_ai.response.tool_call_count"].AsInt64() != 1 {
		t.Fatalf("generation attrs = %#v", attrs)
	}
}

func TestModelGenerateWithResponsesTransportMapsToolContinuationAndNoneEffort(t *testing.T) {
	var got map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/responses" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","status":"completed","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done"}]},{"type":"function_call","call_id":"call_2","name":"search","arguments":"{\"q\":\"go\"}","status":"completed"}],"usage":{"input_tokens":11,"output_tokens":7,"output_tokens_details":{"reasoning_tokens":3}}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil, WithResponsesTransport())
	p.baseURL = ts.URL
	m, err := p.Model("gpt-5.6-terra")
	if err != nil {
		t.Fatal(err)
	}
	res, err := any(m).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{
		Messages: []ai.Message{
			{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{Type: "function", ID: "call_1", Name: "search", Args: json.RawMessage(`{"q":"first"}`)}}}},
			{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "call_1", Name: "search", Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "first result"}}}}}},
			{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "continue"}}},
		},
		Tools:      []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice: ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"search"}},
		Reasoning:  ai.ReasoningConfig{Effort: ai.ReasoningEffortNone},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got["model"] != "gpt-5.6-terra" || got["reasoning"].(map[string]any)["effort"] != "none" || got["tool_choice"].(map[string]any)["name"] != "search" {
		t.Fatalf("unexpected Responses request: %#v", got)
	}
	input := got["input"].([]any)
	if len(input) != 3 || input[0].(map[string]any)["type"] != "function_call" || input[1].(map[string]any)["type"] != "function_call_output" {
		t.Fatalf("native tool continuation = %#v", input)
	}
	if res.Text() != "done" || len(res.ToolCalls()) != 1 || res.ToolCalls()[0].ID != "call_2" || res.InputTokens != 11 || res.OutputTokens != 7 || res.ReasoningTokens != 3 {
		t.Fatalf("response = %#v", res)
	}
}

func TestModelResponsesTransportDisablesResponseStorage(t *testing.T) {
	recorder := obstest.Install(t)
	requests := make(chan map[string]any, 2)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/responses" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		requests <- request
		w.Header().Set("Content-Type", "application/json")
		if r.Header.Get("Accept") == "text/event-stream" {
			_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{}}\n\n"))
			return
		}
		_, _ = w.Write([]byte(`{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":0,"output_tokens":0}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil, WithResponsesTransport())
	p.baseURL = ts.URL
	m, err := p.Model("gpt-5.6-terra")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := any(m).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "sync")}}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "stream")}}) {
		if token.Type() == ai.TokenTypeErr {
			t.Fatalf("GenerateStream token error: %v", token.Err)
		}
	}
	for _, path := range []string{"synchronous", "streaming"} {
		request := <-requests
		if request["store"] != false {
			t.Fatalf("%s request store = %#v, want false", path, request["store"])
		}
		include, ok := request["include"].([]any)
		if !ok || !slices.Contains(include, any("reasoning.encrypted_content")) {
			t.Fatalf("%s request include = %#v, want reasoning.encrypted_content", path, request["include"])
		}
	}
	spans := obstest.RequireGenerationSpans(t, recorder, 2)
	streaming := map[bool]int{}
	for _, span := range spans {
		streaming[obstest.Attributes(span)["gai.gen_ai.streaming"].AsBool()]++
	}
	if streaming[false] != 1 || streaming[true] != 1 {
		t.Fatalf("Responses generation spans by streaming mode = %#v", streaming)
	}
}

func TestModelGenerateWithResponsesTransportPreservesReasoningItemsAcrossToolContinuation(t *testing.T) {
	var continuation map[string]any
	requests := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/responses" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		requests++
		w.Header().Set("Content-Type", "application/json")
		if requests == 1 {
			_, _ = w.Write([]byte(`{"id":"resp_1","status":"completed","output":[{"id":"rs_1","type":"reasoning","encrypted_content":"opaque-reasoning","summary":[],"status":"completed"},{"type":"function_call","call_id":"call_1","name":"search","arguments":"{\"q\":\"go\"}","status":"completed"}],"usage":{"input_tokens":1,"output_tokens":1}}`))
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&continuation); err != nil {
			t.Fatalf("decode continuation: %v", err)
		}
		_, _ = w.Write([]byte(`{"id":"resp_2","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil, WithResponsesTransport())
	p.baseURL = ts.URL
	m, err := p.Model("gpt-5.6-terra")
	if err != nil {
		t.Fatal(err)
	}
	first, err := any(m).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "find go")}, Tools: []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object"}`)}}})
	if err != nil {
		t.Fatalf("first Generate: %v", err)
	}
	if len(first.ToolCalls()) != 1 || len(first.Message.Parts) != 2 || first.Message.Parts[0].Kind != ai.ContentExtension {
		t.Fatalf("first tool calls = %#v, want reasoning signature", first.ToolCalls())
	}
	_, err = any(m).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{
		first.Message,
		{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: first.ToolCalls()[0].ID, Name: first.ToolCalls()[0].Name, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "result"}}}}}},
	}})
	if err != nil {
		t.Fatalf("continuation Generate: %v", err)
	}
	input := continuation["input"].([]any)
	if len(input) != 3 || input[0].(map[string]any)["type"] != "reasoning" || input[0].(map[string]any)["encrypted_content"] != "opaque-reasoning" || input[1].(map[string]any)["type"] != "function_call" || input[2].(map[string]any)["type"] != "function_call_output" {
		t.Fatalf("continuation input = %#v", input)
	}
}

func TestModelGenerateStreamWithResponsesTransportPreservesReasoningItemsAcrossToolContinuation(t *testing.T) {
	var continuation map[string]any
	requests := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/responses" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		requests++
		if requests == 1 {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"encrypted_content\":\"opaque-reasoning\",\"summary\":[],\"status\":\"completed\"}}\n\n"))
			_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\\\"go\\\"}\",\"status\":\"completed\"}}\n\n"))
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&continuation); err != nil {
			t.Fatalf("decode continuation: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_2","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil, WithResponsesTransport())
	p.baseURL = ts.URL
	m, err := p.Model("gpt-5.6-terra")
	if err != nil {
		t.Fatal(err)
	}
	var call *ai.ToolCall
	semantic := ai.Message{Role: ai.RoleAssistant}
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "find go")}, Tools: []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object"}`)}}}) {
		semantic.AppendToken(token)
		if token.Type() == ai.TokenTypeToolCall {
			call = token.ToolCall()
		}
	}
	if call == nil || len(semantic.Parts) != 2 || semantic.Parts[0].Kind != ai.ContentExtension {
		t.Fatalf("streamed tool call = %#v, want reasoning signature", call)
	}
	_, err = any(m).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{
		semantic,
		{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: call.ID, Name: call.Name, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "result"}}}}}},
	}})
	if err != nil {
		t.Fatalf("continuation Generate: %v", err)
	}
	input := continuation["input"].([]any)
	if len(input) != 3 || input[0].(map[string]any)["type"] != "reasoning" || input[0].(map[string]any)["encrypted_content"] != "opaque-reasoning" || input[1].(map[string]any)["type"] != "function_call" || input[2].(map[string]any)["type"] != "function_call_output" {
		t.Fatalf("continuation input = %#v", input)
	}
}

func TestModelGenerateWithResponsesTransportRejectsInvalidToolCallArguments(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/responses" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","status":"completed","output":[{"type":"function_call","call_id":"call_1","name":"search","arguments":"not-json","status":"completed"}],"usage":{"input_tokens":11,"output_tokens":7}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil, WithResponsesTransport())
	p.baseURL = ts.URL
	m, err := p.Model("gpt-5.6-terra")
	if err != nil {
		t.Fatal(err)
	}
	_, err = any(m).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}})
	if err == nil || err.Error() != `invalid JSON arguments for tool "search"` {
		t.Fatalf("Generate error = %v, want invalid tool arguments error", err)
	}
}

func TestModelGenerateWithResponsesTransportRejectsFailedResponseWithoutErrorMessage(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/responses" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","status":"failed","output":[],"usage":{"input_tokens":0,"output_tokens":0}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil, WithResponsesTransport())
	p.baseURL = ts.URL
	m, err := p.Model("gpt-5.6-terra")
	if err != nil {
		t.Fatal(err)
	}
	_, err = any(m).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}})
	if err == nil || err.Error() != "OpenAI Responses API: response failed" {
		t.Fatalf("Generate error = %v, want failed response error", err)
	}
}

func TestModelGenerateWithResponsesTransportReturnsRefusalText(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/responses" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"refusal","refusal":"I can't help with that."}],"status":"completed"}],"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil, WithResponsesTransport())
	p.baseURL = ts.URL
	m, err := p.Model("gpt-5.6-terra")
	if err != nil {
		t.Fatal(err)
	}
	res, err := any(m).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if res.Text() != "I can't help with that." {
		t.Fatalf("response text = %q, want refusal text", res.Text())
	}
}

func TestBuildResponsesParamsValidatesUnsupportedToolChoiceMode(t *testing.T) {
	_, err := buildResponsesParams("gpt-5.6-terra", ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}, Tools: []ai.ToolDefinition{{
		Type:        "function",
		Name:        "search",
		Description: "Search",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}},
		ToolChoice: ai.ToolChoice{Mode: ai.ToolChoiceMode("unsupported")},
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported tool choice mode") {
		t.Fatalf("buildResponsesParams error = %v, want validation error", err)
	}
}

func TestBuildResponsesParamsRejectsUndefinedRequiredTool(t *testing.T) {
	_, err := buildResponsesParams("gpt-5.6-terra", ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}, Tools: []ai.ToolDefinition{{
		Type:        "function",
		Name:        "search",
		Description: "Search",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}},
		ToolChoice: ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"missing"}},
	})
	if err == nil || err.Error() != `required tool "missing" is not defined` {
		t.Fatalf("buildResponsesParams error = %v, want undefined required-tool error", err)
	}
}

func TestModelGenerateStreamWithResponsesTransportMapsTextToolCallsAndTerminalErrors(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello \"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\\\"go\\\"}\",\"status\":\"completed\"}}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"error\",\"code\":\"server_error\",\"message\":\"provider failed\"}\n\n"))
	}))
	defer ts.Close()

	p := New("test-key", nil, WithResponsesTransport())
	p.baseURL = ts.URL
	m, err := p.Model("gpt-5.6-terra")
	if err != nil {
		t.Fatal(err)
	}
	var tokens []ai.Token
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}, Tools: []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object"}`)}}}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 3 || tokens[0].Type() != ai.TokenTypeText || tokens[0].Text() != "hello " {
		t.Fatalf("tokens = %#v", tokens)
	}
	if call := tokens[1].ToolCall(); tokens[1].Type() != ai.TokenTypeToolCall || call == nil || call.ID != "call_1" || call.Name != "search" || string(call.Args) != `{"q":"go"}` {
		t.Fatalf("tool token = %#v", tokens[1])
	}
	if tokens[2].Type() != ai.TokenTypeErr || tokens[2].Err == nil || tokens[2].Err.Error() != "OpenAI Responses API: provider failed" {
		t.Fatalf("error token = %#v", tokens[2])
	}
}

func TestModelGenerateStreamWithResponsesTransportFallsBackForEmptyFailedResponseError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.failed\",\"response\":{}}\n\n"))
	}))
	defer ts.Close()

	p := New("test-key", nil, WithResponsesTransport())
	p.baseURL = ts.URL
	m, err := p.Model("gpt-5.6-terra")
	if err != nil {
		t.Fatal(err)
	}
	var tokens []ai.Token
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 1 || tokens[0].Type() != ai.TokenTypeErr || tokens[0].Err == nil || tokens[0].Err.Error() != "OpenAI Responses API: response failed" {
		t.Fatalf("tokens = %#v", tokens)
	}
}

func TestModelGenerateStreamWithResponsesTransportClassifiesFailedResponse(t *testing.T) {
	tests := []struct {
		name string
		code string
		want ai.ProviderErrorKind
	}{
		{name: "rate limit", code: "rate_limit_exceeded", want: ai.ProviderErrorRateLimited},
		{name: "server error", code: "server_error", want: ai.ProviderErrorTransient},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/responses" {
					t.Fatalf("unexpected path %q", r.URL.Path)
				}
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = fmt.Fprintf(w, "data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_1\",\"error\":{\"code\":%q,\"message\":\"provider failed\"}}}\n\n", tt.code)
			}))
			defer ts.Close()

			p := New("test-key", nil, WithResponsesTransport())
			p.baseURL = ts.URL
			m, err := p.Model("gpt-5.6-terra")
			if err != nil {
				t.Fatal(err)
			}
			for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
				if token.Type() != ai.TokenTypeErr {
					continue
				}
				var providerErr *ai.ProviderError
				if !errors.As(token.Err, &providerErr) || providerErr.Kind != tt.want || providerErr.Code != tt.code || providerErr.RequestID != "resp_1" || errors.Unwrap(providerErr) == nil {
					t.Fatalf("provider error = %#v", token.Err)
				}
				return
			}
			t.Fatal("expected stream error")
		})
	}
}

func TestModelGenerateStreamWithResponsesTransportReturnsRefusalText(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.refusal.delta\",\"delta\":\"I can't help with that.\"}\n\n"))
	}))
	defer ts.Close()

	p := New("test-key", nil, WithResponsesTransport())
	p.baseURL = ts.URL
	m, err := p.Model("gpt-5.6-terra")
	if err != nil {
		t.Fatal(err)
	}
	var tokens []ai.Token
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 1 || tokens[0].Type() != ai.TokenTypeText || tokens[0].Text() != "I can't help with that." {
		t.Fatalf("tokens = %#v, want refusal text token", tokens)
	}
}

func TestNativeMessagesMapUserPayload(t *testing.T) {
	messages, err := mapNativeMessages([]ai.Message{{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "initial request"}}}})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(messages)
	if err != nil {
		t.Fatal(err)
	}
	var payload []map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	if len(payload) != 1 || payload[0]["role"] != "user" || payload[0]["content"] != "initial request" {
		t.Fatalf("payload = %#v", payload)
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
	raw, err := json.Marshal(messages)
	if err != nil {
		t.Fatal(err)
	}
	var payload []map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	if len(payload) != 1 || payload[0]["role"] != "tool" || payload[0]["tool_call_id"] != "call_1" || payload[0]["content"] != `{"error":"upstream unavailable"}` {
		t.Fatalf("payload = %#v", payload)
	}
}

func TestModelGenerateStreamMapsTextAndToolCalls(t *testing.T) {
	recorder := obstest.Install(t)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello \"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\"}}]}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl_no_usage\",\"model\":\"resolved-no-usage\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"go\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(GPT41Mini)
	if err != nil {
		t.Fatalf("Model returned error: %v", err)
	}
	var tokens []ai.Token
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 2 || tokens[0].Type() != ai.TokenTypeText || tokens[0].Text() != "hello " {
		t.Fatalf("unexpected stream tokens: %#v", tokens)
	}
	if call := tokens[1].ToolCall(); tokens[1].Type() != ai.TokenTypeToolCall || call == nil || call.ID != "call_1" || call.Name != "search" || string(call.Args) != `{"q":"go"}` {
		t.Fatalf("unexpected tool token: %#v", tokens[1])
	}
	attrs := obstest.Attributes(obstest.RequireGenerationSpans(t, recorder, 1)[0])
	if attrs["gen_ai.response.id"].AsString() != "chatcmpl_no_usage" || attrs["gen_ai.response.model"].AsString() != "resolved-no-usage" {
		t.Fatalf("generation attrs = %#v", attrs)
	}
	if got := attrs["gen_ai.response.finish_reasons"].AsStringSlice(); len(got) != 1 || got[0] != "tool_calls" {
		t.Fatalf("finish reasons = %v", got)
	}
	if _, ok := attrs["gen_ai.usage.input_tokens"]; ok {
		t.Fatal("stream without usage emitted usage attributes")
	}
}

func TestModelGenerateStreamUsesMetadataAfterUsageChunk(t *testing.T) {
	recorder := obstest.Install(t)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl_early\",\"model\":\"early-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl_final\",\"model\":\"resolved-final\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(GPT41Mini)
	if err != nil {
		t.Fatal(err)
	}
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		if token.Type() == ai.TokenTypeErr {
			t.Fatalf("stream error: %v", token.Err)
		}
	}
	attrs := obstest.Attributes(obstest.RequireGenerationSpans(t, recorder, 1)[0])
	if attrs["gen_ai.response.id"].AsString() != "chatcmpl_final" || attrs["gen_ai.response.model"].AsString() != "resolved-final" || attrs["gen_ai.usage.input_tokens"].AsInt64() != 5 {
		t.Fatalf("generation attrs = %#v", attrs)
	}
	if got := attrs["gen_ai.response.finish_reasons"].AsStringSlice(); len(got) != 1 || got[0] != "stop" {
		t.Fatalf("finish reasons = %v", got)
	}
}

func TestModelGenerateStreamPreservesUsageOnlyTerminalChunk(t *testing.T) {
	recorder := obstest.Install(t)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4.1-mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7,\"completion_tokens_details\":{\"reasoning_tokens\":3},\"prompt_tokens_details\":{\"cached_tokens\":2}}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(GPT41Mini)
	if err != nil {
		t.Fatal(err)
	}
	var tokens []ai.Token
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 2 || tokens[1].Type() != ai.TokenTypeCompletion {
		t.Fatalf("tokens = %#v", tokens)
	}
	completion := tokens[1].Completion
	if completion == nil || completion.Usage.InputTokens != 11 || completion.Usage.OutputTokens != 7 || completion.Usage.ReasoningTokens != 3 || completion.Usage.CachedTokens != 2 || completion.FinishReason != "stop" || completion.RequestID != "chatcmpl_1" || completion.Provider != "openai" || completion.Model != GPT41Mini {
		t.Fatalf("completion = %#v", completion)
	}
	attrs := obstest.Attributes(obstest.RequireGenerationSpans(t, recorder, 1)[0])
	if !attrs["gai.gen_ai.streaming"].AsBool() || attrs["gen_ai.usage.output_tokens"].AsInt64() != 7 {
		t.Fatalf("generation attrs = %#v", attrs)
	}
}

func TestModelGenerateStreamPreservesMetadataFromSoleUsageOnlyChunk(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl_sole\",\"model\":\"gpt-4.1-mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(GPT41Mini)
	if err != nil {
		t.Fatal(err)
	}
	var tokens []ai.Token
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 1 || tokens[0].Type() != ai.TokenTypeCompletion {
		t.Fatalf("tokens = %#v", tokens)
	}
	completion := tokens[0].Completion
	if completion == nil || completion.RequestID != "chatcmpl_sole" || completion.Provider != "openai" || completion.Model != GPT41Mini {
		t.Fatalf("completion = %#v", completion)
	}
}

func TestModelGenerateStreamSnapshotsUsageOnlyCompletion(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7}}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl_later\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(GPT41Mini)
	if err != nil {
		t.Fatal(err)
	}
	var tokens []ai.Token
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 2 || tokens[0].Type() != ai.TokenTypeCompletion {
		t.Fatalf("tokens = %#v", tokens)
	}
	completion := tokens[0].Completion
	if completion == nil || completion.Usage.InputTokens != 11 || completion.Usage.OutputTokens != 7 || completion.RequestID != "" || completion.Model != "" || completion.FinishReason != "" {
		t.Fatalf("completion must be a snapshot of the usage-only chunk, got %#v", completion)
	}
}

func TestModelGenerateStreamSnapshotsRawForEachUsageOnlyCompletion(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7}}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":13,\"completion_tokens\":9}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(GPT41Mini)
	if err != nil {
		t.Fatal(err)
	}
	var completions []*ai.Completion
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		if token.Type() == ai.TokenTypeCompletion {
			completions = append(completions, token.Completion)
		}
	}
	if len(completions) != 2 {
		t.Fatalf("completion count = %d, want 2", len(completions))
	}
	if !strings.Contains(string(completions[0].Raw), `"prompt_tokens":11`) || !strings.Contains(string(completions[1].Raw), `"prompt_tokens":13`) {
		t.Fatalf("completion raw values must remain distinct: first=%s second=%s", completions[0].Raw, completions[1].Raw)
	}
}

func TestModelGenerateStreamEmitsToolCallsByIndex(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_2\",\"type\":\"function\",\"function\":{\"name\":\"second\",\"arguments\":\"{}\"}},{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"first\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(GPT41Mini)
	if err != nil {
		t.Fatalf("Model returned error: %v", err)
	}
	var calls []*ai.ToolCall
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		if token.Type() == ai.TokenTypeToolCall {
			calls = append(calls, token.ToolCall())
		}
	}
	if len(calls) != 2 || calls[0].ID != "call_1" || calls[1].ID != "call_2" {
		t.Fatalf("unexpected tool-call order: %#v", calls)
	}
}

func TestBuildChatCompletionParamsRejectsReasoningEffortForNonReasoningModels(t *testing.T) {
	for _, model := range []string{GPT41, GPT41Mini, GPT41Nano, GPT4o, GPT4oMini} {
		err := openAIDescriptor(model).ValidateRequest(ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}, Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortHigh}})
		if !errors.Is(err, ai.ErrUnsupportedCapability) {
			t.Fatalf("expected unsupported capability error for %q, got %v", model, err)
		}
	}
	if err := openAIDescriptor(O3).ValidateRequest(ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}, Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortHigh}}); err != nil {
		t.Fatalf("expected reasoning model to accept effort: %v", err)
	}
	if _, err := buildChatCompletionParams(GPT41Mini, ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}, false); err != nil {
		t.Fatalf("expected non-reasoning model to accept empty effort: %v", err)
	}
}

func TestBuildChatCompletionParamsValidatesCanonicalRequest(t *testing.T) {
	if _, err := buildChatCompletionParams(GPT41Mini, ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}, ToolChoice: ai.ToolChoice{Mode: ai.ToolChoiceAuto}}, false); err == nil {
		t.Fatal("expected invalid tool choice to fail validation")
	}
}

func TestOpenAIMapsExtendedReasoningEfforts(t *testing.T) {
	tests := []struct {
		model  string
		effort ai.ReasoningEffort
	}{
		{"gpt-5", ai.ReasoningEffortMinimal},
		{"gpt-5.6-terra", ai.ReasoningEffortXHigh},
		{"gpt-5.6-terra", ai.ReasoningEffortMax},
	}
	for _, tt := range tests {
		t.Run(tt.model+"/"+string(tt.effort), func(t *testing.T) {
			effort := tt.effort
			model := tt.model
			req := ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}, Reasoning: ai.ReasoningConfig{Effort: effort}}
			if _, err := buildChatCompletionParams(model, req, false); err != nil {
				t.Fatalf("buildChatCompletionParams(%q): %v", effort, err)
			}
			if _, err := buildResponsesParams(model, req); err != nil {
				t.Fatalf("buildResponsesParams(%q): %v", effort, err)
			}
			if err := openAIDescriptor(model).ValidateRequest(req); err != nil {
				t.Fatalf("descriptor rejected %q: %v", effort, err)
			}
		})
	}
}

func TestModelPreflightRejectsUnsupportedBeforeTransport(t *testing.T) {
	requests := 0
	ts := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { requests++ }))
	defer ts.Close()
	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(GPT41)
	if err != nil {
		t.Fatal(err)
	}
	_, err = any(m).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "")}, Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortHigh}})
	if !errors.Is(err, ai.ErrUnsupportedCapability) || requests != 0 {
		t.Fatalf("Generate error = %v, requests = %d; want local unsupported error and no request", err, requests)
	}
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "")}, Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortHigh}}) {
		if !errors.Is(token.Err, ai.ErrUnsupportedCapability) {
			t.Fatalf("stream error = %v, want unsupported capability", token.Err)
		}
	}
	if requests != 0 {
		t.Fatalf("stream made %d requests, want none", requests)
	}
}

func TestModelPreflightRejectsReasoningToolsOnChatCompletionsBeforeTransport(t *testing.T) {
	requests := 0
	ts := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { requests++ }))
	defer ts.Close()
	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model("gpt-5.6-terra")
	if err != nil {
		t.Fatal(err)
	}
	req := ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "")}, Tools: []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object"}`)}},
		Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortHigh},
	}
	if _, err := any(m).(ai.ModelGenerator).Generate(t.Context(), req); !errors.Is(err, ai.ErrUnsupportedCapability) || requests != 0 {
		t.Fatalf("Generate error = %v, requests = %d; want local unsupported error and no request", err, requests)
	}
	for token := range m.GenerateStream(t.Context(), req) {
		if !errors.Is(token.Err, ai.ErrUnsupportedCapability) {
			t.Fatalf("stream error = %v, want unsupported capability", token.Err)
		}
	}
	if requests != 0 {
		t.Fatalf("stream made %d requests, want none", requests)
	}
}

func TestModelPreflightRejectsNoneReasoningToolsOnChatCompletionsBeforeTransport(t *testing.T) {
	requests := 0
	ts := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { requests++ }))
	defer ts.Close()
	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model("gpt-5.6-terra")
	if err != nil {
		t.Fatal(err)
	}
	req := ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "")}, Tools: []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object"}`)}},
		Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortNone},
	}
	if _, err := any(m).(ai.ModelGenerator).Generate(t.Context(), req); !errors.Is(err, ai.ErrUnsupportedCapability) || requests != 0 {
		t.Fatalf("Generate error = %v, requests = %d; want local unsupported error and no request", err, requests)
	}
	for token := range m.GenerateStream(t.Context(), req) {
		if !errors.Is(token.Err, ai.ErrUnsupportedCapability) {
			t.Fatalf("stream error = %v, want unsupported capability", token.Err)
		}
	}
	if requests != 0 {
		t.Fatalf("stream made %d requests, want none", requests)
	}
}

func TestModelAllowsUnknownDynamicReasoningEffort(t *testing.T) {
	var got map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer ts.Close()
	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model("gpt-dynamic-reasoner")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := any(m).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "")}, Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortHigh}}); err != nil {
		t.Fatalf("dynamic model rejected locally: %v", err)
	}
	if got["reasoning_effort"] != "high" {
		t.Fatalf("reasoning effort not mapped: %#v", got)
	}
}

func TestOpenAIDescriptorUsesReasoningFamilyOverlay(t *testing.T) {
	tests := []struct {
		model   string
		efforts []ai.ReasoningEffort
	}{
		{"o5-pro", []ai.ReasoningEffort{ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh}},
		{"gpt-5", []ai.ReasoningEffort{ai.ReasoningEffortMinimal, ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh}},
		{"gpt-5.1-codex", []ai.ReasoningEffort{ai.ReasoningEffortNone, ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh}},
		{"gpt-5.6-terra", []ai.ReasoningEffort{ai.ReasoningEffortNone, ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh, ai.ReasoningEffortXHigh, ai.ReasoningEffortMax}},
	}
	for _, tt := range tests {
		d := openAIDescriptor(tt.model)
		if d.ReasoningEffort != ai.FeatureSupportSupported || !slices.Equal(d.ReasoningEfforts, tt.efforts) {
			t.Fatalf("descriptor for %q = %#v, want efforts %#v", tt.model, d, tt.efforts)
		}
	}
	d := openAIDescriptor("future-chat-model")
	if d.ReasoningEffort != ai.FeatureSupportUnknown || len(d.ReasoningEfforts) != 0 {
		t.Fatalf("unrelated unknown descriptor = %#v, want unknown reasoning effort", d)
	}
	d = openAIDescriptor("gpt-5.99-preview")
	if d.ReasoningEffort != ai.FeatureSupportUnknown || len(d.ReasoningEfforts) != 0 {
		t.Fatalf("unknown GPT-5 revision descriptor = %#v, want unknown reasoning effort", d)
	}
}

func TestModelCounterUsesLocalEncodingWhenKnown(t *testing.T) {
	m := &Model{name: GPT41}
	if tokenizer := m.TokenCounter(); tokenizer == nil {
		t.Fatal("TokenCounter() = nil, want local counter for a known encoding")
	}
}

func TestModelDescriptorAdvertisesPortableCapabilities(t *testing.T) {
	d := openAIDescriptor(GPT41)
	if d.NativeMessages != ai.FeatureSupportSupported || d.NativeTools != ai.FeatureSupportSupported || d.Usage != ai.FeatureSupportSupported || d.FinishReason != ai.FeatureSupportSupported || d.StreamingUsage != ai.FeatureSupportSupported {
		t.Fatalf("capability descriptor = %#v", d)
	}
}

func TestApplyToolsRestrictsRequiredToolNames(t *testing.T) {
	definitions := []ai.ToolDefinition{
		{Type: "function", Name: "first", Description: "First", Parameters: json.RawMessage(`{"type":"object"}`)},
		{Type: "function", Name: "second", Description: "Second", Parameters: json.RawMessage(`{"type":"object"}`)},
		{Type: "function", Name: "excluded", Description: "Excluded", Parameters: json.RawMessage(`{"type":"object"}`)},
	}
	params, err := buildChatCompletionParams(GPT41Mini, ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}, Tools: definitions,
		ToolChoice: ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"second", "first"}},
	}, false)
	if err != nil {
		t.Fatalf("buildChatCompletionParams returned error: %v", err)
	}
	if len(params.Tools) != 2 || params.Tools[0].Function.Name != "first" || params.Tools[1].Function.Name != "second" {
		t.Fatalf("unexpected restricted tools: %#v", params.Tools)
	}

	if _, err := buildChatCompletionParams(GPT41Mini, ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}, Tools: definitions,
		ToolChoice: ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"missing"}},
	}, false); err == nil {
		t.Fatal("expected an undefined required tool to be rejected")
	}
}

func TestModelGenerateValidatesToolCallArguments(t *testing.T) {
	for _, tt := range []struct {
		name      string
		arguments string
		wantArgs  string
		wantErr   bool
	}{
		{name: "empty defaults to object", arguments: "", wantArgs: "{}"},
		{name: "invalid JSON is rejected", arguments: "not-json", wantErr: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[{"message":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"search","arguments":` + mustMarshalJSON(t, tt.arguments) + `}}]}}]}`))
			}))
			defer ts.Close()

			p := New("test-key", nil)
			p.baseURL = ts.URL
			m, err := p.Model(GPT41Mini)
			if err != nil {
				t.Fatalf("Model returned error: %v", err)
			}
			res, err := any(m).(ai.ModelGenerator).Generate(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}})
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected Generate to reject invalid tool-call arguments")
				}
				return
			}
			if err != nil {
				t.Fatalf("Generate returned error: %v", err)
			}
			if len(res.ToolCalls()) != 1 || string(res.ToolCalls()[0].Args) != tt.wantArgs {
				t.Fatalf("unexpected tool calls: %#v", res.ToolCalls())
			}
		})
	}
}

func mustMarshalJSON(t *testing.T, value string) string {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal JSON: %v", err)
	}
	return string(raw)
}

func TestModelGenerateStreamClassifiesProviderRateLimit(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("x-request-id", "req_openai")
		w.Header().Set("Retry-After", "4")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"code":"rate_limit_exceeded","message":"slow down","type":"rate_limit_error"}}`))
	}))
	defer ts.Close()
	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(GPT41Mini)
	if err != nil {
		t.Fatal(err)
	}
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		if token.Type() != ai.TokenTypeErr {
			continue
		}
		var providerErr *ai.ProviderError
		if !errors.As(token.Err, &providerErr) || providerErr.Kind != ai.ProviderErrorRateLimited || providerErr.StatusCode != http.StatusTooManyRequests || providerErr.Code != "rate_limit_exceeded" || providerErr.RequestID != "req_openai" || providerErr.RetryAfter != 4*time.Second || errors.Unwrap(providerErr) == nil {
			t.Fatalf("provider error = %#v", token.Err)
		}
		return
	}
	t.Fatal("expected stream error")
}
