package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/internal/obstest"
)

func testModel(t *testing.T, handler http.HandlerFunc) *Model {
	t.Helper()
	ts := httptest.NewServer(handler)
	t.Cleanup(ts.Close)
	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(ClaudeSonnet4_6)
	if err != nil {
		t.Fatal(err)
	}
	return m.(*Model)
}

func decodeRequest(t *testing.T, r *http.Request) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	return body
}

func object(t *testing.T, value any) map[string]any {
	t.Helper()
	got, ok := value.(map[string]any)
	if !ok {
		t.Fatalf("value = %#v, want JSON object", value)
	}
	return got
}

func array(t *testing.T, value any) []any {
	t.Helper()
	got, ok := value.([]any)
	if !ok {
		t.Fatalf("value = %#v, want JSON array", value)
	}
	return got
}

func TestGenerateSendsAnthropicRequestAndMapsBlocksAndUsage(t *testing.T) {
	recorder := obstest.Install(t)
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" || r.Method != http.MethodPost {
			t.Fatalf("request = %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("x-api-key") != "test-key" || r.Header.Get("anthropic-version") != "2023-06-01" {
			t.Fatalf("unexpected Anthropic headers: %#v", r.Header)
		}
		body := decodeRequest(t, r)
		messages := array(t, body["messages"])
		content := array(t, object(t, messages[0])["content"])
		if body["model"] != ClaudeSonnet4_6 || body["max_tokens"] != float64(4096) || body["stream"] != nil || len(messages) != 1 || object(t, messages[0])["role"] != "user" || len(content) != 1 || object(t, content[0])["type"] != "text" || object(t, content[0])["text"] != "hello" {
			t.Fatalf("unexpected request: %#v", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"answer"},{"type":"thinking","thinking":"reason"},{"type":"tool_use","id":"toolu_1","name":"search","input":{"q":"x"}}],"usage":{"input_tokens":10,"cache_creation_input_tokens":2,"cache_read_input_tokens":3,"output_tokens":4,"output_tokens_details":{"thinking_tokens":2}}}`))
	})

	got, err := m.Generate(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}})
	if err != nil {
		t.Fatal(err)
	}
	if got.Text() != "answer" || got.Reasoning() != "reason" || got.InputTokens != 15 || got.OutputTokens != 4 || got.ReasoningTokens != 2 || len(got.ToolCalls()) != 1 {
		t.Fatalf("unexpected response: %#v", got)
	}
	if got.ToolCalls()[0].ID != "toolu_1" || got.ToolCalls()[0].Type != "function" || got.ToolCalls()[0].Name != "search" || string(got.ToolCalls()[0].Args) != `{"q":"x"}` {
		t.Fatalf("unexpected tool call: %#v", got.ToolCalls()[0])
	}
	attrs := obstest.Attributes(obstest.RequireGenerationSpans(t, recorder, 1)[0])
	if attrs["gen_ai.provider.name"].AsString() != "anthropic" || attrs["gen_ai.usage.input_tokens"].AsInt64() != 15 || attrs["gai.gen_ai.response.tool_call_count"].AsInt64() != 1 {
		t.Fatalf("generation attrs = %#v", attrs)
	}
}

func TestGenerateAndStreamRejectMaxTokens(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		body := decodeRequest(t, r)
		if body["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_limited\",\"model\":\"claude-test\"}}\n\n"))
			_, _ = w.Write([]byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n"))
			_, _ = w.Write([]byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n"))
			_, _ = w.Write([]byte("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"))
			_, _ = w.Write([]byte("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"input_tokens\":5,\"output_tokens\":3}}\n\n"))
			_, _ = w.Write([]byte("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_limited","model":"claude-test","role":"assistant","content":[{"type":"text","text":"partial"}],"stop_reason":"max_tokens","usage":{"input_tokens":5,"output_tokens":3}}`))
	})

	response, err := m.Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}})
	var terminal *ai.TerminalError
	if response != nil || !errors.As(err, &terminal) || terminal.Provider != "anthropic" || terminal.Reason != "max_tokens" {
		t.Fatalf("Generate response = %#v, error = %#v", response, err)
	}

	var tokens []ai.Token
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 3 || tokens[0].Text() != "partial" || tokens[1].Type() != ai.TokenTypeCompletion || tokens[1].Completion.FinishReason != "max_tokens" || tokens[1].Completion.Usage.OutputTokens != 3 || tokens[2].Type() != ai.TokenTypeErr {
		t.Fatalf("tokens = %#v", tokens)
	}
	terminal = nil
	if !errors.As(tokens[2].Err, &terminal) || terminal.Provider != "anthropic" || terminal.Reason != "max_tokens" {
		t.Fatalf("stream error = %#v", tokens[2].Err)
	}
}

func TestGenerateStreamRejectsOutputLimitBeforeTruncatedToolCall(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_limited\",\"model\":\"claude-test\"}}\n\n"))
		_, _ = w.Write([]byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"search\"}}\n\n"))
		_, _ = w.Write([]byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\"}}\n\n"))
		_, _ = w.Write([]byte("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"))
		_, _ = w.Write([]byte("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"input_tokens\":5,\"output_tokens\":3}}\n\n"))
		_, _ = w.Write([]byte("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
	})

	var tokens []ai.Token
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 2 || tokens[0].Type() != ai.TokenTypeCompletion || tokens[0].Completion.FinishReason != "max_tokens" || tokens[0].Completion.Usage.OutputTokens != 3 || tokens[1].Type() != ai.TokenTypeErr {
		t.Fatalf("tokens = %#v", tokens)
	}
	var terminal *ai.TerminalError
	if !errors.As(tokens[1].Err, &terminal) || terminal.Provider != "anthropic" || terminal.Reason != "max_tokens" {
		t.Fatalf("stream error = %#v", tokens[1].Err)
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
	if len(payload) != 1 || payload[0]["role"] != "user" || object(t, array(t, payload[0]["content"])[0])["text"] != "initial request" {
		t.Fatalf("payload = %#v", payload)
	}
}

func TestNativeMessagesGroupAdjacentToolResults(t *testing.T) {
	messages, err := mapNativeMessages([]ai.Message{
		{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "call_1", Name: "search", Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "first"}}}}}},
		{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "call_2", Name: "lookup", Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "second"}}, IsError: true}}}},
	})
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
	if len(payload) != 1 || payload[0]["role"] != "user" {
		t.Fatalf("payload = %#v", payload)
	}
	content := array(t, payload[0]["content"])
	if len(content) != 2 || object(t, content[0])["tool_use_id"] != "call_1" || object(t, content[1])["tool_use_id"] != "call_2" || object(t, content[1])["is_error"] != true {
		t.Fatalf("tool result content = %#v", content)
	}
}

func TestGenerateMapsCapabilitiesAndRejectsUnsupportedResponseFormat(t *testing.T) {
	var got map[string]any
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("anthropic-beta") != "" {
			t.Fatalf("anthropic-beta = %q", r.Header.Get("anthropic-beta"))
		}
		got = decodeRequest(t, r)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[],"usage":{}}`))
	})
	_, err := m.Generate(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}, MaxTokens: 2048,
		Tools:          []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice:     ai.ToolChoice{Mode: ai.ToolChoiceAuto},
		ResponseFormat: ai.ResponseFormat{Type: ai.ResponseFormatJSONSchema, Name: "answer", Schema: json.RawMessage(`{"type":"object"}`)},
		Reasoning:      ai.ReasoningConfig{Enabled: true, BudgetTokens: 1024},
	})
	if err != nil {
		t.Fatal(err)
	}
	tools := array(t, got["tools"])
	tool := object(t, tools[0])
	outputConfig := object(t, got["output_config"])
	format := object(t, outputConfig["format"])
	thinking := object(t, got["thinking"])
	if got["max_tokens"] != float64(2048) || len(tools) != 1 || tool["input_schema"] == nil || object(t, got["tool_choice"])["type"] != "auto" || format["type"] != "json_schema" || thinking["budget_tokens"] != float64(1024) {
		t.Fatalf("unexpected capability mapping: %#v", got)
	}
	_, err = m.Generate(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}, ResponseFormat: ai.ResponseFormat{Type: ai.ResponseFormatJSONObject}})
	if !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("JSON object error = %v, want unsupported capability", err)
	}
	_, err = m.Generate(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}, MaxTokens: 1024, Reasoning: ai.ReasoningConfig{Enabled: true, BudgetTokens: 1024}})
	if !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("invalid thinking budget error = %v", err)
	}
}

func TestGenerateMapsBudgetThinkingForKnownBudgetOnlyModel(t *testing.T) {
	var got map[string]any
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		got = decodeRequest(t, r)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[],"usage":{}}`))
	})
	m.name = ClaudeFable5

	_, err := m.Generate(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}, MaxTokens: 4096,
		Reasoning: ai.ReasoningConfig{Enabled: true, BudgetTokens: 2000},
	})
	if err != nil {
		t.Fatal(err)
	}
	if thinking := object(t, got["thinking"]); thinking["type"] != "enabled" || thinking["budget_tokens"] != float64(2000) {
		t.Fatalf("thinking = %#v, want enabled budget thinking", thinking)
	}
}

func TestGenerateMapsAdaptiveThinkingForCatalogModel(t *testing.T) {
	var got map[string]any
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		got = decodeRequest(t, r)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[],"usage":{}}`))
	})
	m.name = "claude-dynamic"
	m.client.catalog.Replace([]ai.ModelDescriptor{{Model: m.name, Reasoning: ai.FeatureSupportSupported, ReasoningEffort: ai.FeatureSupportSupported, ReasoningEfforts: []ai.ReasoningEffort{ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh}}})

	_, err := m.Generate(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}, Reasoning: ai.ReasoningConfig{Enabled: true, Effort: ai.ReasoningEffortHigh}})
	if err != nil {
		t.Fatal(err)
	}
	if thinking := object(t, got["thinking"]); thinking["type"] != "adaptive" {
		t.Fatalf("thinking = %#v, want adaptive thinking", thinking)
	}
}

func TestGenerateMapsAdaptiveReasoningAndToolChoice(t *testing.T) {
	tests := []struct {
		name    string
		choice  ai.ToolChoice
		want    map[string]any
		wantErr bool
	}{
		{name: "auto", choice: ai.ToolChoice{}, want: map[string]any{"type": "auto"}},
		{name: "none", choice: ai.ToolChoice{Mode: ai.ToolChoiceNone}, want: map[string]any{"type": "none"}},
		{name: "any", choice: ai.ToolChoice{Mode: ai.ToolChoiceRequired}, wantErr: true},
		{name: "named", choice: ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"search"}}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
				body := decodeRequest(t, r)
				if got := object(t, body["tool_choice"]); !equalJSON(got, tt.want) {
					t.Fatalf("tool_choice = %#v, want %#v", got, tt.want)
				}
				thinking := object(t, body["thinking"])
				if thinking["type"] != "adaptive" || thinking["display"] != "omitted" || object(t, body["output_config"])["effort"] != "high" {
					t.Fatalf("unexpected adaptive reasoning: %#v", body)
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"content":[],"usage":{}}`))
			})
			_, err := m.Generate(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}, Tools: []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object"}`)}}, ToolChoice: tt.choice, Reasoning: ai.ReasoningConfig{Enabled: true, Effort: ai.ReasoningEffortHigh}})
			if tt.wantErr {
				if !errors.Is(err, ai.ErrUnsupportedCapability) {
					t.Fatalf("Generate error = %v, want unsupported capability", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
		})
	}
	m := testModel(t, func(http.ResponseWriter, *http.Request) { t.Fatal("unexpected request") })
	if _, err := m.Generate(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}, Tools: []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object"}`)}}, ToolChoice: ai.ToolChoice{Mode: ai.ToolChoiceAuto, Names: []string{"search"}}}); err == nil {
		t.Fatal("restricted auto error = nil, want error")
	}
}

func equalJSON(got, want map[string]any) bool {
	gotJSON, _ := json.Marshal(got)
	wantJSON, _ := json.Marshal(want)
	return string(gotJSON) == string(wantJSON)
}

func TestGenerateReturnsProviderError(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"bad input"}}`))
	})
	_, err := m.Generate(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}})
	var providerErr *Error
	if !errors.As(err, &providerErr) || providerErr.StatusCode != http.StatusBadRequest || providerErr.Type != "invalid_request_error" || providerErr.Message != "bad input" {
		t.Fatalf("unexpected error: %#v", err)
	}
}

func TestGenerateStreamMapsInterleavedBlocksAndToolJSON(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		body := decodeRequest(t, r)
		if body["stream"] != true {
			t.Fatal("stream was not requested")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"search\"}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"thinking\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\\\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"why\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"x\\\"}\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":2}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
	})
	var tokens []ai.Token
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 3 || tokens[0].Type() != ai.TokenTypeText || tokens[0].Text() != "hi" || tokens[1].Type() != ai.TokenTypeThought || tokens[1].Text() != "why" || tokens[2].Type() != ai.TokenTypeToolCall || tokens[2].ToolCall() == nil || string(tokens[2].ToolCall().Args) != `{"q":"x"}` {
		t.Fatalf("unexpected tokens: %#v", tokens)
	}
}

func TestGenerateStreamEmitsTerminalCompletionSnapshot(t *testing.T) {
	recorder := obstest.Install(t)
	m := testModel(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-test\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":10,\"cache_creation_input_tokens\":2,\"cache_read_input_tokens\":3,\"output_tokens\":4,\"output_tokens_details\":{\"thinking_tokens\":1}}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
	})

	var completion *ai.Completion
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		if token.Type() == ai.TokenTypeCompletion {
			completion = token.Completion
		}
	}
	if completion == nil || completion.Provider != "anthropic" || completion.RequestID != "msg_1" || completion.Model != "claude-test" || completion.FinishReason != "end_turn" {
		t.Fatalf("completion = %#v", completion)
	}
	if completion.Usage.InputTokens != 15 || completion.Usage.OutputTokens != 4 || completion.Usage.ReasoningTokens != 1 || completion.Usage.CachedTokens != 3 || completion.Usage.CacheCreationTokens != 2 || !json.Valid(completion.Raw) {
		t.Fatalf("completion usage = %#v, raw = %q", completion.Usage, completion.Raw)
	}
	attrs := obstest.Attributes(obstest.RequireGenerationSpans(t, recorder, 1)[0])
	if !attrs["gai.gen_ai.streaming"].AsBool() || attrs["gen_ai.usage.input_tokens"].AsInt64() != 15 {
		t.Fatalf("generation attrs = %#v", attrs)
	}
}

func TestGenerateStreamErrorAndCancellation(t *testing.T) {
	recorder := obstest.Install(t)
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n"))
	})
	var tokens []ai.Token
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 1 || tokens[0].Type() != ai.TokenTypeErr || tokens[0].Err == nil || !strings.Contains(tokens[0].Err.Error(), "busy") {
		t.Fatalf("unexpected stream error: %#v", tokens)
	}

	started := make(chan struct{})
	release := make(chan struct{})
	m = testModel(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		close(started)
		select {
		case <-r.Context().Done():
		case <-release:
		}
	})
	ctx, cancel := context.WithCancel(context.Background())
	ch := m.GenerateStream(ctx, ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}})
	<-started
	cancel()
	defer close(release)
	select {
	case _, ok := <-ch:
		if ok {
			for range ch {
			}
		}
	case <-time.After(time.Second):
		t.Fatal("stream did not close after cancellation")
	}
	spans := obstest.RequireGenerationSpansEventually(t, recorder, 2, time.Second)
	for _, span := range spans {
		if span.Status().Code.String() != "Error" {
			t.Fatalf("generation status = %s, want Error", span.Status().Code)
		}
	}
}

func TestGenerateStreamDetectsTextFallbackToolCall(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"{\\\"type\\\":\\\"function\\\",\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"\\\"name\\\":\\\"search\\\",\\\"arguments\\\":{}}\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
	})

	var tokens []ai.Token
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 1 || tokens[0].Type() != ai.TokenTypeToolCall || tokens[0].ToolCall() == nil || tokens[0].ToolCall().Name != "search" {
		t.Fatalf("unexpected tokens: %#v", tokens)
	}
}

func TestGenerateStreamRejectsTruncatedToolBlock(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"search\"}}\n\n"))
	})
	var tokens []ai.Token
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 1 || tokens[0].Type() != ai.TokenTypeErr || tokens[0].Err == nil || !strings.Contains(tokens[0].Err.Error(), "open content") {
		t.Fatalf("unexpected tokens: %#v", tokens)
	}
}

func TestGenerateStreamPrefersStreamErrorOverOpenBlock(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"search\"}}\n\nevent: content_block_delta\ndata: not-json\n\n"))
	})
	var tokens []ai.Token
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 1 || tokens[0].Type() != ai.TokenTypeErr || tokens[0].Err == nil || strings.Contains(tokens[0].Err.Error(), "open content") {
		t.Fatalf("unexpected tokens: %#v", tokens)
	}
}

func TestGenerateStreamClassifiesProviderRateLimit(t *testing.T) {
	recorder := obstest.Install(t)
	m := testModel(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("request-id", "req_anthropic")
		w.Header().Set("Retry-After", "3")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`))
	})
	sawError := false
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}) {
		if token.Type() != ai.TokenTypeErr {
			continue
		}
		var providerErr *ai.ProviderError
		if !errors.As(token.Err, &providerErr) || providerErr.Kind != ai.ProviderErrorRateLimited || providerErr.StatusCode != http.StatusTooManyRequests || providerErr.Code != "rate_limit_error" || providerErr.RequestID != "req_anthropic" || providerErr.RetryAfter != 3*time.Second || errors.Unwrap(providerErr) == nil {
			t.Fatalf("provider error = %#v", token.Err)
		}
		sawError = true
	}
	if !sawError {
		t.Fatal("expected stream error")
	}
	attrs := obstest.Attributes(obstest.RequireGenerationSpans(t, recorder, 1)[0])
	if got := attrs["http.response.status_code"].AsInt64(); got != http.StatusTooManyRequests {
		t.Fatalf("HTTP status attribute = %d, want %d", got, http.StatusTooManyRequests)
	}
}
