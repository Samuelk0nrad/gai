package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/internal/obstest"
	"google.golang.org/genai"
)

func TestModelDescriptorRejectsUnknownReasoningEffortBeforeTransport(t *testing.T) {
	requests := 0
	ts := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { requests++ }))
	defer ts.Close()
	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(Gemini2_5Flash)
	if err != nil {
		t.Fatal(err)
	}
	_, err = any(m).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffort("maximum")}})
	if !errors.Is(err, ai.ErrUnsupportedCapability) || requests != 0 {
		t.Fatalf("Generate error = %v, requests = %d; want local unsupported error and no request", err, requests)
	}
}

func TestModelGenerateStreamEmitsCompletionForIdentityMetadata(t *testing.T) {
	recorder := obstest.Install(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-test\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hello\"}]}}]}\n\n"))
	}))
	defer server.Close()

	provider := New("test-api-key", nil)
	provider.baseURL = server.URL
	provider.httpClient = server.Client()
	model, err := provider.Model("gemini-test")
	if err != nil {
		t.Fatal(err)
	}

	var completion *ai.Completion
	for token := range model.GenerateStream(t.Context(), ai.AIRequest{Prompt: "hello"}) {
		if token.Err != nil {
			t.Fatalf("unexpected stream error: %v", token.Err)
		}
		if token.Type == ai.TokenTypeCompletion {
			completion = token.Completion
		}
	}
	if completion == nil || completion.Provider != "gemini" || completion.RequestID != "resp_1" || completion.Model != "gemini-test" || !json.Valid(completion.Raw) {
		t.Fatalf("completion = %#v", completion)
	}
	var raw struct {
		ResponseID string `json:"responseId"`
	}
	if err := json.Unmarshal(completion.Raw, &raw); err != nil {
		t.Fatalf("unmarshal completion raw: %v", err)
	}
	if raw.ResponseID != "resp_1" {
		t.Fatalf("completion raw responseId = %q, want resp_1", raw.ResponseID)
	}
	attrs := obstest.Attributes(obstest.RequireGenerationSpans(t, recorder, 1)[0])
	if attrs["gen_ai.provider.name"].AsString() != "gcp.gemini" || attrs["ai.provider"].AsString() != "gemini" || !attrs["gai.gen_ai.streaming"].AsBool() || attrs["gen_ai.response.id"].AsString() != "resp_1" {
		t.Fatalf("generation attrs = %#v", attrs)
	}
}

func TestModelGenerateStreamPreservesCompletionBeforeError(t *testing.T) {
	recorder := obstest.Install(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"responseId\":\"resp_before_error\",\"modelVersion\":\"gemini-resolved\"}\n\n"))
		_, _ = w.Write([]byte("data: {not-json}\n\n"))
	}))
	defer server.Close()

	provider := New("test-api-key", nil)
	provider.baseURL = server.URL
	provider.httpClient = server.Client()
	model, err := provider.Model("gemini-test")
	if err != nil {
		t.Fatal(err)
	}

	var completionCount int
	var streamError error
	for token := range model.GenerateStream(t.Context(), ai.AIRequest{Prompt: "hello"}) {
		if token.Type == ai.TokenTypeCompletion {
			completionCount++
		}
		if token.Type == ai.TokenTypeErr {
			streamError = token.Err
		}
	}
	if streamError == nil || completionCount != 0 {
		t.Fatalf("stream error = %v, completion count = %d", streamError, completionCount)
	}
	span := obstest.RequireGenerationSpans(t, recorder, 1)[0]
	attrs := obstest.Attributes(span)
	if span.Status().Code.String() != "Error" || attrs["gen_ai.response.id"].AsString() != "resp_before_error" || attrs["gen_ai.response.model"].AsString() != "gemini-resolved" {
		t.Fatalf("generation status = %s, attrs = %#v", span.Status().Code, attrs)
	}
}

func TestMapFunctionCall(t *testing.T) {
	got, err := mapFunctionCall(&genai.FunctionCall{
		ID:   "call_1",
		Name: "echo_tool",
		Args: map[string]any{
			"query": "hello",
		},
	})
	if err != nil {
		t.Fatalf("mapFunctionCall error: %v", err)
	}

	if !strings.HasPrefix(got.ID, "call_echo_tool_") {
		t.Fatalf("expected generated tool id for echo_tool, got %q", got.ID)
	}
	if got.Type != "function" {
		t.Fatalf("expected tool call type=function, got %q", got.Type)
	}
	if got.Name != "echo_tool" {
		t.Fatalf("expected tool name to be function name, got %q", got.Name)
	}

	var args map[string]any
	if err := json.Unmarshal(got.Args, &args); err != nil {
		t.Fatalf("unmarshal args: %v", err)
	}
	if args["query"] != "hello" {
		t.Fatalf("unexpected args: %#v", args)
	}
}

func TestNativeContentsMapUserPayload(t *testing.T) {
	contents, err := nativeContents(ai.AIRequest{Messages: []ai.RequestMessage{{Role: ai.RequestMessageRoleUser, Text: "initial request"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(contents) != 1 || contents[0].Role != genai.RoleUser || len(contents[0].Parts) != 1 || contents[0].Parts[0].Text != "initial request" {
		t.Fatalf("contents = %#v", contents)
	}
}

func TestNativeContentsAllowsFunctionNameAfterResult(t *testing.T) {
	contents, err := nativeContents(ai.AIRequest{Messages: []ai.RequestMessage{
		{Role: ai.RequestMessageRoleAssistant, ToolCalls: []ai.RequestToolCall{{ID: "call_1", Name: "echo", Arguments: json.RawMessage(`{"message":"first"}`)}}},
		{Role: ai.RequestMessageRoleTool, ToolResult: &ai.RequestToolResult{ToolCallID: "call_1", Name: "echo", Content: "first"}},
		{Role: ai.RequestMessageRoleAssistant, ToolCalls: []ai.RequestToolCall{{ID: "call_2", Name: "echo", Arguments: json.RawMessage(`{"message":"second"}`)}}},
	}})
	if err != nil {
		t.Fatalf("nativeContents error: %v", err)
	}
	if len(contents) != 3 {
		t.Fatalf("expected three contents, got %#v", contents)
	}
}

func TestNativeContentsPreservesThoughtSignatureOnFunctionCall(t *testing.T) {
	signature := []byte("opaque-thought-signature")
	contents, err := nativeContents(ai.AIRequest{Messages: []ai.RequestMessage{
		{Role: ai.RequestMessageRoleAssistant, ToolCalls: []ai.RequestToolCall{{
			ID:               "call_1",
			Name:             "echo",
			Arguments:        json.RawMessage(`{"message":"hello"}`),
			ThoughtSignature: signature,
		}}},
		{Role: ai.RequestMessageRoleTool, ToolResult: &ai.RequestToolResult{ToolCallID: "call_1", Name: "echo", Content: "hello"}},
	}})
	if err != nil {
		t.Fatalf("nativeContents error: %v", err)
	}
	if got := contents[0].Parts[0].ThoughtSignature; string(got) != string(signature) {
		t.Fatalf("thought signature = %q, want %q", got, signature)
	}
}

func TestGenerateEmitsObservationOnGenerationFailure(t *testing.T) {
	recorder := obstest.Install(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":{"code":429,"message":"generation failed"}}`, http.StatusTooManyRequests)
	}))
	defer server.Close()

	var events []gai.Observation
	provider := New("test-api-key", gai.ObservationSinkFunc(func(_ context.Context, event gai.Observation) {
		events = append(events, event)
	}))
	provider.httpClient = server.Client()
	provider.baseURL = server.URL
	model, err := provider.Model("gemini-test")
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	_, err = any(model).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Prompt: "hello"})
	if err == nil {
		t.Fatal("Generate error = nil, want API error")
	}
	found := false
	for _, event := range events {
		if event.Name == "gemini_generate_content_failed" && event.Err == nil && event.Fields["outcome"] == "error" {
			found = true
		}
	}
	if !found {
		t.Fatalf("gemini_generate_content_failed event not emitted: %#v", events)
	}
	spans := obstest.RequireGenerationSpans(t, recorder, 1)
	if spans[0].Status().Code.String() != "Error" {
		t.Fatalf("generation status = %s", spans[0].Status().Code)
	}
	attrs := obstest.Attributes(spans[0])
	if attrs["http.response.status_code"].AsInt64() != http.StatusTooManyRequests {
		t.Fatalf("HTTP status attribute = %d, want %d", attrs["http.response.status_code"].AsInt64(), http.StatusTooManyRequests)
	}
}

func TestModelGenerateStreamRecordsAPIErrorStatus(t *testing.T) {
	recorder := obstest.Install(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":{"code":429,"message":"rate limited"}}`, http.StatusTooManyRequests)
	}))
	defer server.Close()

	provider := New("test-api-key", nil)
	provider.httpClient = server.Client()
	provider.baseURL = server.URL
	model, err := provider.Model("gemini-test")
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	var streamErr error
	for token := range model.GenerateStream(t.Context(), ai.AIRequest{Prompt: "hello"}) {
		if token.Type == ai.TokenTypeErr {
			streamErr = token.Err
		}
	}
	if streamErr == nil {
		t.Fatal("stream error = nil, want API error")
	}
	var providerErr *ai.ProviderError
	if !errors.As(streamErr, &providerErr) || providerErr.Kind != ai.ProviderErrorRateLimited || providerErr.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("stream error = %#v, want classified rate-limit error", streamErr)
	}
	span := obstest.RequireGenerationSpans(t, recorder, 1)[0]
	attrs := obstest.Attributes(span)
	if attrs["http.response.status_code"].AsInt64() != http.StatusTooManyRequests {
		t.Fatalf("HTTP status attribute = %d, want %d", attrs["http.response.status_code"].AsInt64(), http.StatusTooManyRequests)
	}
}

func TestMapFunctionCallEmptyName(t *testing.T) {
	if _, err := mapFunctionCall(&genai.FunctionCall{ID: "call_1"}); err == nil {
		t.Fatal("expected error for empty function name")
	}
}

func TestMarshalArgsNilDefaultsToObject(t *testing.T) {
	raw, err := marshalArgs(nil)
	if err != nil {
		t.Fatalf("marshalArgs error: %v", err)
	}
	if string(raw) != "{}" {
		t.Fatalf("expected {}, got %s", string(raw))
	}
}

func TestBuildTextToken(t *testing.T) {
	tok := buildTextToken(&genai.Part{Text: "hello"})
	if tok.Type != ai.TokenTypeText {
		t.Fatalf("expected text token, got %s", tok.Type)
	}
	if string(tok.Data) != "hello" {
		t.Fatalf("expected token data to be plain text, got %q", string(tok.Data))
	}
	if tok.Text != "hello" {
		t.Fatalf("expected token text to be set, got %q", tok.Text)
	}
}

func TestBuildThoughtToken(t *testing.T) {
	tok := buildTextToken(&genai.Part{Text: "thinking", Thought: true})
	if tok.Type != ai.TokenTypeThought {
		t.Fatalf("expected thought token, got %s", tok.Type)
	}
	if string(tok.Data) != "thinking" {
		t.Fatalf("expected token data to be plain text, got %q", string(tok.Data))
	}
}

func TestBuildGenerateContentConfigMapsCapabilities(t *testing.T) {
	cfg, err := buildGenerateContentConfig(ai.AIRequest{
		MaxTokens: 64,
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
		Reasoning: ai.ReasoningConfig{
			Enabled:         true,
			IncludeThoughts: true,
			BudgetTokens:    128,
			Effort:          ai.ReasoningEffortHigh,
		},
	})
	if err != nil {
		t.Fatalf("buildGenerateContentConfig error: %v", err)
	}

	if cfg.MaxOutputTokens != 64 {
		t.Fatalf("unexpected max output tokens: %d", cfg.MaxOutputTokens)
	}
	if len(cfg.Tools) != 1 || len(cfg.Tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("expected one function declaration, got %#v", cfg.Tools)
	}
	if cfg.Tools[0].FunctionDeclarations[0].Name != "search" {
		t.Fatalf("unexpected function declaration: %#v", cfg.Tools[0].FunctionDeclarations[0])
	}
	if cfg.ToolConfig == nil || cfg.ToolConfig.FunctionCallingConfig == nil {
		t.Fatal("expected function calling config")
	}
	if cfg.ToolConfig.FunctionCallingConfig.Mode != genai.FunctionCallingConfigModeAny {
		t.Fatalf("unexpected function calling mode: %s", cfg.ToolConfig.FunctionCallingConfig.Mode)
	}
	if len(cfg.ToolConfig.FunctionCallingConfig.AllowedFunctionNames) != 1 || cfg.ToolConfig.FunctionCallingConfig.AllowedFunctionNames[0] != "search" {
		t.Fatalf("unexpected allowed functions: %#v", cfg.ToolConfig.FunctionCallingConfig.AllowedFunctionNames)
	}
	if cfg.ResponseMIMEType != "application/json" || cfg.ResponseJsonSchema == nil {
		t.Fatalf("expected JSON response schema, got mime=%q schema=%#v", cfg.ResponseMIMEType, cfg.ResponseJsonSchema)
	}
	if cfg.ThinkingConfig == nil || !cfg.ThinkingConfig.IncludeThoughts {
		t.Fatalf("expected thinking config with included thoughts, got %#v", cfg.ThinkingConfig)
	}
	if cfg.ThinkingConfig.ThinkingBudget == nil || *cfg.ThinkingConfig.ThinkingBudget != 128 {
		t.Fatalf("unexpected thinking budget: %#v", cfg.ThinkingConfig.ThinkingBudget)
	}
	if cfg.ThinkingConfig.ThinkingLevel != genai.ThinkingLevelHigh {
		t.Fatalf("unexpected thinking level: %s", cfg.ThinkingConfig.ThinkingLevel)
	}
}

func TestBuildGenerateContentConfigRejectsUnsupportedToolChoices(t *testing.T) {
	tool := ai.ToolDefinition{
		Type:        "function",
		Name:        "search",
		Description: "Searches documents.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
	}

	tests := []struct {
		name    string
		choice  ai.ToolChoice
		wantErr string
	}{
		{
			name: "auto names unsupported",
			choice: ai.ToolChoice{
				Mode:  ai.ToolChoiceAuto,
				Names: []string{"search"},
			},
			wantErr: "Gemini SDK cannot enforce allowed tool names in auto mode",
		},
		{
			name: "none names invalid",
			choice: ai.ToolChoice{
				Mode:  ai.ToolChoiceNone,
				Names: []string{"search"},
			},
			wantErr: "no tools may be called",
		},
		{
			name: "unknown mode",
			choice: ai.ToolChoice{
				Mode: "sometimes",
			},
			wantErr: `unsupported gemini tool choice mode "sometimes"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := buildGenerateContentConfig(ai.AIRequest{
				Tools:      []ai.ToolDefinition{tool},
				ToolChoice: tt.choice,
			})
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %q", tt.wantErr, err.Error())
			}
		})
	}
}

func TestMapGenerateContentResponseSeparatesTextReasoningAndToolCalls(t *testing.T) {
	text, reasoning, toolCalls, err := mapGenerateContentResponse(&genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{Text: "visible"},
						{Text: "private", Thought: true},
						{FunctionCall: &genai.FunctionCall{Name: "search", Args: map[string]any{"query": "x"}}},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("mapGenerateContentResponse error: %v", err)
	}
	if text != "visible" {
		t.Fatalf("unexpected text: %q", text)
	}
	if reasoning != "private" {
		t.Fatalf("unexpected reasoning: %q", reasoning)
	}
	if len(toolCalls) != 1 || toolCalls[0].Name != "search" {
		t.Fatalf("unexpected tool calls: %#v", toolCalls)
	}
}

func TestModelTokenizer(t *testing.T) {
	m := &Model{name: "gemini-2.5-flash"}
	tokenizer := m.Tokenizer()
	if tokenizer == nil {
		t.Fatal("expected tokenizer")
	}
	if tokenizer.ID() != "gemini.gemini-2.5-flash" {
		t.Fatalf("unexpected tokenizer ID: %q", tokenizer.ID())
	}
}
