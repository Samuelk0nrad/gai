package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestClassifyProviderErrorPrioritizesUnsupported(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		code       string
	}{
		{name: "not implemented", statusCode: http.StatusNotImplemented},
		{name: "unsupported server error", statusCode: http.StatusInternalServerError, code: "unsupported_feature"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ai.ClassifyProviderError(errors.New("provider error"), tt.statusCode, tt.code, "", nil)
			var providerErr *ai.ProviderError
			if !errors.As(err, &providerErr) {
				t.Fatalf("error = %T, want *ai.ProviderError", err)
			}
			if providerErr.Kind != ai.ProviderErrorUnsupported {
				t.Fatalf("kind = %q, want %q", providerErr.Kind, ai.ProviderErrorUnsupported)
			}
		})
	}
}

func TestSendTokenStopsWhenContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if ai.SendToken(ctx, make(chan ai.Token), ai.Token{Type: ai.TokenTypeText, Text: "ignored"}) {
		t.Fatal("expected canceled send to return false")
	}
}

func TestAIResponseAppendTokenSeparatesThoughtsAndToolCalls(t *testing.T) {
	var response ai.AIResponse

	response.AppendToken(ai.Token{Type: ai.TokenTypeText, Text: "answer", TokenUsage: 2})
	response.AppendToken(ai.Token{Type: ai.TokenTypeThought, Text: "reasoning", TokenUsage: 3})
	response.AppendToken(ai.Token{
		Type: ai.TokenTypeToolCall,
		ToolCall: &ai.ToolCall{
			ID:   "call-1",
			Type: "function",
			Name: "search",
			Args: json.RawMessage(`{"query":"x"}`),
		},
		TokenUsage: 1,
	})

	if response.Text != "answer" {
		t.Fatalf("expected visible text only, got %q", response.Text)
	}
	if response.Reasoning != "reasoning" {
		t.Fatalf("expected reasoning to be separated, got %q", response.Reasoning)
	}
	if len(response.ToolCalls) != 1 || response.ToolCalls[0].Name != "search" {
		t.Fatalf("expected tool call to be recorded, got %#v", response.ToolCalls)
	}
	if response.OutputTokens != 6 {
		t.Fatalf("unexpected output tokens: %d", response.OutputTokens)
	}
	if response.ReasoningTokens != 3 {
		t.Fatalf("unexpected reasoning tokens: %d", response.ReasoningTokens)
	}
}

func TestAIResponseAppendTokenCompletionUsesLatestUsage(t *testing.T) {
	var response ai.AIResponse

	response.AppendToken(ai.Token{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{
		UsageReported: true,
		Usage:         ai.Usage{InputTokens: 10, OutputTokens: 4, ReasoningTokens: 2},
	}})
	response.AppendToken(ai.Token{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{
		UsageReported: true,
		Usage:         ai.Usage{InputTokens: 12, OutputTokens: 6, ReasoningTokens: 3},
	}})

	if response.InputTokens != 12 || response.OutputTokens != 6 || response.ReasoningTokens != 3 {
		t.Fatalf("completion usage should use the latest provider values, got %#v", response)
	}
}

func TestAIResponseAppendTokenCompletionPreservesUnreportedUsage(t *testing.T) {
	response := ai.AIResponse{InputTokens: 10, OutputTokens: 4, ReasoningTokens: 2}

	response.AppendToken(ai.Token{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{
		FinishReason: "stop",
		Raw:          json.RawMessage(`{"id":"response-1"}`),
	}})

	if response.InputTokens != 10 || response.OutputTokens != 4 || response.ReasoningTokens != 2 {
		t.Fatalf("unreported completion usage should preserve accumulated values, got %#v", response)
	}
	if response.FinishReason != "stop" || string(response.Raw) != `{"id":"response-1"}` {
		t.Fatalf("completion metadata should still be updated, got %#v", response)
	}
}

func TestAIResponseAppendTokenCompletionAcceptsReportedZeroUsage(t *testing.T) {
	response := ai.AIResponse{InputTokens: 10, OutputTokens: 4, ReasoningTokens: 2}

	response.AppendToken(ai.Token{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{UsageReported: true}})

	if response.InputTokens != 0 || response.OutputTokens != 0 || response.ReasoningTokens != 0 {
		t.Fatalf("reported zero usage should replace accumulated values, got %#v", response)
	}
}

func TestUsageAddPreservesProviderSpecificBreakdowns(t *testing.T) {
	usage := ai.Usage{InputTokens: 1, CachedTokens: 2, CacheCreationTokens: 3, ToolUseTokens: 4}
	usage.Add(ai.Usage{InputTokens: 5, CachedTokens: 6, CacheCreationTokens: 7, ToolUseTokens: 8})
	if usage.InputTokens != 6 || usage.CachedTokens != 8 || usage.CacheCreationTokens != 10 || usage.ToolUseTokens != 12 {
		t.Fatalf("usage = %#v", usage)
	}
}

type expectedWrapToken struct {
	typ          ai.TokenType
	data         string
	checkData    bool
	toolType     string
	toolName     string
	toolArgsJSON string
}

func TestDetectToolCallsInStream(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		input  []ai.Token
		output []expectedWrapToken
	}{
		{
			name: "Detects leading tool call and passes through remainder",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(" \n\t{")},
				{Type: ai.TokenTypeText, Data: []byte(`"id":"call-1","type":"function","name":"echo","arguments":{"x":1}}`)},
				{Type: ai.TokenTypeText, Data: []byte(" trailing text")},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"x":1}`},
				{typ: ai.TokenTypeText, data: " trailing text", checkData: true},
			},
		},
		{
			name: "Passes through non-JSON leading text",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte("hello")},
				{Type: ai.TokenTypeText, Data: []byte(" world")},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: "hello", checkData: true},
				{typ: ai.TokenTypeText, data: " world", checkData: true},
			},
		},
		{
			name: "Replays pending when non-text arrives before decision",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte("  ")},
				{Type: ai.TokenTypeErr, Data: []byte("boom")},
				{Type: ai.TokenTypeText, Data: []byte("after")},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: "  ", checkData: true},
				{typ: ai.TokenTypeErr, data: "boom", checkData: true},
				{typ: ai.TokenTypeText, data: "after", checkData: true},
			},
		},
		{
			name: "Replays pending when non-JSON text arrives before decision",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte("Test\n\n{")},
				{Type: ai.TokenTypeText, Data: []byte(`"id":"call-1","type":"function","name":"echo","arguments":`)},
				{Type: ai.TokenTypeText, Data: []byte(`{"x":1}}\n\n trailing text `)},
				{Type: ai.TokenTypeText, Data: []byte(`{"kind":1} tail`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: "Test\n\n", checkData: true},
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"x":1}`},
				{typ: ai.TokenTypeText, data: `\n\n trailing text {"kind":1} tail`, checkData: true},
			},
		},
		{
			name: "Validates that JSON must be a tool call, not just any JSON",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"kind":1} tail`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: `{"kind":1} tail`},
			},
		},
		{
			name: "Replays when JSON is not a valid tool call",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-1","type":"not-function","name":"echo"}`)},
				{Type: ai.TokenTypeText, Data: []byte("tail")},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: `{"id":"call-1","type":"not-function","name":"echo"}`, checkData: true},
				{typ: ai.TokenTypeText, data: "tail", checkData: true},
			},
		},
		{
			name: "Replays unclosed JSON at end of stream",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-1","type":"function","name":"echo"`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: `{"id":"call-1","type":"function","name":"echo"`, checkData: true},
			},
		},
		{
			name: "Handles braces inside strings",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-2","type":"function","name":"echo","arguments":{"msg":"{\\\"a\\\":1}"`)},
				{Type: ai.TokenTypeText, Data: []byte(`,"items":[1,2,3]}}`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"items":[1,2,3],"msg":"{\\\"a\\\":1}"}`},
			},
		},
		{
			name: "Defaults missing arguments to empty object",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-3","type":"function","name":"echo"}`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{}`},
			},
		},
		{
			name: "Detects tool call and preserves trailing text in same token",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-10","type":"function","name":"echo","arguments":{"x":1}} trailing`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"x":1}`},
				{typ: ai.TokenTypeText, data: " trailing", checkData: true},
			},
		},
		{
			name: "Detects adjacent tool calls in same token",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-11","type":"function","name":"echo","arguments":{"x":1}}{"id":"call-12","type":"function","name":"echo","arguments":{"y":2}} tail`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"x":1}`},
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"y":2}`},
				{typ: ai.TokenTypeText, data: " tail", checkData: true},
			},
		},
		{
			name: "Detects adjacent tool calls across token boundary",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-11","type":"function","name":"echo","arguments":{"x":1}}`)},
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-12","type":"function","name":"echo","arguments":{"y":2}} tail`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"x":1}`},
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"y":2}`},
				{typ: ai.TokenTypeText, data: " tail", checkData: true},
			},
		},
		{
			name: "Detects multiple tool calls separated by blank lines",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte("intro\n\n")},
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-13","type":"function","name":"echo","arguments":{"x":1}}`)},
				{Type: ai.TokenTypeText, Data: []byte("\n\n")},
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-14","type":"function","name":"echo","arguments":{"y":2}} tail`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: "intro\n\n", checkData: true},
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"x":1}`},
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"y":2}`},
				{typ: ai.TokenTypeText, data: " tail", checkData: true},
			},
		},
		{
			name: "Detects tool call from production-like chunked JSON",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte("\n")},
				{Type: ai.TokenTypeText, Data: []byte("\n")},
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"`)},
				{Type: ai.TokenTypeText, Data: []byte(`echo","`)},
				{Type: ai.TokenTypeText, Data: []byte(`type":"function","`)},
				{Type: ai.TokenTypeText, Data: []byte(`name":"echo","`)},
				{Type: ai.TokenTypeText, Data: []byte(`arguments":{"`)},
				{Type: ai.TokenTypeText, Data: []byte(`text":"try`)},
				{Type: ai.TokenTypeText, Data: []byte(` the echo tool"}}`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"text":"try the echo tool"}`},
			},
		},
		{
			name: "Does not detect tool call when text prefix exists",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte("Sure, I can help. ")},
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-9","type":"function","name":"echo","arguments":{"x":1}}`)},
				{Type: ai.TokenTypeText, Data: []byte(" done")},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: "Sure, I can help. ", checkData: true},
				{typ: ai.TokenTypeText, data: `{"id":"call-9","type":"function","name":"echo","arguments":{"x":1}}`, checkData: true},
				{typ: ai.TokenTypeText, data: " done", checkData: true},
			},
		},
		{
			name: "Preserves non-tool JSON object",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"kind":"event","value":123}`)},
				{Type: ai.TokenTypeText, Data: []byte(" tail")},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: `{"kind":"event","value":123}`, checkData: true},
				{typ: ai.TokenTypeText, data: " tail", checkData: true},
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			in := make(chan ai.Token, len(tt.input))
			for _, tok := range tt.input {
				in <- tok
			}
			close(in)

			out := normalizeTokens(collectTokens(ai.DetectToolCallsInStream(t.Context(), in, nil)))
			expectedOut := normalizeExpectedTokens(tt.output)
			if len(out) != len(expectedOut) {
				t.Fatalf("expected %d output tokens, got %d: %#v", len(expectedOut), len(out), out)
			}

			for i, expected := range expectedOut {
				got := out[i]
				if got.Type != expected.typ {
					t.Fatalf("token %d unexpected type: got=%q want=%q", i, got.Type, expected.typ)
				}

				if expected.checkData && string(got.Data) != expected.data {
					t.Fatalf("token %d unexpected data: got=%q want=%q", i, string(got.Data), expected.data)
				}

				if expected.typ == ai.TokenTypeToolCall {
					if got.ToolCall == nil {
						t.Fatalf("token %d expected tool call metadata, got nil", i)
					}
					if got.ToolCall.ID == "" {
						t.Fatalf("token %d unexpected empty tool call id", i)
					}
					if got.ToolCall.Type != expected.toolType {
						t.Fatalf("token %d unexpected tool call type: got=%q want=%q", i, got.ToolCall.Type, expected.toolType)
					}
					if got.ToolCall.Name != expected.toolName {
						t.Fatalf("token %d unexpected tool call name: got=%q want=%q", i, got.ToolCall.Name, expected.toolName)
					}
					if normalizeJSON(got.ToolCall.Args) != expected.toolArgsJSON {
						t.Fatalf("token %d unexpected tool call arguments: got=%s want=%s", i, string(got.ToolCall.Args), expected.toolArgsJSON)
					}
				}
			}
		})
	}
}

func normalizeTokens(tokens []ai.Token) []ai.Token {
	var out []ai.Token
	for _, tok := range tokens {
		if tok.Type == ai.TokenTypeText && len(out) > 0 && out[len(out)-1].Type == ai.TokenTypeText {
			out[len(out)-1].Data = append(out[len(out)-1].Data, tok.Data...)
			continue
		}
		out = append(out, tok)
	}
	return out
}

func normalizeExpectedTokens(tokens []expectedWrapToken) []expectedWrapToken {
	var out []expectedWrapToken
	for _, tok := range tokens {
		if tok.typ == ai.TokenTypeText && len(out) > 0 && out[len(out)-1].typ == ai.TokenTypeText {
			out[len(out)-1].data += tok.data
			out[len(out)-1].checkData = out[len(out)-1].checkData || tok.checkData || tok.data != ""
			continue
		}
		out = append(out, tok)
	}
	return out
}

func collectTokens(in <-chan ai.Token) []ai.Token {
	var out []ai.Token
	for t := range in {
		out = append(out, t)
	}
	return out
}

func normalizeJSON(v []byte) string {
	var payload any
	if err := json.Unmarshal(v, &payload); err != nil {
		return string(v)
	}

	b, err := json.Marshal(payload)
	if err != nil {
		return string(v)
	}
	return string(b)
}

func TestAIResponseCanonicalDeltasPreserveOrderAndMetadata(t *testing.T) {
	var response ai.AIResponse
	response.AppendToken(ai.Token{Type: ai.TokenTypeThought, Text: "consider "})
	response.AppendToken(ai.Token{Type: ai.TokenTypeThought, Text: "options"})
	signature := ai.ContentPart{Kind: ai.ContentReasoning, Extensions: []ai.Extension{{
		Namespace: "anthropic", Type: "signature", Data: json.RawMessage(`"opaque-state"`), Required: true,
	}}}
	response.AppendToken(ai.Token{Type: ai.TokenTypePart, Part: &signature})
	response.AppendToken(ai.Token{Type: ai.TokenTypeText, Text: "before "})
	response.AppendToken(ai.Token{Type: ai.TokenTypeText, Data: []byte("call")})
	call := canonicalCall("first")
	response.AppendToken(ai.Token{Type: ai.TokenTypePart, Part: &call, Text: "ignored compatibility text"})
	response.AppendToken(ai.Token{Type: ai.TokenTypeText, Text: "after"})
	response.AppendToken(ai.Token{Type: ai.TokenTypeErr, Err: errors.New("transport failed"), Data: []byte("not conversation")})
	response.AppendToken(ai.Token{Type: ai.TokenTypeCompletion, Data: []byte("not conversation"), Completion: &ai.Completion{FinishReason: "tool_calls"}})

	parts := response.Message.Parts
	if response.Message.Role != ai.RoleAssistant || len(parts) != 4 {
		t.Fatalf("unexpected accumulated message: %#v", response.Message)
	}
	if parts[0].Kind != ai.ContentReasoning || parts[0].Text != "consider options" || len(parts[0].Extensions) != 1 || string(parts[0].Extensions[0].Data) != `"opaque-state"` {
		t.Fatalf("thought metadata must attach to its reasoning part: %#v", parts[0])
	}
	if parts[1].Kind != ai.ContentText || parts[1].Text != "before call" || parts[2].Kind != ai.ContentToolCall || parts[2].ToolCall.ID != "first" || parts[3].Text != "after" {
		t.Fatalf("semantic part order changed: %#v", parts)
	}
	if response.Text != "before callafter" || response.Reasoning != "consider options" || len(response.ToolCalls) != 1 || response.FinishReason != "tool_calls" {
		t.Fatalf("compatibility projections disagree with canonical message: %#v", response)
	}
	// Stream producers and event observers may reuse their buffers or edit views.
	signature.Extensions[0].Data[1] = 'X'
	call.ToolCall.Args[6] = 'y'
	response.ToolCalls[0].Args[6] = 'z'
	if string(parts[0].Extensions[0].Data) != `"opaque-state"` || string(parts[2].ToolCall.Args) != `{"q":"x"}` {
		t.Fatal("stream input or convenience view aliases canonical message")
	}
}

func TestAIResponseCanonicalReasoningDeltaCountsUsage(t *testing.T) {
	var response ai.AIResponse
	response.AppendToken(ai.Token{Type: ai.TokenTypePart, Part: &ai.ContentPart{Kind: ai.ContentReasoning, Text: "think"}, TokenUsage: 3})
	if response.OutputTokens != 3 || response.ReasoningTokens != 3 {
		t.Fatalf("canonical reasoning delta usage = output %d, reasoning %d; want 3, 3", response.OutputTokens, response.ReasoningTokens)
	}
}

func TestAIResponseSetMessageReplacesAndSnapshotsProjections(t *testing.T) {
	call := canonicalCall("first")
	message := ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{
		{Kind: ai.ContentReasoning, Text: "think"},
		{Kind: ai.ContentText, Text: "answer"},
		{Kind: ai.ContentJSON, JSON: json.RawMessage(`{"ok":true}`)},
		call,
	}}
	response := ai.AIResponse{Text: "stale", Reasoning: "stale", ToolCalls: []ai.ToolCall{{Name: "stale"}}, InputTokens: 12, OutputTokens: 8}
	response.SetMessage(message)
	if response.Text != `answer{"ok":true}` || response.Reasoning != "think" || len(response.ToolCalls) != 1 || response.ToolCalls[0].ID != "first" {
		t.Fatalf("SetMessage projections = %#v", response)
	}
	if response.InputTokens != 12 || response.OutputTokens != 8 {
		t.Fatal("SetMessage changed usage metadata")
	}
	message.Parts[0].Text = "changed"
	message.Parts[2].JSON[2] = 'X'
	message.Parts[3].ToolCall.Args[6] = 'y'
	response.ToolCalls[0].Args[6] = 'z'
	if response.Message.Reasoning() != "think" || response.Message.Text() != `answer{"ok":true}` || string(response.Message.Parts[3].ToolCall.Args) != `{"q":"x"}` {
		t.Fatal("SetMessage did not isolate canonical message from source or convenience view")
	}
}

func TestDetectToolCallsPreservesAuthoritativeCanonicalPart(t *testing.T) {
	part := ai.ContentPart{Kind: ai.ContentText, Text: "canonical output", Extensions: []ai.Extension{{Namespace: "provider", Type: "continuity", Data: json.RawMessage(`"opaque"`)}}}
	in := make(chan ai.Token, 1)
	in <- ai.Token{Type: ai.TokenTypeText, Part: &part, Text: "stale view", Data: []byte(`{"type":"function","name":"unintended","arguments":{}}`)}
	close(in)
	out := collectTokens(ai.DetectToolCallsInStream(t.Context(), in, nil))
	if len(out) != 1 || out[0].Part == nil || out[0].Part.Text != "canonical output" || len(out[0].Part.Extensions) != 1 || out[0].ToolCall != nil {
		t.Fatalf("canonical part was discarded or parsed as a legacy tool call: %#v", out)
	}
}

func TestTokenCloneIsolatesSemanticAndCompletionPayloads(t *testing.T) {
	part := canonicalCall("first")
	part.ToolCall.Extensions = []ai.Extension{{Namespace: "provider", Type: "state", Data: json.RawMessage(`"opaque"`)}}
	token := ai.Token{Type: ai.TokenTypePart, Data: []byte("raw"), Part: &part, ToolCall: part.ToolCall, Completion: &ai.Completion{Raw: json.RawMessage(`{"id":1}`), Usage: ai.Usage{InputTokens: 12}}}
	copy := token.Clone()
	copy.Data[0] = 'X'
	copy.Part.ToolCall.ID = "changed"
	copy.Part.ToolCall.Args[6] = 'y'
	copy.Part.ToolCall.Extensions[0].Data[1] = 'X'
	copy.ToolCall.Name = "changed"
	copy.ToolCall.Args[6] = 'z'
	copy.Completion.Raw[6] = '9'
	copy.Completion.Usage.InputTokens = 99
	if string(token.Data) != "raw" || token.Part.ToolCall.ID != "first" || token.ToolCall.Name != "search" || string(token.ToolCall.Args) != `{"q":"x"}` || string(token.Part.ToolCall.Extensions[0].Data) != `"opaque"` || string(token.Completion.Raw) != `{"id":1}` || token.Completion.Usage.InputTokens != 12 {
		t.Fatalf("cloned token aliases source payloads: %#v", token)
	}
}

func TestDetectToolCallsUsesCanonicalTextForCompatibilityProtocol(t *testing.T) {
	in := make(chan ai.Token, 2)
	for _, text := range []string{`{"type":"function","name":"echo",`, `"arguments":{"text":"canonical"}}`} {
		part := ai.ContentPart{Kind: ai.ContentText, Text: text}
		in <- ai.Token{Type: ai.TokenTypeText, Part: &part, Data: []byte("stale")}
	}
	close(in)
	out := collectTokens(ai.DetectToolCallsInStream(t.Context(), in, nil))
	if len(out) != 1 || out[0].ToolCall == nil || out[0].ToolCall.Name != "echo" || string(out[0].ToolCall.Args) != `{"text":"canonical"}` {
		t.Fatalf("canonical text protocol lost: %#v", out)
	}
}
