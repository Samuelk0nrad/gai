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

	if ai.SendToken(ctx, make(chan ai.Token), ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "ignored"}}) {
		t.Fatal("expected canceled send to return false")
	}
}

func TestAIResponseAppendTokenSeparatesThoughtsAndToolCalls(t *testing.T) {
	var response ai.AIResponse

	response.AppendToken(ai.Token{TokenUsage: 2, Part: &ai.ContentPart{Kind: ai.ContentText, Text: "answer"}})
	response.AppendToken(ai.Token{TokenUsage: 3, Part: &ai.ContentPart{Kind: ai.ContentReasoning, Text: "reasoning"}})
	response.AppendToken(ai.Token{

		TokenUsage: 1, Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{
			ID:   "call-1",
			Type: "function",
			Name: "search",
			Args: json.RawMessage(`{"query":"x"}`),
		}},
	})

	if response.Text() != "answer" {
		t.Fatalf("expected visible text only, got %q", response.Text())
	}
	if response.Reasoning() != "reasoning" {
		t.Fatalf("expected reasoning to be separated, got %q", response.Reasoning())
	}
	if len(response.ToolCalls()) != 1 || response.ToolCalls()[0].Name != "search" {
		t.Fatalf("expected tool call to be recorded, got %#v", response.ToolCalls())
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

	response.AppendToken(ai.Token{Completion: &ai.Completion{
		UsageReported: true,
		Usage:         ai.Usage{InputTokens: 10, OutputTokens: 4, ReasoningTokens: 2},
	}})
	response.AppendToken(ai.Token{Completion: &ai.Completion{
		UsageReported: true,
		Usage:         ai.Usage{InputTokens: 12, OutputTokens: 6, ReasoningTokens: 3},
	}})

	if response.InputTokens != 12 || response.OutputTokens != 6 || response.ReasoningTokens != 3 {
		t.Fatalf("completion usage should use the latest provider values, got %#v", response)
	}
}

func TestAIResponseAppendTokenCompletionPreservesUnreportedUsage(t *testing.T) {
	response := ai.AIResponse{InputTokens: 10, OutputTokens: 4, ReasoningTokens: 2}

	response.AppendToken(ai.Token{Completion: &ai.Completion{
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

	response.AppendToken(ai.Token{Completion: &ai.Completion{UsageReported: true}})

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
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: " \n\t{"}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `"id":"call-1","type":"function","name":"echo","arguments":{"x":1}}`}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: " trailing text"}},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"x":1}`},
				{typ: ai.TokenTypeText, data: " trailing text", checkData: true},
			},
		},
		{
			name: "Passes through non-JSON leading text",
			input: []ai.Token{
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "hello"}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: " world"}},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: "hello", checkData: true},
				{typ: ai.TokenTypeText, data: " world", checkData: true},
			},
		},
		{
			name: "Replays pending when non-text arrives before decision",
			input: []ai.Token{
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "  "}},
				{Err: errors.New("boom")},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "after"}},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: "  ", checkData: true},
				{typ: ai.TokenTypeErr},
				{typ: ai.TokenTypeText, data: "after", checkData: true},
			},
		},
		{
			name: "Replays pending when non-JSON text arrives before decision",
			input: []ai.Token{
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "Test\n\n{"}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `"id":"call-1","type":"function","name":"echo","arguments":`}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"x":1}}\n\n trailing text `}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"kind":1} tail`}},
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
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"kind":1} tail`}},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: `{"kind":1} tail`},
			},
		},
		{
			name: "Replays when JSON is not a valid tool call",
			input: []ai.Token{
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"id":"call-1","type":"not-function","name":"echo"}`}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "tail"}},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: `{"id":"call-1","type":"not-function","name":"echo"}`, checkData: true},
				{typ: ai.TokenTypeText, data: "tail", checkData: true},
			},
		},
		{
			name: "Replays unclosed JSON at end of stream",
			input: []ai.Token{
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"id":"call-1","type":"function","name":"echo"`}},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: `{"id":"call-1","type":"function","name":"echo"`, checkData: true},
			},
		},
		{
			name: "Handles braces inside strings",
			input: []ai.Token{
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"id":"call-2","type":"function","name":"echo","arguments":{"msg":"{\\\"a\\\":1}"`}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `,"items":[1,2,3]}}`}},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"items":[1,2,3],"msg":"{\\\"a\\\":1}"}`},
			},
		},
		{
			name: "Defaults missing arguments to empty object",
			input: []ai.Token{
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"id":"call-3","type":"function","name":"echo"}`}},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{}`},
			},
		},
		{
			name: "Detects tool call and preserves trailing text in same token",
			input: []ai.Token{
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"id":"call-10","type":"function","name":"echo","arguments":{"x":1}} trailing`}},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"x":1}`},
				{typ: ai.TokenTypeText, data: " trailing", checkData: true},
			},
		},
		{
			name: "Detects adjacent tool calls in same token",
			input: []ai.Token{
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"id":"call-11","type":"function","name":"echo","arguments":{"x":1}}{"id":"call-12","type":"function","name":"echo","arguments":{"y":2}} tail`}},
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
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"id":"call-11","type":"function","name":"echo","arguments":{"x":1}}`}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"id":"call-12","type":"function","name":"echo","arguments":{"y":2}} tail`}},
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
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "intro\n\n"}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"id":"call-13","type":"function","name":"echo","arguments":{"x":1}}`}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "\n\n"}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"id":"call-14","type":"function","name":"echo","arguments":{"y":2}} tail`}},
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
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "\n"}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "\n"}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"id":"`}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `echo","`}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `type":"function","`}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `name":"echo","`}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `arguments":{"`}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `text":"try`}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: ` the echo tool"}}`}},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"text":"try the echo tool"}`},
			},
		},
		{
			name: "Does not detect tool call when text prefix exists",
			input: []ai.Token{
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "Sure, I can help. "}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"id":"call-9","type":"function","name":"echo","arguments":{"x":1}}`}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: " done"}},
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
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: `{"kind":"event","value":123}`}},
				{Part: &ai.ContentPart{Kind: ai.ContentText, Text: " tail"}},
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
				if got.Type() != expected.typ {
					t.Fatalf("token %d unexpected type: got=%q want=%q", i, got.Type(), expected.typ)
				}

				if expected.checkData && got.Text() != expected.data {
					t.Fatalf("token %d unexpected data: got=%q want=%q", i, got.Text(), expected.data)
				}

				if expected.typ == ai.TokenTypeToolCall {
					if got.ToolCall() == nil {
						t.Fatalf("token %d expected tool call metadata, got nil", i)
					}
					if got.ToolCall().ID == "" {
						t.Fatalf("token %d unexpected empty tool call id", i)
					}
					if got.ToolCall().Type != expected.toolType {
						t.Fatalf("token %d unexpected tool call type: got=%q want=%q", i, got.ToolCall().Type, expected.toolType)
					}
					if got.ToolCall().Name != expected.toolName {
						t.Fatalf("token %d unexpected tool call name: got=%q want=%q", i, got.ToolCall().Name, expected.toolName)
					}
					if normalizeJSON(got.ToolCall().Args) != expected.toolArgsJSON {
						t.Fatalf("token %d unexpected tool call arguments: got=%s want=%s", i, string(got.ToolCall().Args), expected.toolArgsJSON)
					}
				}
			}
		})
	}
}

func normalizeTokens(tokens []ai.Token) []ai.Token {
	var out []ai.Token
	for _, tok := range tokens {
		if tok.Type() == ai.TokenTypeText && len(out) > 0 && out[len(out)-1].Type() == ai.TokenTypeText {
			out[len(out)-1].Part.Text += tok.Text()
			continue
		}
		out = append(out, tok.Clone())
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
	response.AppendToken(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentReasoning, Text: "consider "}})
	response.AppendToken(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentReasoning, Text: "options"}})
	signature := ai.ContentPart{Kind: ai.ContentReasoning, Extensions: []ai.Extension{{
		Namespace: "anthropic", Type: "signature", Data: json.RawMessage(`"opaque-state"`), Required: true,
	}}}
	response.AppendToken(ai.Token{Part: &signature})
	response.AppendToken(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "before "}})
	response.AppendToken(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "call"}})
	call := canonicalCall("first")
	response.AppendToken(ai.Token{Part: &call})
	response.AppendToken(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "after"}})
	response.AppendToken(ai.Token{Err: errors.New("transport failed")})
	response.AppendToken(ai.Token{Completion: &ai.Completion{FinishReason: "tool_calls"}})

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
	if response.Text() != "before callafter" || response.Reasoning() != "consider options" || len(response.ToolCalls()) != 1 || response.FinishReason != "tool_calls" {
		t.Fatalf("derived views disagree with canonical message: %#v", response)
	}
	// Stream producers and event observers may reuse their buffers or edit views.
	signature.Extensions[0].Data[1] = 'X'
	call.ToolCall.Args[6] = 'y'
	response.ToolCalls()[0].Args[6] = 'z'
	if string(parts[0].Extensions[0].Data) != `"opaque-state"` || string(parts[2].ToolCall.Args) != `{"q":"x"}` {
		t.Fatal("stream input or convenience view aliases canonical message")
	}
}

func TestAIResponsePreservesAdjacentSignedReasoningBlocks(t *testing.T) {
	var response ai.AIResponse
	response.AppendToken(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentReasoning, Text: "reasoning"}})
	for _, signature := range []string{"first", "empty-second"} {
		data, err := json.Marshal(signature)
		if err != nil {
			t.Fatal(err)
		}
		response.AppendToken(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentReasoning, Extensions: []ai.Extension{{Namespace: "anthropic", Type: "signature", Data: data, Required: true}}}})
	}
	parts := response.Message.Parts
	if len(parts) != 2 || parts[0].Text != "reasoning" || parts[1].Text != "" {
		t.Fatalf("thinking blocks merged: %#v", parts)
	}
	for i, signature := range []string{`"first"`, `"empty-second"`} {
		if len(parts[i].Extensions) != 1 || string(parts[i].Extensions[0].Data) != signature {
			t.Fatalf("block %d lost its signature: %#v", i, parts[i])
		}
	}
	if err := response.Message.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestAIResponseCanonicalReasoningDeltaCountsUsage(t *testing.T) {
	var response ai.AIResponse
	response.AppendToken(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentReasoning, Text: "think"}, TokenUsage: 3})
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
	response := ai.AIResponse{InputTokens: 12, OutputTokens: 8, Message: ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentReasoning, Text: "stale"}, {Kind: ai.ContentText, Text: "stale"}, {Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{Name: "stale"}}}}}
	response.SetMessage(message)
	if response.Text() != `answer{"ok":true}` || response.Reasoning() != "think" || len(response.ToolCalls()) != 1 || response.ToolCalls()[0].ID != "first" {
		t.Fatalf("SetMessage projections = %#v", response)
	}
	if response.InputTokens != 12 || response.OutputTokens != 8 {
		t.Fatal("SetMessage changed usage metadata")
	}
	message.Parts[0].Text = "changed"
	message.Parts[2].JSON[2] = 'X'
	message.Parts[3].ToolCall.Args[6] = 'y'
	response.ToolCalls()[0].Args[6] = 'z'
	if response.Message.Reasoning() != "think" || response.Message.Text() != `answer{"ok":true}` || string(response.Message.Parts[3].ToolCall.Args) != `{"q":"x"}` {
		t.Fatal("SetMessage did not isolate canonical message from source or convenience view")
	}
}

func TestDetectToolCallsPreservesAuthoritativeCanonicalPart(t *testing.T) {
	part := ai.ContentPart{Kind: ai.ContentText, Text: "canonical output", Extensions: []ai.Extension{{Namespace: "provider", Type: "continuity", Data: json.RawMessage(`"opaque"`)}}}
	in := make(chan ai.Token, 1)
	in <- ai.Token{Part: &part}
	close(in)
	out := collectTokens(ai.DetectToolCallsInStream(t.Context(), in, nil))
	if len(out) != 1 || out[0].Part == nil || out[0].Part.Text != "canonical output" || len(out[0].Part.Extensions) != 1 || out[0].ToolCall() != nil {
		t.Fatalf("canonical part was discarded or parsed as a tool call: %#v", out)
	}
}

func TestTokenCloneIsolatesPayloads(t *testing.T) {
	part := canonicalCall("first")
	part.ToolCall.Extensions = []ai.Extension{{Namespace: "provider", Type: "state", Data: json.RawMessage(`"opaque"`)}}
	token := ai.Token{Part: &part}
	copy := token.Clone()
	copy.Part.ToolCall.ID = "changed"
	copy.Part.ToolCall.Args[6] = 'y'
	copy.Part.ToolCall.Extensions[0].Data[1] = 'X'
	view := token.ToolCall()
	view.Name = "changed"
	view.Args[6] = 'z'
	if token.Part.ToolCall.ID != "first" || token.ToolCall().Name != "search" || string(token.Part.ToolCall.Args) != `{"q":"x"}` || string(token.Part.ToolCall.Extensions[0].Data) != `"opaque"` {
		t.Fatal("cloned call or derived view aliases source")
	}
	completion := ai.Token{Completion: &ai.Completion{Raw: json.RawMessage(`{"id":1}`), Usage: ai.Usage{InputTokens: 12}}}
	copy = completion.Clone()
	copy.Completion.Raw[6] = '9'
	copy.Completion.Usage.InputTokens = 99
	if string(completion.Completion.Raw) != `{"id":1}` || completion.Completion.Usage.InputTokens != 12 {
		t.Fatal("cloned completion aliases source")
	}
}

func TestTokenRequiresExactlyOnePayload(t *testing.T) {
	text := &ai.ContentPart{Kind: ai.ContentText, Text: ""}
	for _, event := range []ai.Token{{}, {Part: text, Err: errors.New("failed")}, {Part: text, Completion: &ai.Completion{}}, {Err: errors.New("failed"), Completion: &ai.Completion{}}, {Part: &ai.ContentPart{Kind: "invalid"}}} {
		if event.Validate() == nil {
			t.Fatalf("invalid token accepted: %#v", event)
		}
	}
	for _, event := range []ai.Token{{Part: text}, {Err: errors.New("failed")}, {Completion: &ai.Completion{}}} {
		if err := event.Validate(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestDetectToolCallsUsesCanonicalTextProtocol(t *testing.T) {
	in := make(chan ai.Token, 2)
	for _, text := range []string{`{"type":"function","name":"echo",`, `"arguments":{"text":"canonical"}}`} {
		part := ai.ContentPart{Kind: ai.ContentText, Text: text}
		in <- ai.Token{Part: &part}
	}
	close(in)
	out := collectTokens(ai.DetectToolCallsInStream(t.Context(), in, nil))
	if len(out) != 1 || out[0].ToolCall() == nil || out[0].ToolCall().Name != "echo" || string(out[0].ToolCall().Args) != `{"text":"canonical"}` {
		t.Fatalf("canonical text protocol lost: %#v", out)
	}
}

func TestDetectToolCallsRejectsInvalidPayloadBeforeParsing(t *testing.T) {
	text := &ai.ContentPart{Kind: ai.ContentText, Text: `{"type":"function","name":"echo","arguments":{}}`}
	for _, token := range []ai.Token{{Part: text, Err: errors.New("transport")}, {Part: text, Completion: &ai.Completion{}}, {Part: &ai.ContentPart{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "call", Name: "echo", Parts: ai.TextParts("result")}}}} {
		in := make(chan ai.Token, 1)
		in <- token
		close(in)
		output := collectTokens(ai.DetectToolCallsInStream(t.Context(), in, nil))
		if len(output) != 1 || output[0].Err == nil || output[0].Part != nil {
			t.Fatalf("malformed model event transformed instead of rejected: %#v", output)
		}
	}
}

func TestDetectToolCallsPreservesTextAndUsageAcrossReplacements(t *testing.T) {
	cases := []struct {
		name    string
		texts   []string
		usage   []int
		visible string
		calls   int
	}{
		{"plain text and empty delta", []string{"hello", "", " world"}, []int{2, 3, 4}, "hello world", 0},
		{"split call and trailing text", []string{`{"type":"function","name":"echo",`, `"arguments":{}}tail`}, []int{2, 3}, "tail", 1},
		{"prose then call", []string{"before\n\n" + `{"type":"function","name":"echo","arguments":{}}tail`}, []int{7}, "before\n\ntail", 1},
		{"whitespace then call", []string{"\n\n", `{"type":"function","name":"echo","arguments":{}}tail`}, []int{2, 5}, "tail", 1},
		{"rejected JSON with suffix", []string{`{"kind":1}tail`}, []int{6}, `{"kind":1}tail`, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in := make(chan ai.Token, len(tc.texts))
			wantUsage := 0
			for i, text := range tc.texts {
				in <- ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: text}, TokenUsage: tc.usage[i]}
				wantUsage += tc.usage[i]
			}
			close(in)
			out := collectTokens(ai.DetectToolCallsInStream(t.Context(), in, nil))
			var response ai.AIResponse
			for _, token := range out {
				if err := token.Validate(); err != nil {
					t.Fatal(err)
				}
				response.AppendToken(token)
			}
			if response.Text() != tc.visible || len(response.ToolCalls()) != tc.calls || response.OutputTokens != wantUsage {
				t.Fatalf("output text=%q calls=%d usage=%d; want %q, %d, %d", response.Text(), len(response.ToolCalls()), response.OutputTokens, tc.visible, tc.calls, wantUsage)
			}
		})
	}
}
