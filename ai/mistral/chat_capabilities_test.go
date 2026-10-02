package mistral

import (
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/lace-ai/gai/ai"
)

// chatServer records raw request bodies and replies with the queued responses.
// A response starting with "data:" is sent as an SSE stream.
type chatServer struct {
	mu        sync.Mutex
	bodies    [][]byte
	responses []string
}

func newChatServer(t *testing.T, responses ...string) (*chatServer, *Provider) {
	t.Helper()
	cs := &chatServer{responses: responses}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		body, _ := io.ReadAll(r.Body)
		cs.mu.Lock()
		cs.bodies = append(cs.bodies, body)
		index := len(cs.bodies) - 1
		cs.mu.Unlock()
		if index >= len(cs.responses) {
			http.Error(w, "unexpected request", http.StatusBadRequest)
			return
		}
		response := cs.responses[index]
		if strings.HasPrefix(response, "data:") {
			w.Header().Set("Content-Type", "text/event-stream")
		} else {
			w.Header().Set("Content-Type", "application/json")
		}
		_, _ = io.WriteString(w, response)
	}))
	t.Cleanup(server.Close)
	p := New("test-key", nil, WithBaseURL(server.URL), WithHTTPClient(server.Client()))
	return cs, p
}

func (cs *chatServer) requests() [][]byte {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	return append([][]byte(nil), cs.bodies...)
}

func (cs *chatServer) request(t *testing.T, index int) map[string]any {
	t.Helper()
	bodies := cs.requests()
	if index >= len(bodies) {
		t.Fatalf("request %d not sent; got %d requests", index, len(bodies))
	}
	var decoded map[string]any
	if err := json.Unmarshal(bodies[index], &decoded); err != nil {
		t.Fatal(err)
	}
	return decoded
}

func sse(events ...string) string {
	var out strings.Builder
	for _, event := range events {
		out.WriteString("data: " + event + "\n\n")
	}
	out.WriteString("data: [DONE]\n\n")
	return out.String()
}

func userRequest(text string) ai.AIRequest {
	return ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, text)}}
}

func streamResponse(t *testing.T, model *Model, req ai.AIRequest) (ai.AIResponse, []ai.Token, error) {
	t.Helper()
	var response ai.AIResponse
	var tokens []ai.Token
	var streamErr error
	for token := range model.GenerateStream(t.Context(), req) {
		if err := token.Validate(); err != nil {
			t.Fatalf("invalid token %#v: %v", token, err)
		}
		tokens = append(tokens, token)
		if token.Err != nil {
			streamErr = token.Err
			continue
		}
		response.AppendToken(token)
	}
	return response, tokens, streamErr
}

func signatureOf(t *testing.T, part ai.ContentPart) string {
	t.Helper()
	signature, err := thinkingSignature(part.Extensions)
	if err != nil {
		t.Fatal(err)
	}
	return signature
}

func partKinds(parts []ai.ContentPart) []ai.ContentKind {
	kinds := make([]ai.ContentKind, len(parts))
	for i, part := range parts {
		kinds[i] = part.Kind
	}
	return kinds
}

const thinkingResponse = `{"id":"req-1","model":"mistral-small-latest","choices":[{"message":{"role":"assistant","content":[` +
	`{"type":"thinking","thinking":[{"type":"text","text":"Consider "},{"type":"text","text":"the weather."}],"signature":"sig-1"},` +
	`{"type":"text","text":"Checking now."}],` +
	`"tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}}]},` +
	`"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":9}}`

func TestGenerateDecodesStringAndChunkContent(t *testing.T) {
	cases := []struct {
		name      string
		content   string
		wantKinds []ai.ContentKind
		wantText  string
		wantThink string
	}{
		{name: "string", content: `"plain answer"`, wantKinds: []ai.ContentKind{ai.ContentText}, wantText: "plain answer"},
		{name: "null", content: `null`, wantKinds: []ai.ContentKind{ai.ContentText}},
		{name: "empty array", content: `[]`, wantKinds: []ai.ContentKind{ai.ContentText}},
		{name: "thinking then text", content: `[{"type":"thinking","thinking":[{"type":"text","text":"why"}]},{"type":"text","text":"answer"}]`,
			wantKinds: []ai.ContentKind{ai.ContentReasoning, ai.ContentText}, wantText: "answer", wantThink: "why"},
		{name: "untyped text chunk", content: `[{"text":"legacy"}]`, wantKinds: []ai.ContentKind{ai.ContentText}, wantText: "legacy"},
		{name: "adjacent text merges", content: `[{"type":"text","text":"a"},{"type":"text","text":"b"}]`,
			wantKinds: []ai.ContentKind{ai.ContentText}, wantText: "ab"},
		{name: "interleaved order kept", content: `[{"type":"thinking","thinking":[{"type":"text","text":"1"}]},{"type":"text","text":"2"},{"type":"thinking","thinking":[{"type":"text","text":"3"}]}]`,
			wantKinds: []ai.ContentKind{ai.ContentReasoning, ai.ContentText, ai.ContentReasoning}, wantText: "2", wantThink: "13"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, p := newChatServer(t, `{"choices":[{"message":{"content":`+tc.content+`},"finish_reason":"stop"}]}`)
			model, _ := p.TypedModel(MistralSmallLatest)
			response, err := model.Generate(t.Context(), userRequest("hi"))
			if err != nil {
				t.Fatal(err)
			}
			if got := partKinds(response.Message.Parts); !reflect.DeepEqual(got, tc.wantKinds) {
				t.Fatalf("kinds = %v, want %v", got, tc.wantKinds)
			}
			if response.Text() != tc.wantText || response.Reasoning() != tc.wantThink {
				t.Fatalf("text = %q, reasoning = %q", response.Text(), response.Reasoning())
			}
			if err := response.Message.Validate(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestGenerateKeepsReasoningTextAndToolCallOrder(t *testing.T) {
	_, p := newChatServer(t, thinkingResponse)
	model, _ := p.TypedModel(MistralSmallLatest)
	response, err := model.Generate(t.Context(), userRequest("weather?"))
	if err != nil {
		t.Fatal(err)
	}
	parts := response.Message.Parts
	if got := partKinds(parts); !reflect.DeepEqual(got, []ai.ContentKind{ai.ContentReasoning, ai.ContentText, ai.ContentToolCall}) {
		t.Fatalf("kinds = %v", got)
	}
	if parts[0].Text != "Consider the weather." || signatureOf(t, parts[0]) != "sig-1" || parts[0].Extensions[0].Required {
		t.Fatalf("reasoning part = %#v", parts[0])
	}
	if parts[2].ToolCall.ID != "call_1" || string(parts[2].ToolCall.Args) != `{"city":"Paris"}` {
		t.Fatalf("tool call = %#v", parts[2].ToolCall)
	}
	if response.FinishReason != "tool_calls" || response.InputTokens != 5 || response.OutputTokens != 9 {
		t.Fatalf("metadata = %#v", response)
	}
}

func TestGenerateRejectsUnsupportedOrMalformedChunks(t *testing.T) {
	cases := map[string]string{
		"unknown top-level":       `[{"type":"reference","reference_ids":[1]}]`,
		"image output":            `[{"type":"image_url","image_url":"https://example.com/a.png"}]`,
		"nested reference":        `[{"type":"thinking","thinking":[{"type":"reference","reference_ids":[1]}]}]`,
		"nested tool reference":   `[{"type":"thinking","thinking":[{"type":"tool_reference","tool":"web_search","title":"x"}]}]`,
		"text chunk without text": `[{"type":"text"}]`,
		"number content":          `42`,
	}
	for name, content := range cases {
		t.Run(name, func(t *testing.T) {
			_, p := newChatServer(t, `{"choices":[{"message":{"content":`+content+`}}]}`)
			model, _ := p.TypedModel(MistralSmallLatest)
			_, err := model.Generate(t.Context(), userRequest("hi"))
			if err == nil {
				t.Fatal("malformed or unsupported content accepted")
			}
			if strings.Contains(name, "reference") || name == "image output" {
				if !errors.Is(err, ai.ErrUnsupportedCapability) {
					t.Fatalf("error = %v, want unsupported capability", err)
				}
			}
		})
	}
}

func TestStreamSeparatesReasoningThroughContentShapeTransitions(t *testing.T) {
	_, p := newChatServer(t, sse(
		`{"id":"req-2","model":"mistral-small-latest","choices":[{"delta":{"role":"assistant","content":""}}]}`,
		`{"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"Let me "}]}]}}]}`,
		`{"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"think."}]}]}}]}`,
		`{"choices":[{"delta":{"content":[{"type":"thinking","thinking":[],"signature":"sig-s"},{"type":"text","text":"The "}]}}]}`,
		`{"choices":[{"delta":{"content":"answer"}}]}`,
		`{"choices":[{"delta":{"content":" is 4."},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":7}}`,
	))
	model, _ := p.TypedModel(MistralSmallLatest)
	response, tokens, err := streamResponse(t, model, userRequest("2+2"))
	if err != nil {
		t.Fatal(err)
	}
	var thoughts, text strings.Builder
	sawText := false
	for _, token := range tokens {
		switch token.Type() {
		case ai.TokenTypeThought:
			if sawText {
				t.Fatal("reasoning token after answer text")
			}
			thoughts.WriteString(token.Text())
		case ai.TokenTypeText:
			sawText = true
			text.WriteString(token.Text())
		}
	}
	if thoughts.String() != "Let me think." || text.String() != "The answer is 4." {
		t.Fatalf("thoughts = %q, text = %q", thoughts.String(), text.String())
	}
	parts := response.Message.Parts
	if got := partKinds(parts); !reflect.DeepEqual(got, []ai.ContentKind{ai.ContentReasoning, ai.ContentText}) {
		t.Fatalf("kinds = %v", got)
	}
	if parts[0].Text != "Let me think." || signatureOf(t, parts[0]) != "sig-s" {
		t.Fatalf("reasoning = %#v", parts[0])
	}
	if response.FinishReason != "stop" || response.InputTokens != 3 || response.OutputTokens != 7 {
		t.Fatalf("completion = %#v", response)
	}
}

func TestStreamAttachesSignatureSentWithThinkingText(t *testing.T) {
	_, p := newChatServer(t, sse(
		`{"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"a"}]}]}}]}`,
		`{"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"b"}],"signature":"sig-ab"}]}}]}`,
		`{"choices":[{"delta":{"content":[{"type":"text","text":"done"}]},"finish_reason":"stop"}]}`,
	))
	model, _ := p.TypedModel(MistralSmallLatest)
	response, _, err := streamResponse(t, model, userRequest("x"))
	if err != nil {
		t.Fatal(err)
	}
	parts := response.Message.Parts
	if len(parts) != 2 || parts[0].Text != "ab" || signatureOf(t, parts[0]) != "sig-ab" || parts[1].Text != "done" {
		t.Fatalf("parts = %#v", parts)
	}
}

func TestGenerateAndStreamBuildSameSignedReasoningParts(t *testing.T) {
	_, p := newChatServer(t,
		`{"choices":[{"message":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"a"}]},{"type":"thinking","thinking":[{"type":"text","text":"b"}],"signature":"s"},{"type":"text","text":"done"}]}}]}`,
		sse(`{"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"a"}]}]}}]}`,
			`{"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"b"}],"signature":"s"}]}}]}`,
			`{"choices":[{"delta":{"content":[{"type":"text","text":"done"}]}}]}`),
	)
	model, _ := p.TypedModel(MistralSmallLatest)
	sync, err := model.Generate(t.Context(), userRequest("x"))
	if err != nil {
		t.Fatal(err)
	}
	streamed, _, err := streamResponse(t, model, userRequest("x"))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(sync.Message.Parts, streamed.Message.Parts) {
		t.Fatalf("sync = %#v\nstream = %#v", sync.Message.Parts, streamed.Message.Parts)
	}
	if parts := sync.Message.Parts; len(parts) != 2 || parts[0].Text != "ab" || signatureOf(t, parts[0]) != "s" {
		t.Fatalf("parts = %#v", parts)
	}
}

func TestStreamReasoningWithToolCallsAndCompletion(t *testing.T) {
	_, p := newChatServer(t, sse(
		`{"id":"req-3","choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"Need weather."}]}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_w","type":"function","function":{"name":"weather","arguments":"{\"city\":"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Oslo\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":2,"completion_tokens":4}}`,
	))
	model, _ := p.TypedModel(MistralSmallLatest)
	response, tokens, err := streamResponse(t, model, ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "weather")}, Tools: []ai.ToolDefinition{weatherTool()}})
	if err != nil {
		t.Fatal(err)
	}
	if got := partKinds(response.Message.Parts); !reflect.DeepEqual(got, []ai.ContentKind{ai.ContentReasoning, ai.ContentToolCall}) {
		t.Fatalf("kinds = %v", got)
	}
	call := response.Message.Parts[1].ToolCall
	if call.ID != "call_w" || string(call.Args) != `{"city":"Oslo"}` {
		t.Fatalf("call = %#v", call)
	}
	if last := tokens[len(tokens)-1]; last.Completion == nil || last.Completion.FinishReason != "tool_calls" || last.Completion.RequestID != "req-3" {
		t.Fatalf("last token = %#v", last)
	}
}

func TestStreamRejectsUnsupportedChunks(t *testing.T) {
	for name, delta := range map[string]string{
		"unknown":          `[{"type":"reference","reference_ids":[1]}]`,
		"nested reference": `[{"type":"thinking","thinking":[{"type":"reference","reference_ids":[1]}]}]`,
		"object content":   `{"type":"text","text":"x"}`,
	} {
		t.Run(name, func(t *testing.T) {
			_, p := newChatServer(t, sse(
				`{"id":"req-4","choices":[{"delta":{"content":"before"}}]}`,
				`{"choices":[{"delta":{"content":`+delta+`}}]}`,
				`{"choices":[{"delta":{"content":"after"}}]}`,
			))
			model, _ := p.TypedModel(MistralSmallLatest)
			response, _, err := streamResponse(t, model, userRequest("x"))
			if err == nil {
				t.Fatal("unsupported stream chunk accepted")
			}
			if name != "object content" && !errors.Is(err, ai.ErrUnsupportedCapability) {
				t.Fatalf("error = %v", err)
			}
			if strings.Contains(response.Text(), "after") {
				t.Fatalf("stream continued after error: %q", response.Text())
			}
		})
	}
}

func weatherTool() ai.ToolDefinition {
	return ai.ToolDefinition{Type: "function", Name: "weather", Description: "Gets weather.", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)}
}

func TestReasoningToolCallContinuationReplaysState(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		name := "sync"
		first := thinkingResponse
		if streaming {
			name = "stream"
			first = sse(
				`{"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"Consider the weather."}],"signature":"sig-1"}]}}]}`,
				`{"choices":[{"delta":{"content":[{"type":"text","text":"Checking now."}]}}]}`,
				`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":"tool_calls"}]}`,
			)
		}
		t.Run(name, func(t *testing.T) {
			cs, p := newChatServer(t, first, `{"choices":[{"message":{"content":"Sunny."},"finish_reason":"stop"}]}`)
			model, _ := p.TypedModel(MistralSmallLatest)
			req := ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "weather?")}, Tools: []ai.ToolDefinition{weatherTool()}, Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortHigh}}
			var assistant ai.Message
			if streaming {
				response, _, err := streamResponse(t, model, req)
				if err != nil {
					t.Fatal(err)
				}
				assistant = response.Message
			} else {
				response, err := model.Generate(t.Context(), req)
				if err != nil {
					t.Fatal(err)
				}
				assistant = response.Message
			}
			calls := assistant.ToolCalls()
			if len(calls) != 1 {
				t.Fatalf("calls = %#v", calls)
			}
			req.Messages = append(req.Messages, assistant, ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{
				ToolCallID: calls[0].ID, Name: calls[0].Name, Parts: ai.TextParts(`{"sky":"clear"}`),
			}}}})
			if _, err := model.Generate(t.Context(), req); err != nil {
				t.Fatal(err)
			}
			second := cs.request(t, 1)
			messages := second["messages"].([]any)
			if len(messages) != 3 {
				t.Fatalf("messages = %#v", messages)
			}
			wantAssistant := map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "thinking", "thinking": []any{map[string]any{"type": "text", "text": "Consider the weather."}}, "signature": "sig-1"},
					map[string]any{"type": "text", "text": "Checking now."},
				},
				"tool_calls": []any{map[string]any{"id": "call_1", "type": "function", "function": map[string]any{"name": "weather", "arguments": `{"city":"Paris"}`}}},
			}
			if !reflect.DeepEqual(messages[1], wantAssistant) {
				got, _ := json.Marshal(messages[1])
				t.Fatalf("assistant replay = %s", got)
			}
			wantTool := map[string]any{"role": "tool", "content": `{"sky":"clear"}`, "tool_call_id": "call_1"}
			if !reflect.DeepEqual(messages[2], wantTool) {
				t.Fatalf("tool message = %#v", messages[2])
			}
			if second["reasoning_effort"] != "high" {
				t.Fatalf("reasoning_effort = %#v", second["reasoning_effort"])
			}
		})
	}
}

func TestReplayRejectsContentThatCannotKeepPosition(t *testing.T) {
	call := ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "c", Type: "function", Name: "weather", Args: json.RawMessage(`{}`)}}
	for name, later := range map[string]ai.ContentPart{
		"text":      {Kind: ai.ContentText, Text: "after"},
		"reasoning": {Kind: ai.ContentReasoning, Text: "after"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := mapNativeMessages([]ai.Message{{Role: ai.RoleAssistant, Parts: []ai.ContentPart{call, later}}})
			if !errors.Is(err, ai.ErrUnsupportedCapability) {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestReplayIgnoresForeignOptionalStateAndRejectsInvalidSignature(t *testing.T) {
	foreign := ai.ContentPart{Kind: ai.ContentReasoning, Text: "t", Extensions: []ai.Extension{{Namespace: "anthropic", Type: "signature", Data: json.RawMessage(`"x"`)}}}
	messages, err := mapNativeMessages([]ai.Message{{Role: ai.RoleAssistant, Parts: []ai.ContentPart{foreign}}})
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(messages[0].Content)
	if string(encoded) != `[{"type":"thinking","thinking":[{"type":"text","text":"t"}]}]` {
		t.Fatalf("content = %s", encoded)
	}
	bad := ai.ContentPart{Kind: ai.ContentReasoning, Text: "t", Extensions: []ai.Extension{{Namespace: ExtensionNamespace, Type: ExtensionThinkingSignature, Data: json.RawMessage(`42`)}}}
	if _, err := mapNativeMessages([]ai.Message{{Role: ai.RoleAssistant, Parts: []ai.ContentPart{bad}}}); err == nil {
		t.Fatal("non-string signature accepted")
	}
}

func TestReasoningConfigMapping(t *testing.T) {
	high, none, low, xhigh := "high", "none", "low", "xhigh"
	cases := []struct {
		name   string
		model  string
		config ai.ReasoningConfig
		want   *string
		err    bool
	}{
		{name: "omitted", model: MistralSmallLatest},
		{name: "explicit none", model: MistralSmallLatest, config: ai.ReasoningConfig{Effort: ai.ReasoningEffortNone}, want: &none},
		{name: "high", model: MistralMedium35, config: ai.ReasoningConfig{Effort: ai.ReasoningEffortHigh}, want: &high},
		{name: "enabled defaults to high", model: MistralSmallLatest, config: ai.ReasoningConfig{Enabled: true}, want: &high},
		{name: "include thoughts defaults to high", model: MistralSmallLatest, config: ai.ReasoningConfig{IncludeThoughts: true}, want: &high},
		{name: "undocumented effort rejected for known model", model: MistralSmallLatest, config: ai.ReasoningConfig{Effort: ai.ReasoningEffortLow}, err: true},
		{name: "unknown model passes effort through", model: "mistral-future", config: ai.ReasoningConfig{Effort: ai.ReasoningEffortLow}, want: &low},
		{name: "xhigh on unknown model", model: "mistral-future", config: ai.ReasoningConfig{Effort: ai.ReasoningEffortXHigh}, want: &xhigh},
		{name: "max is not a Mistral effort", model: "mistral-future", config: ai.ReasoningConfig{Effort: ai.ReasoningEffortMax}, err: true},
		{name: "budget unsupported", model: MistralSmallLatest, config: ai.ReasoningConfig{Enabled: true, BudgetTokens: 1024}, err: true},
		{name: "none with enabled", model: MistralSmallLatest, config: ai.ReasoningConfig{Enabled: true, Effort: ai.ReasoningEffortNone}, err: true},
		{name: "none with thoughts", model: MistralSmallLatest, config: ai.ReasoningConfig{IncludeThoughts: true, Effort: ai.ReasoningEffortNone}, err: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cs, p := newChatServer(t, `{"choices":[{"message":{"content":"ok"}}]}`)
			model, _ := p.TypedModel(tc.model)
			req := userRequest("hi")
			req.Reasoning = tc.config
			_, err := model.Generate(t.Context(), req)
			if tc.err {
				if !errors.Is(err, ai.ErrUnsupportedCapability) || len(cs.requests()) != 0 {
					t.Fatalf("error = %v, requests = %d", err, len(cs.requests()))
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			got, present := cs.request(t, 0)["reasoning_effort"]
			if tc.want == nil {
				if present {
					t.Fatalf("reasoning_effort sent: %#v", got)
				}
				return
			}
			if got != *tc.want {
				t.Fatalf("reasoning_effort = %#v, want %q", got, *tc.want)
			}
		})
	}
}

func TestDescriptorsReflectMappedReasoningSupport(t *testing.T) {
	p := New("test-key", nil)
	small, _ := p.TypedModel(MistralSmallLatest)
	d := small.Descriptor()
	if d.Reasoning != ai.FeatureSupportSupported || d.ReasoningEffort != ai.FeatureSupportSupported || !reflect.DeepEqual(d.ReasoningEfforts, adjustableReasoningEfforts) {
		t.Fatalf("small descriptor = %#v", d)
	}
	unknown, _ := p.TypedModel(CodestralLatest)
	if d := unknown.Descriptor(); d.Reasoning != ai.FeatureSupportUnknown || d.ReasoningEffort != ai.FeatureSupportUnknown || len(d.ReasoningEfforts) != 0 {
		t.Fatalf("codestral descriptor = %#v", d)
	}

	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(*http.Request) (*http.Response, error) {
		return response(http.StatusOK, `{"data":[`+
			`{"id":"magistral-medium-latest","capabilities":{"completion_chat":true,"reasoning":true}},`+
			`{"id":"mistral-small-latest","capabilities":{"completion_chat":true,"reasoning":false}},`+
			`{"id":"plain","capabilities":{"completion_chat":true,"reasoning":false}}]}`), nil
	})}
	descriptors, err := p.ListModelDescriptors(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]ai.ModelDescriptor{}
	for _, descriptor := range descriptors {
		byName[descriptor.Model] = descriptor
	}
	if d := byName["magistral-medium-latest"]; d.Reasoning != ai.FeatureSupportSupported || d.ReasoningEffort != ai.FeatureSupportUnknown {
		t.Fatalf("catalog reasoning model = %#v", d)
	}
	if d := byName["mistral-small-latest"]; d.Reasoning != ai.FeatureSupportSupported || !reflect.DeepEqual(d.ReasoningEfforts, adjustableReasoningEfforts) {
		t.Fatalf("catalog false must not downgrade adjustable model: %#v", d)
	}
	if d := byName["plain"]; d.Reasoning != ai.FeatureSupportUnknown {
		t.Fatalf("plain = %#v", d)
	}
}

func TestImageInputMapsURLAndInlineBytesInOrder(t *testing.T) {
	cs, p := newChatServer(t, `{"choices":[{"message":{"content":"a cat"}}]}`)
	model, _ := p.TypedModel("pixtral-large-latest")
	_, err := model.Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{{Role: ai.RoleUser, Parts: []ai.ContentPart{
		{Kind: ai.ContentText, Text: "Compare "},
		{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", URI: "https://example.com/cat.png"}},
		{Kind: ai.ContentText, Text: " with "},
		{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/JPG", Data: []byte{0xff, 0xd8, 0xff}}},
	}}}})
	if err != nil {
		t.Fatal(err)
	}
	want := []any{
		map[string]any{"type": "text", "text": "Compare "},
		map[string]any{"type": "image_url", "image_url": "https://example.com/cat.png"},
		map[string]any{"type": "text", "text": " with "},
		map[string]any{"type": "image_url", "image_url": "data:image/jpeg;base64,/9j/"},
	}
	got := cs.request(t, 0)["messages"].([]any)[0].(map[string]any)["content"]
	if !reflect.DeepEqual(got, want) {
		encoded, _ := json.Marshal(got)
		t.Fatalf("content = %s", encoded)
	}
}

func TestImageInputRejectsUnsupportedCombinations(t *testing.T) {
	image := func(media ai.MediaPart) ai.ContentPart { return ai.ContentPart{Kind: ai.ContentMedia, Media: &media} }
	cases := map[string]ai.Message{
		"non-image MIME":    {Role: ai.RoleUser, Parts: []ai.ContentPart{image(ai.MediaPart{MIMEType: "application/pdf", URI: "https://example.com/a.pdf"})}},
		"unsupported image": {Role: ai.RoleUser, Parts: []ai.ContentPart{image(ai.MediaPart{MIMEType: "image/tiff", Data: []byte{1}})}},
		"file scheme":       {Role: ai.RoleUser, Parts: []ai.ContentPart{image(ai.MediaPart{MIMEType: "image/png", URI: "file:///tmp/a.png"})}},
		"cloud scheme":      {Role: ai.RoleUser, Parts: []ai.ContentPart{image(ai.MediaPart{MIMEType: "image/png", URI: "gs://bucket/a.png"})}},
		"mismatched data":   {Role: ai.RoleUser, Parts: []ai.ContentPart{image(ai.MediaPart{MIMEType: "image/png", URI: "data:image/gif;base64,R0lG"})}},
		"system image":      {Role: ai.RoleSystem, Parts: []ai.ContentPart{image(ai.MediaPart{MIMEType: "image/png", URI: "https://example.com/a.png"})}},
		"assistant image":   {Role: ai.RoleAssistant, Parts: []ai.ContentPart{image(ai.MediaPart{MIMEType: "image/png", URI: "https://example.com/a.png"})}},
	}
	for name, message := range cases {
		t.Run(name, func(t *testing.T) {
			cs, p := newChatServer(t)
			model, _ := p.TypedModel(MistralSmallLatest)
			_, err := model.Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{message}})
			var contentErr *ai.UnsupportedContentError
			if !errors.As(err, &contentErr) || !errors.Is(err, ai.ErrUnsupportedCapability) || len(cs.requests()) != 0 {
				t.Fatalf("error = %v, requests = %d", err, len(cs.requests()))
			}
		})
	}
}

func TestImageInputUsesDiscoveredVisionSupport(t *testing.T) {
	cs, p := newChatServer(t, `{"choices":[{"message":{"content":"ok"}}]}`, sse(`{"choices":[{"delta":{"content":"ok"}}]}`))
	chat := p.httpClient.Transport
	if chat == nil {
		chat = http.DefaultTransport
	}
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path == "/v1/models" {
			return response(http.StatusOK, `{"data":[{"id":"text-only","capabilities":{"completion_chat":true,"vision":false}},{"id":"seeing","capabilities":{"completion_chat":true,"vision":true}}]}`), nil
		}
		return chat.RoundTrip(r)
	})}
	if _, err := p.ListModelDescriptors(t.Context()); err != nil {
		t.Fatal(err)
	}
	req := ai.AIRequest{Messages: []ai.Message{{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", URI: "https://example.com/a.png"}}}}}}

	textOnly, _ := p.TypedModel("text-only")
	if _, err := textOnly.Generate(t.Context(), req); !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("generate error = %v", err)
	}
	if _, _, err := streamResponse(t, textOnly, req); !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("stream error = %v", err)
	}
	if len(cs.requests()) != 0 {
		t.Fatalf("rejected image requests reached provider: %d", len(cs.requests()))
	}
	seeing, _ := p.TypedModel("seeing")
	if _, err := seeing.Generate(t.Context(), req); err != nil {
		t.Fatal(err)
	}
	if _, _, err := streamResponse(t, seeing, req); err != nil {
		t.Fatal(err)
	}
	if len(cs.requests()) != 2 {
		t.Fatalf("requests = %d", len(cs.requests()))
	}
}

func TestTextOnlyRequestContentStaysString(t *testing.T) {
	cs, p := newChatServer(t, `{"choices":[{"message":{"content":"ok"}}]}`)
	model, _ := p.TypedModel(MistralSmallLatest)
	if _, err := model.Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{
		ai.TextMessage(ai.RoleSystem, "rules"),
		{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "a"}, {Kind: ai.ContentJSON, JSON: json.RawMessage(`{"b":1}`)}}},
	}}); err != nil {
		t.Fatal(err)
	}
	messages := cs.request(t, 0)["messages"].([]any)
	if messages[0].(map[string]any)["content"] != "rules" || messages[1].(map[string]any)["content"] != `a{"b":1}` {
		t.Fatalf("messages = %#v", messages)
	}
}

func TestTypedNativeOptionsPreserveExplicitZeroValues(t *testing.T) {
	zero, falseValue, seed, key := 0.0, false, 0, "tenant-42"
	cs, p := newChatServer(t, `{"choices":[{"message":{"content":"ok"}}]}`, sse(`{"choices":[{"delta":{"content":"ok"}}]}`), `{"choices":[{"message":{"content":"ok"}}]}`)
	model, err := p.TypedModel(MistralSmallLatest, WithChatCompletionOptions(ChatCompletionOptions{
		Temperature: &zero, TopP: &zero, RandomSeed: &seed, SafePrompt: &falseValue,
		Stop: []string{"END"}, PresencePenalty: &zero, FrequencyPenalty: &zero,
		ParallelToolCalls: &falseValue, Prediction: &Prediction{Content: "draft"}, PromptCacheKey: &key,
	}))
	if err != nil {
		t.Fatal(err)
	}
	req := ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}, Tools: []ai.ToolDefinition{weatherTool()}, MaxTokens: 0}
	if _, err := model.Generate(t.Context(), req); err != nil {
		t.Fatal(err)
	}
	if _, _, err := streamResponse(t, model, req); err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"temperature": 0.0, "top_p": 0.0, "random_seed": 0.0, "safe_prompt": false,
		"stop": []any{"END"}, "presence_penalty": 0.0, "frequency_penalty": 0.0,
		"parallel_tool_calls": false, "prediction": map[string]any{"type": "content", "content": "draft"}, "prompt_cache_key": "tenant-42",
	}
	for i := 0; i < 2; i++ {
		body := cs.request(t, i)
		for field, value := range want {
			if !reflect.DeepEqual(body[field], value) {
				t.Fatalf("request %d %s = %#v, want %#v", i, field, body[field], value)
			}
		}
	}

	plain, _ := p.TypedModel(MistralSmallLatest)
	if _, err := plain.Generate(t.Context(), userRequest("x")); err != nil {
		t.Fatal(err)
	}
	body := cs.request(t, 2)
	for field := range want {
		if _, present := body[field]; present {
			t.Fatalf("omitted option %s was sent", field)
		}
	}
}

func TestTypedNativeOptionsValidateBeforeTransport(t *testing.T) {
	negative, overOne, nan, blank := -0.1, 1.5, math.NaN(), " "
	seed := -1
	cases := map[string]ChatCompletionOptions{
		"negative temperature": {Temperature: &negative},
		"top_p above one":      {TopP: &overOne},
		"nan penalty":          {PresencePenalty: &nan},
		"negative seed":        {RandomSeed: &seed},
		"empty stop":           {Stop: []string{"ok", ""}},
		"empty prediction":     {Prediction: &Prediction{}},
		"blank cache key":      {PromptCacheKey: &blank},
	}
	for name, options := range cases {
		t.Run(name, func(t *testing.T) {
			cs, p := newChatServer(t)
			model, _ := p.TypedModel(MistralSmallLatest, WithChatCompletionOptions(options))
			if _, err := model.Generate(t.Context(), userRequest("x")); !errors.Is(err, ErrInvalidChatCompletionOptions) {
				t.Fatalf("generate error = %v", err)
			}
			if _, _, err := streamResponse(t, model, userRequest("x")); !errors.Is(err, ErrInvalidChatCompletionOptions) {
				t.Fatalf("stream error = %v", err)
			}
			if len(cs.requests()) != 0 {
				t.Fatalf("invalid options reached provider")
			}
		})
	}
}

func TestModelWithDerivesIndependentOptions(t *testing.T) {
	temperature := 0.2
	stop := []string{"A"}
	base, _ := New("test-key", nil).TypedModel(MistralSmallLatest, WithChatCompletionOptions(ChatCompletionOptions{Temperature: &temperature, Stop: stop}))
	temperature, stop[0] = 0.9, "mutated"
	if *base.chatOptions.Temperature != 0.2 || base.chatOptions.Stop[0] != "A" {
		t.Fatalf("construction did not snapshot options: %#v", base.chatOptions)
	}
	key := "per-call"
	derived := base.With(WithChatCompletionOptions(ChatCompletionOptions{PromptCacheKey: &key}))
	if base.chatOptions.PromptCacheKey != nil {
		t.Fatal("With modified the base model")
	}
	if derived.chatOptions.PromptCacheKey == nil || *derived.chatOptions.PromptCacheKey != "per-call" || derived.Name() != base.Name() {
		t.Fatalf("derived = %#v", derived.chatOptions)
	}
	copied := base.With()
	*copied.chatOptions.Temperature = 1
	if *base.chatOptions.Temperature != 0.2 {
		t.Fatal("With shared option pointers")
	}
}
