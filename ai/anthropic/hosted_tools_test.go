package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/ai"
	antropic "github.com/anthropics/anthropic-sdk-go"
)

const (
	serverToolUseBlock = `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"go 1.26 release"}}`
	searchResultBlock  = `{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://go.dev/doc/go1.26","title":"Go 1.26 Release Notes","encrypted_content":"opaque-result","page_age":null}]}`
	searchCitation     = `{"type":"web_search_result_location","url":"https://go.dev/doc/go1.26","title":"Go 1.26 Release Notes","encrypted_index":"opaque-index","cited_text":"Go 1.26 is released."}`
)

func assertSearchAndCitation(t *testing.T, message ai.Message) {
	t.Helper()
	calls := message.HostedToolCalls()
	if len(calls) != 1 || calls[0].Provider != "anthropic" || calls[0].ID != "srvtoolu_1" || calls[0].Kind != ai.HostedToolWebSearch || calls[0].Name != "web_search" || calls[0].Query != "go 1.26 release" {
		t.Fatalf("hosted tool calls = %#v", calls)
	}
	citations := message.Citations()
	if len(citations) != 1 || citations[0].URL != "https://go.dev/doc/go1.26" || citations[0].Title != "Go 1.26 Release Notes" || citations[0].Source != "Go 1.26 is released." || citations[0].Text != "Go 1.26 is out." {
		t.Fatalf("citations = %#v", citations)
	}
	if err := message.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestGenerateKeepsServerToolBlocksAndCitations(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1},"content":[` +
			serverToolUseBlock + `,` + searchResultBlock + `,{"type":"text","text":"Go 1.26 is out.","citations":[` + searchCitation + `]}]}`))
	})
	response, err := m.Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")}})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	parts := response.Message.Parts
	if len(parts) != 3 || parts[0].Kind != ai.ContentExtension || parts[1].Extensions[0].Type != hostedToolResultExtension || parts[2].Text != "Go 1.26 is out." {
		t.Fatalf("parts = %#v", parts)
	}
	assertSearchAndCitation(t, response.Message)
	c := parts[2].Citations()[0]
	if c.Start != 0 || c.End != len("Go 1.26 is out.") {
		t.Fatalf("citation range = %d..%d", c.Start, c.End)
	}
}

func sse(event, data string) string { return "event: " + event + "\ndata: " + data + "\n\n" }

func TestGenerateStreamKeepsServerToolBlocksAndCitations(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		var b strings.Builder
		b.WriteString(sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}`))
		b.WriteString(sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"go 1.26 release\"}"}}`))
		b.WriteString(sse("content_block_stop", `{"type":"content_block_stop","index":0}`))
		b.WriteString(sse("content_block_start", `{"type":"content_block_start","index":1,"content_block":`+searchResultBlock+`}`))
		b.WriteString(sse("content_block_stop", `{"type":"content_block_stop","index":1}`))
		// An uncited block merges with the cited one in the caller's text part.
		b.WriteString(sse("content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`))
		b.WriteString(sse("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Ja: "}}`))
		b.WriteString(sse("content_block_stop", `{"type":"content_block_stop","index":2}`))
		b.WriteString(sse("content_block_start", `{"type":"content_block_start","index":3,"content_block":{"type":"text","text":"","citations":[]}}`))
		b.WriteString(sse("content_block_delta", `{"type":"content_block_delta","index":3,"delta":{"type":"citations_delta","citation":`+searchCitation+`}}`))
		b.WriteString(sse("content_block_delta", `{"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"Go 1.26 "}}`))
		b.WriteString(sse("content_block_delta", `{"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"is out."}}`))
		b.WriteString(sse("content_block_stop", `{"type":"content_block_stop","index":3}`))
		b.WriteString(sse("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`))
		b.WriteString(sse("message_stop", `{"type":"message_stop"}`))
		_, _ = w.Write([]byte(b.String()))
	})
	message := ai.Message{Role: ai.RoleAssistant}
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")}}) {
		if token.Err != nil {
			t.Fatalf("stream error: %v", token.Err)
		}
		message.AppendToken(token)
	}
	if len(message.Parts) != 3 || message.Parts[2].Text != "Ja: Go 1.26 is out." {
		t.Fatalf("parts = %#v", message.Parts)
	}
	assertSearchAndCitation(t, message)
	c := message.Parts[2].Citations()[0]
	if got := string([]rune(message.Parts[2].Text)[c.Start:c.End]); got != "Go 1.26 is out." {
		t.Fatalf("citation covers %q", got)
	}
	var call struct {
		Input map[string]string `json:"input"`
	}
	if err := json.Unmarshal(message.HostedToolCalls()[0].Raw, &call); err != nil || call.Input["query"] != "go 1.26 release" {
		t.Fatalf("streamed server_tool_use raw = %s", message.HostedToolCalls()[0].Raw)
	}
}

func TestServerToolBlocksReplayInOrder(t *testing.T) {
	var body map[string]any
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		body = decodeRequest(t, r)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_2","type":"message","role":"assistant","model":"claude-test","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1},"content":[{"type":"text","text":"done"}]}`))
	})
	first, err := mapCanonicalMessageContent(nil)
	if err != nil {
		t.Fatal(err)
	}
	first.Parts = []ai.ContentPart{
		serverToolUsePart(json.RawMessage(serverToolUseBlock)),
		serverToolResultPart(json.RawMessage(searchResultBlock)),
		{Kind: ai.ContentText, Text: "Go 1.26 is out.", Extensions: ai.CitationsExtension(extensionNamespace, textCitations([]json.RawMessage{json.RawMessage(searchCitation)}, "Go 1.26 is out."))},
	}
	stored, _ := json.Marshal(first)
	var restored ai.Message
	if err := json.Unmarshal(stored, &restored); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news"), restored, ai.TextMessage(ai.RoleUser, "more")}}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	messages := body["messages"].([]any)
	content := object(t, messages[1])["content"].([]any)
	var types []string
	for _, block := range content {
		types = append(types, object(t, block)["type"].(string))
	}
	if strings.Join(types, ",") != "server_tool_use,web_search_tool_result,text" {
		t.Fatalf("assistant block types = %v", types)
	}
	result := object(t, content[1])
	if result["tool_use_id"] != "srvtoolu_1" || !strings.Contains(mustJSON(t, result), "opaque-result") {
		t.Fatalf("replayed result = %#v", result)
	}
}

func TestServerToolBlocksAreOptionalForOtherProviders(t *testing.T) {
	message := ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{
		serverToolUsePart(json.RawMessage(serverToolUseBlock)),
		ai.HostedToolCallPart("openai", ai.HostedToolCall{Kind: ai.HostedToolWebSearch, Raw: json.RawMessage(`{"type":"web_search_call","id":"ws_1"}`)}),
		{Kind: ai.ContentText, Text: "ok"},
	}}
	out, err := mapNativeMessages([]ai.Message{ai.TextMessage(ai.RoleUser, "x"), message}, false)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(out[1].Content); got != 2 {
		t.Fatalf("assistant blocks = %d, want the Anthropic call and text only", got)
	}
}

func mustJSON(t *testing.T, value any) string {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func TestGenerateContinuesPausedTurn(t *testing.T) {
	var bodies []map[string]any
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		bodies = append(bodies, decodeRequest(t, r))
		w.Header().Set("Content-Type", "application/json")
		if len(bodies) == 1 {
			_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","stop_reason":"pause_turn","usage":{"input_tokens":10,"output_tokens":3},"content":[` + serverToolUseBlock + `,` + searchResultBlock + `]}`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"msg_2","type":"message","role":"assistant","model":"claude-test","stop_reason":"end_turn","usage":{"input_tokens":20,"output_tokens":5},"content":[{"type":"text","text":"Go 1.26 is out.","citations":[` + searchCitation + `]}]}`))
	})
	response, err := m.Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")}})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	assertSearchAndCitation(t, response.Message)
	if response.FinishReason != "end_turn" || response.InputTokens != 30 || response.OutputTokens != 8 {
		t.Fatalf("response = %+v", response)
	}
	assertPausedTurnSentBack(t, bodies)
}

func assertPausedTurnSentBack(t *testing.T, bodies []map[string]any) {
	t.Helper()
	if len(bodies) != 2 {
		t.Fatalf("requests = %d, want 2", len(bodies))
	}
	messages := bodies[1]["messages"].([]any)
	if len(messages) != 2 || object(t, messages[1])["role"] != "assistant" {
		t.Fatalf("continuation messages = %#v", messages)
	}
	var types []string
	for _, block := range object(t, messages[1])["content"].([]any) {
		types = append(types, object(t, block)["type"].(string))
	}
	if strings.Join(types, ",") != "server_tool_use,web_search_tool_result" {
		t.Fatalf("continued turn blocks = %v", types)
	}
	blocks := object(t, messages[1])["content"].([]any)
	if input := object(t, object(t, blocks[0])["input"]); input["query"] != "go 1.26 release" {
		t.Fatalf("continued server_tool_use input = %#v", input)
	}
	if !strings.Contains(mustJSON(t, blocks[1]), "opaque-result") {
		t.Fatalf("continued result lost its encrypted content: %s", mustJSON(t, blocks[1]))
	}
}

func TestGenerateStreamContinuesPausedTurn(t *testing.T) {
	var bodies []map[string]any
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		bodies = append(bodies, decodeRequest(t, r))
		w.Header().Set("Content-Type", "text/event-stream")
		var b strings.Builder
		if len(bodies) == 1 {
			b.WriteString(sse("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}`))
			b.WriteString(sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}`))
			b.WriteString(sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"go 1.26 release\"}"}}`))
			b.WriteString(sse("content_block_stop", `{"type":"content_block_stop","index":0}`))
			b.WriteString(sse("content_block_start", `{"type":"content_block_start","index":1,"content_block":`+searchResultBlock+`}`))
			b.WriteString(sse("content_block_stop", `{"type":"content_block_stop","index":1}`))
			b.WriteString(sse("message_delta", `{"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":3}}`))
		} else {
			b.WriteString(sse("message_start", `{"type":"message_start","message":{"id":"msg_2","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":20,"output_tokens":0}}}`))
			b.WriteString(sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"","citations":[]}}`))
			b.WriteString(sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":`+searchCitation+`}}`))
			b.WriteString(sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Go 1.26 is out."}}`))
			b.WriteString(sse("content_block_stop", `{"type":"content_block_stop","index":0}`))
			b.WriteString(sse("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`))
		}
		b.WriteString(sse("message_stop", `{"type":"message_stop"}`))
		_, _ = w.Write([]byte(b.String()))
	})
	message := ai.Message{Role: ai.RoleAssistant}
	var completions []ai.Completion
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")}}) {
		if token.Err != nil {
			t.Fatalf("stream error: %v", token.Err)
		}
		if token.Completion != nil {
			completions = append(completions, *token.Completion)
		}
		message.AppendToken(token)
	}
	assertSearchAndCitation(t, message)
	if len(completions) != 1 || completions[0].FinishReason != "end_turn" || completions[0].RequestID != "msg_2" || completions[0].Usage.InputTokens != 30 || completions[0].Usage.OutputTokens != 8 || completions[0].InitialInputTokens != 10 {
		t.Fatalf("completions = %+v", completions)
	}
	assertPausedTurnSentBack(t, bodies)
}

const pausedStream = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":` + serverToolUseBlock + `}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":` + searchResultBlock + `}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":3}}

event: message_stop
data: {"type":"message_stop"}

`

func failSecondRequest(t *testing.T, first func(w http.ResponseWriter)) *Model {
	requests := 0
	return testModel(t, func(w http.ResponseWriter, _ *http.Request) {
		requests++
		if requests == 1 {
			first(w)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"boom"}}`))
	})
}

// A continuation that fails still reports the usage of the paused request,
// which was billed.
func TestGenerateStreamReportsPausedUsageWhenContinuationFails(t *testing.T) {
	m := failSecondRequest(t, func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(pausedStream))
	})
	var completions []ai.Completion
	var streamErr error
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")}}) {
		if token.Completion != nil {
			if streamErr != nil {
				t.Fatal("completion after the error")
			}
			completions = append(completions, *token.Completion)
		}
		if token.Err != nil {
			streamErr = token.Err
		}
	}
	if streamErr == nil {
		t.Fatal("missing continuation error")
	}
	if len(completions) != 1 || !completions[0].UsageReported || completions[0].Usage.InputTokens != 10 || completions[0].Usage.OutputTokens != 3 || completions[0].InitialInputTokens != 10 || completions[0].FinishReason != "" {
		t.Fatalf("completions = %+v", completions)
	}
}

// A continuation that fails after its message_start still reports the usage
// that message_start carried, output tokens included.
func TestGenerateStreamReportsContinuationStartUsageWhenItFails(t *testing.T) {
	requests := 0
	m := testModel(t, func(w http.ResponseWriter, _ *http.Request) {
		requests++
		w.Header().Set("Content-Type", "text/event-stream")
		if requests == 1 {
			_, _ = w.Write([]byte(pausedStream))
			return
		}
		_, _ = w.Write([]byte(sse("message_start", `{"type":"message_start","message":{"id":"msg_2","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":20,"output_tokens":1}}}`) +
			sse("content_block_delta", `{"type":"content_block_delta","index":7,"delta":{"type":"text_delta","text":"x"}}`)))
	})
	var completions []ai.Completion
	var streamErr error
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")}}) {
		if token.Completion != nil {
			completions = append(completions, *token.Completion)
		}
		if token.Err != nil {
			streamErr = token.Err
		}
	}
	if streamErr == nil {
		t.Fatal("missing continuation error")
	}
	if len(completions) != 1 || completions[0].Usage.InputTokens != 30 || completions[0].Usage.OutputTokens != 4 || completions[0].InitialInputTokens != 10 {
		t.Fatalf("completions = %+v", completions)
	}
}

func TestGenerateRecordsPausedUsageWhenContinuationFails(t *testing.T) {
	var finished *gai.Observation
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests++
		w.Header().Set("Content-Type", "application/json")
		if requests == 1 {
			_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","stop_reason":"pause_turn","usage":{"input_tokens":10,"output_tokens":3},"content":[` + serverToolUseBlock + `,` + searchResultBlock + `]}`))
			return
		}
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"boom"}}`))
	}))
	t.Cleanup(server.Close)
	provider := New("test-key", gai.ObservationSinkFunc(func(_ context.Context, event gai.Observation) {
		if event.Name == "generation_finished" {
			finished = &event
		}
	}))
	provider.baseURL = server.URL
	model, err := provider.Model(ClaudeSonnet4_6)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.(*Model).Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")}}); err == nil {
		t.Fatal("missing continuation error")
	}
	if finished == nil || finished.Fields["input_tokens"] != 10 || finished.Fields["output_tokens"] != 3 {
		t.Fatalf("generation_finished = %+v", finished)
	}
}

func TestGenerateStopsContinuingPausedTurnAtLimit(t *testing.T) {
	requests := 0
	m := testModel(t, func(w http.ResponseWriter, _ *http.Request) {
		requests++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg","type":"message","role":"assistant","model":"claude-test","stop_reason":"pause_turn","usage":{"input_tokens":1,"output_tokens":1},"content":[` + serverToolUseBlock + `,` + searchResultBlock + `]}`))
	})
	_, err := m.Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")}})
	var terminal *ai.TerminalError
	if !errors.As(err, &terminal) || terminal.Reason != "pause_turn" || requests != maxPauseContinuations+1 {
		t.Fatalf("error = %v, requests = %d", err, requests)
	}
}

func responseBlocks(t *testing.T, raw string) []antropic.ContentBlockUnion {
	t.Helper()
	var message antropic.Message
	if err := json.Unmarshal([]byte(`{"id":"m","type":"message","role":"assistant","content":[`+raw+`]}`), &message); err != nil {
		t.Fatal(err)
	}
	return message.Content
}

func blockTypes(t *testing.T, blocks []antropic.ContentBlockParamUnion) string {
	t.Helper()
	var types []string
	for _, block := range blocks {
		var head struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(mustJSON(t, block)), &head); err != nil {
			t.Fatal(err)
		}
		types = append(types, head.Type)
	}
	return strings.Join(types, ",")
}

func TestContinuePausedTurnExtendsTrailingAssistant(t *testing.T) {
	messages := []antropic.MessageParam{antropic.NewUserMessage(antropic.NewTextBlock("q")), antropic.NewAssistantMessage(antropic.NewTextBlock("a"))}
	out := continuePausedTurn(messages, responseBlocks(t, `{"type":"text","text":"b"}`))
	if len(out) != 2 || len(out[1].Content) != 2 || len(messages[1].Content) != 1 {
		t.Fatalf("out = %#v, source = %#v", out, messages)
	}
}

// A paused turn that switched models with a fallback block must not send the
// declined attempt's state back.
func TestContinuePausedTurnDropsDeclinedAttempt(t *testing.T) {
	content := responseBlocks(t, `{"type":"thinking","thinking":"t","signature":"s"},`+serverToolUseBlock+`,`+searchResultBlock+`,{"type":"text","text":"kept"},{"type":"fallback","from":{"model":"a"},"to":{"model":"b"}},`+serverToolUseBlock)
	out := continuePausedTurn([]antropic.MessageParam{antropic.NewUserMessage(antropic.NewTextBlock("q"))}, content)
	if got := blockTypes(t, out[1].Content); got != "text,fallback,server_tool_use" {
		t.Fatalf("continued blocks = %s", got)
	}
}

func TestUnreplayableHostedItemsAreRejected(t *testing.T) {
	call := serverToolUsePart(json.RawMessage(serverToolUseBlock))
	for name, part := range map[string]ai.ContentPart{
		"corrupt result":     {Kind: ai.ContentExtension, Extensions: []ai.Extension{{Namespace: extensionNamespace, Type: hostedToolResultExtension, Data: json.RawMessage(`"truncated`)}}},
		"call without block": ai.HostedToolCallPart(extensionNamespace, ai.HostedToolCall{Kind: ai.HostedToolWebSearch}),
	} {
		t.Run(name, func(t *testing.T) {
			_, err := mapNativeMessages([]ai.Message{ai.TextMessage(ai.RoleUser, "x"), {Role: ai.RoleAssistant, Parts: []ai.ContentPart{call, part}}}, false)
			if err == nil {
				t.Fatal("unreplayable hosted item was dropped silently")
			}
		})
	}
}

func TestHostedPartWithRequiredSiblingIsRejected(t *testing.T) {
	part := serverToolUsePart(json.RawMessage(serverToolUseBlock))
	part.Extensions = append(part.Extensions, ai.Extension{Namespace: "other", Type: "state", Data: json.RawMessage(`"s"`), Required: true})
	_, err := mapNativeMessages([]ai.Message{ai.TextMessage(ai.RoleUser, "x"), {Role: ai.RoleAssistant, Parts: []ai.ContentPart{part}}}, false)
	if !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("error = %v", err)
	}
}
