package anthropic

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
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
