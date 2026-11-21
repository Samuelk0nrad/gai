package openai

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
)

const (
	webSearchItem = `{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"go 1.26 release"}}`
	citedMessage  = `{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Go 1.26 is out.","annotations":[{"type":"url_citation","url":"https://go.dev/doc/go1.26","title":"Go 1.26 Release Notes","start_index":0,"end_index":15}]}]}`
	reasoningItem = `{"id":"rs_1","type":"reasoning","encrypted_content":"opaque-reasoning","summary":[],"status":"completed"}`
	searchCall    = `{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"}`
)

func responsesTestModel(t *testing.T, handler http.HandlerFunc) ai.Model {
	t.Helper()
	ts := httptest.NewServer(handler)
	t.Cleanup(ts.Close)
	p := New("test-key", nil, WithResponsesTransport())
	p.baseURL = ts.URL
	m, err := p.Model("gpt-5.6-terra")
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func assertWebSearchAndCitation(t *testing.T, message ai.Message) {
	t.Helper()
	calls := HostedToolCalls(message)
	if len(calls) != 1 || calls[0].ID != "ws_1" || calls[0].Type != "web_search_call" || calls[0].Status != "completed" || calls[0].Action.Type != "search" || calls[0].Action.Query != "go 1.26 release" {
		t.Fatalf("hosted tool calls = %#v", calls)
	}
	citations := Citations(message)
	if len(citations) != 1 || citations[0].URL != "https://go.dev/doc/go1.26" || citations[0].Title != "Go 1.26 Release Notes" || citations[0].EndIndex != 15 {
		t.Fatalf("citations = %#v", citations)
	}
	if message.Text() != "Go 1.26 is out." {
		t.Fatalf("text = %q", message.Text())
	}
	if err := message.Validate(); err != nil {
		t.Fatalf("message invalid: %v", err)
	}
}

func TestResponsesGenerateKeepsHostedToolItemsAndCitations(t *testing.T) {
	m := responsesTestModel(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","status":"completed","output":[` + reasoningItem + `,` + webSearchItem + `,` + citedMessage + `],"usage":{"input_tokens":1,"output_tokens":1}}`))
	})
	response, err := any(m).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")}})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	parts := response.Message.Parts
	if len(parts) != 3 || parts[0].Extensions[0].Type != "responses_reasoning" || parts[1].Extensions[0].Type != ExtensionHostedTool || parts[2].Kind != ai.ContentText {
		t.Fatalf("parts = %#v", parts)
	}
	assertWebSearchAndCitation(t, response.Message)
}

func TestResponsesGenerateStillRejectsUnknownOutputItems(t *testing.T) {
	m := responsesTestModel(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","status":"completed","output":[{"id":"cu_1","type":"computer_call","call_id":"c","status":"completed","action":{"type":"screenshot"},"pending_safety_checks":[]}]}`))
	})
	_, err := any(m).(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x")}})
	if !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("error = %v, want unsupported capability", err)
	}
}

func sseEvent(data string) string { return "data: " + data + "\n\n" }

func TestResponsesStreamKeepsHostedToolItemsAndCitations(t *testing.T) {
	m := responsesTestModel(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		var body strings.Builder
		body.WriteString(sseEvent(`{"type":"response.output_item.done","output_index":0,"item":` + reasoningItem + `}`))
		body.WriteString(sseEvent(`{"type":"response.output_item.done","output_index":1,"item":` + webSearchItem + `}`))
		body.WriteString(sseEvent(`{"type":"response.output_text.delta","item_id":"msg_1","output_index":2,"content_index":0,"delta":"Go 1.26 "}`))
		body.WriteString(sseEvent(`{"type":"response.output_text.delta","item_id":"msg_1","output_index":2,"content_index":0,"delta":"is out."}`))
		body.WriteString(sseEvent(`{"type":"response.output_text.annotation.added","item_id":"msg_1","output_index":2,"content_index":0,"annotation_index":0,"annotation":{"type":"url_citation","url":"https://go.dev/doc/go1.26","title":"Go 1.26 Release Notes","start_index":0,"end_index":15}}`))
		body.WriteString(sseEvent(`{"type":"response.content_part.done","item_id":"msg_1","output_index":2,"content_index":0,"part":{"type":"output_text","text":"Go 1.26 is out.","annotations":[{"type":"url_citation","url":"https://go.dev/doc/go1.26","title":"Go 1.26 Release Notes","start_index":0,"end_index":15}]}}`))
		body.WriteString(sseEvent(`{"type":"response.output_item.done","output_index":2,"item":` + citedMessage + `}`))
		body.WriteString(sseEvent(`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1}}}`))
		_, _ = w.Write([]byte(body.String()))
	})
	message := ai.Message{Role: ai.RoleAssistant}
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")}}) {
		if token.Err != nil {
			t.Fatalf("stream error: %v", token.Err)
		}
		if err := token.Validate(); err != nil {
			t.Fatalf("invalid token %#v: %v", token, err)
		}
		message.AppendToken(token)
	}
	if len(message.Parts) != 3 || message.Parts[1].Extensions[0].Type != ExtensionHostedTool || message.Parts[2].Kind != ai.ContentText {
		t.Fatalf("parts = %#v", message.Parts)
	}
	assertWebSearchAndCitation(t, message)
}

// A search in one round and the answer in the next: the search item must go
// back to OpenAI in its original place among the reasoning and function items.
func TestResponsesReplaysHostedToolItemsInOrder(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		name := "generate"
		if streaming {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			var continuation map[string]any
			requests := 0
			m := responsesTestModel(t, func(w http.ResponseWriter, r *http.Request) {
				requests++
				if requests == 1 {
					if streaming {
						w.Header().Set("Content-Type", "text/event-stream")
						for _, item := range []string{reasoningItem, webSearchItem, searchCall} {
							_, _ = w.Write([]byte(sseEvent(`{"type":"response.output_item.done","item":` + item + `}`)))
						}
						return
					}
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(`{"id":"resp_1","status":"completed","output":[` + reasoningItem + `,` + webSearchItem + `,` + searchCall + `]}`))
					return
				}
				if err := json.NewDecoder(r.Body).Decode(&continuation); err != nil {
					t.Errorf("decode continuation: %v", err)
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"resp_2","status":"completed","output":[` + citedMessage + `]}`))
			})
			request := ai.AIRequest{
				Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")},
				Tools:    []ai.ToolDefinition{{Type: "function", Name: "lookup", Description: "Lookup", Parameters: json.RawMessage(`{"type":"object"}`)}},
			}
			var first ai.Message
			if streaming {
				first = ai.Message{Role: ai.RoleAssistant}
				for token := range m.GenerateStream(t.Context(), request) {
					if token.Err != nil {
						t.Fatalf("stream error: %v", token.Err)
					}
					first.AppendToken(token)
				}
			} else {
				response, err := any(m).(ai.ModelGenerator).Generate(t.Context(), request)
				if err != nil {
					t.Fatalf("first Generate: %v", err)
				}
				first = response.Message
			}
			call := first.ToolCalls()
			if len(call) != 1 || len(HostedToolCalls(first)) != 1 {
				t.Fatalf("first round = %#v", first.Parts)
			}
			// History round-trips through storage as JSON.
			stored, err := json.Marshal(first)
			if err != nil {
				t.Fatal(err)
			}
			var restored ai.Message
			if err := json.Unmarshal(stored, &restored); err != nil {
				t.Fatal(err)
			}
			request.Messages = append(request.Messages, restored, ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: call[0].ID, Name: call[0].Name, Parts: ai.TextParts("ok")}}}})
			second, err := any(m).(ai.ModelGenerator).Generate(t.Context(), request)
			if err != nil {
				t.Fatalf("second Generate: %v", err)
			}
			if len(Citations(second.Message)) != 1 {
				t.Fatalf("second round citations = %#v", second.Message.Parts)
			}
			input := continuation["input"].([]any)
			var types []string
			for _, item := range input {
				object := item.(map[string]any)
				itemType, _ := object["type"].(string)
				if itemType == "" {
					itemType = "message"
				}
				types = append(types, itemType)
			}
			if strings.Join(types, ",") != "message,reasoning,web_search_call,function_call,function_call_output" {
				t.Fatalf("continuation item types = %v", types)
			}
			search := input[2].(map[string]any)
			action, _ := search["action"].(map[string]any)
			if search["id"] != "ws_1" || search["status"] != "completed" || action["query"] != "go 1.26 release" {
				t.Fatalf("replayed search item = %#v", search)
			}
		})
	}
}

func TestResponsesReplayRejectsUnknownHostedToolItem(t *testing.T) {
	part := hostedToolExtension(json.RawMessage(`{"id":"x","type":"computer_call"}`))
	_, err := mapResponsesMessages([]ai.Message{{Role: ai.RoleAssistant, Parts: []ai.ContentPart{part}}})
	if !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("error = %v", err)
	}
}

func TestHostedToolAndCitationPartsAreOptionalForChatCompletions(t *testing.T) {
	message := ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{
		hostedToolExtension(json.RawMessage(webSearchItem)),
		{Kind: ai.ContentText, Text: "Go 1.26 is out.", Extensions: annotationsExtension(`[{"type":"url_citation","url":"https://go.dev","title":"Go","start_index":0,"end_index":2}]`)},
	}}
	if _, err := mapNativeMessages([]ai.Message{ai.TextMessage(ai.RoleUser, "news"), message}); err != nil {
		t.Fatalf("chat mapping: %v", err)
	}
}
