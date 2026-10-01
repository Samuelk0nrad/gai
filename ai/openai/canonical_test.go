package openai

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestCanonicalChatRolesParallelCallsAndGoogleSignature(t *testing.T) {
	signature, _ := json.Marshal([]byte("opaque"))
	req := ai.AIRequest{Messages: []ai.Message{
		ai.TextMessage(ai.RoleSystem, "instructions"),
		ai.TextMessage(ai.RoleUser, "question"),
		{Role: ai.RoleAssistant, Parts: []ai.ContentPart{
			{Kind: ai.ContentText, Text: "checking"},
			{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "one", Type: "function", Name: "lookup", Args: json.RawMessage(`{"x":1}`), Extensions: []ai.Extension{{Namespace: "google", Type: "thought_signature", Data: signature, Required: true}}}},
			{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "two", Type: "function", Name: "lookup", Args: json.RawMessage(`{"x":2}`)}},
		}},
	}}
	params, err := buildChatCompletionParams("test", req, false)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		Messages []struct {
			Role, Content string
			ToolCalls     []struct {
				ID           string
				ExtraContent struct {
					Google struct {
						Signature []byte `json:"thought_signature"`
					} `json:"google"`
				} `json:"extra_content"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Messages) != 3 || body.Messages[0].Role != "system" || body.Messages[0].Content != "instructions" || len(body.Messages[2].ToolCalls) != 2 || string(body.Messages[2].ToolCalls[0].ExtraContent.Google.Signature) != "opaque" {
		t.Fatalf("canonical request: %s", raw)
	}
}

func TestCanonicalChatRejectsUnsupportedOrderingAndOpaqueContent(t *testing.T) {
	for _, parts := range [][]ai.ContentPart{
		{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "one", Type: "function", Name: "lookup", Args: json.RawMessage(`{}`)}}, {Kind: ai.ContentText, Text: "after"}},
		{{Kind: ai.ContentText, Text: "text", Extensions: []ai.Extension{{Namespace: "other", Type: "state", Data: json.RawMessage(`"secret"`), Required: true}}}},
		{{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", URI: "https://example.com/image.png"}}},
	} {
		_, err := mapNativeMessages([]ai.Message{{Role: ai.RoleAssistant, Parts: parts}})
		if !errors.Is(err, ai.ErrUnsupportedCapability) {
			t.Fatalf("error = %v", err)
		}
	}
}

func TestResponsesPreservesInterleavedTextCallsAndReasoningWithoutCalls(t *testing.T) {
	reasoning := reasoningExtension(json.RawMessage(`{"type":"reasoning","id":"rs1","encrypted_content":"opaque","summary":[]}`))
	messages := []ai.Message{{Role: ai.RoleAssistant, Parts: []ai.ContentPart{
		{Kind: ai.ContentText, Text: "first"}, reasoning,
		{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "one", Type: "function", Name: "lookup", Args: json.RawMessage(`{}`)}},
		{Kind: ai.ContentText, Text: "last"},
	}}}
	input, err := mapResponsesMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(input)
	var parts []map[string]any
	if err := json.Unmarshal(raw, &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 4 || parts[0]["content"] != "first" || parts[1]["encrypted_content"] != "opaque" || parts[2]["type"] != "function_call" || parts[3]["content"] != "last" {
		t.Fatalf("ordered input = %s", raw)
	}
	input, err = mapResponsesMessages([]ai.Message{{Role: ai.RoleAssistant, Parts: []ai.ContentPart{reasoning, {Kind: ai.ContentText, Text: "done"}}}})
	if err != nil || len(input) != 2 {
		t.Fatalf("reasoning without tool: %v %v", input, err)
	}
}

func TestResponsesRejectsCallAttachedReasoningState(t *testing.T) {
	part := reasoningExtension(json.RawMessage(`{"type":"reasoning","id":"rs1","encrypted_content":"opaque","summary":[]}`))
	call := &ai.ToolCall{ID: "call_1", Type: "function", Name: "lookup", Args: json.RawMessage(`{}`), Extensions: part.Extensions}
	_, err := mapResponsesMessages([]ai.Message{{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: call}}}})
	if !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("call-attached reasoning error = %v", err)
	}
	// Its canonical ordered representation still maps independently of a call.
	call.Extensions = nil
	input, err := mapResponsesMessages([]ai.Message{{Role: ai.RoleAssistant, Parts: []ai.ContentPart{part, {Kind: ai.ContentToolCall, ToolCall: call}}}})
	if err != nil || len(input) != 2 {
		t.Fatalf("ordered reasoning input = %#v, error = %v", input, err)
	}
}
