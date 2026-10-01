package anthropic

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"testing"

	antropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/lace-ai/gai/ai"
)

func TestCanonicalAnthropicRoundTripPreservesOrderedSignedReasoning(t *testing.T) {
	var blocks []antropic.ContentBlockUnion
	if err := json.Unmarshal([]byte(`[{"type":"text","text":"before"},{"type":"thinking","thinking":"reason","signature":"signed"},{"type":"tool_use","id":"one","name":"lookup","input":{"x":1}},{"type":"text","text":"after"},{"type":"redacted_thinking","data":"opaque"}]`), &blocks); err != nil {
		t.Fatal(err)
	}
	message, err := mapCanonicalMessageContent(blocks)
	if err != nil {
		t.Fatal(err)
	}
	stored, err := json.Marshal(message)
	if err != nil {
		t.Fatal(err)
	}
	var restored ai.Message
	if err := json.Unmarshal(stored, &restored); err != nil {
		t.Fatal(err)
	}
	params, err := buildMessagesRequest(ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleSystem, "rules"), restored}}, ai.ModelDescriptor{Model: "test"})
	if err != nil {
		t.Fatal(err)
	}
	if len(params.System) != 1 || params.System[0].Text != "rules" || len(params.Messages) != 1 {
		t.Fatalf("params = %#v", params)
	}
	raw, _ := json.Marshal(params.Messages[0])
	var mapped struct {
		Content []map[string]any `json:"content"`
	}
	if err := json.Unmarshal(raw, &mapped); err != nil {
		t.Fatal(err)
	}
	if len(mapped.Content) != 5 || mapped.Content[1]["signature"] != "signed" || mapped.Content[2]["id"] != "one" || mapped.Content[3]["text"] != "after" || mapped.Content[4]["data"] != "opaque" {
		t.Fatalf("roundtrip = %s", raw)
	}
}

func TestCanonicalAnthropicRejectsUnsignedReasoning(t *testing.T) {
	_, err := mapNativeMessages([]ai.Message{{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentReasoning, Text: "private"}}}})
	if !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("error = %v", err)
	}
}

func TestCanonicalAnthropicStreamSignatureRoundTrip(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		events := []string{
			`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reason"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
			`{"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"empty-signed"}}`,
			`{"type":"content_block_stop","index":1}`,
			`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"one","name":"lookup","input":{}}}`,
			`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"x\":1}"}}`,
			`{"type":"content_block_stop","index":2}`,
			`{"type":"message_stop"}`,
		}
		for _, event := range events {
			var envelope struct {
				Type string `json:"type"`
			}
			_ = json.Unmarshal([]byte(event), &envelope)
			_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", envelope.Type, event)
		}
	})
	message := ai.Message{Role: ai.RoleAssistant}
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "question")}}) {
		if token.Err != nil {
			t.Fatal(token.Err)
		}
		message.AppendToken(token)
	}
	data, _ := json.Marshal(message)
	var restored ai.Message
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatal(err)
	}
	if len(restored.Parts) != 3 || restored.Parts[0].Text != "reason" || restored.Parts[1].Kind != ai.ContentReasoning || restored.Parts[1].Text != "" {
		t.Fatalf("streamed message = %s", data)
	}
	mapped, err := mapNativeMessages([]ai.Message{restored})
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(mapped)
	var messages []struct {
		Content []map[string]any `json:"content"`
	}
	if err := json.Unmarshal(raw, &messages); err != nil {
		t.Fatal(err)
	}
	if len(messages) != 1 || len(messages[0].Content) != 3 || messages[0].Content[0]["signature"] != "signed" || messages[0].Content[1]["signature"] != "empty-signed" || messages[0].Content[2]["id"] != "one" {
		t.Fatalf("mapped = %s", raw)
	}
}
