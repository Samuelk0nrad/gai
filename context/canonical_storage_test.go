package context_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

func TestStoredMessageRoundTripPreservesOrderedPartsAndExtensions(t *testing.T) {
	t.Parallel()
	want := gaictx.StoredMessage{
		SchemaVersion: gaictx.MessageSchemaVersion,
		ID:            "message-1", SessionID: "session-1", TurnID: "turn-1",
		Message: ai.Message{Role: ai.RoleAssistant,
			Extensions: []ai.Extension{{Namespace: "future-provider", Type: "continuation", Data: json.RawMessage(`{"opaque":[1,2,3]}`), Required: true}},
			Parts: []ai.ContentPart{
				{Kind: ai.ContentReasoning, Text: "working", Extensions: []ai.Extension{{Namespace: "provider", Type: "signature", Data: json.RawMessage(`"AAEC"`)}}},
				{Kind: ai.ContentText, Text: "before"},
				{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "call-a", Type: "function", Name: "search", Args: json.RawMessage(`{"query":"first"}`)}},
				{Kind: ai.ContentText, Text: "between"},
				{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "call-b", Type: "function", Name: "search", Args: json.RawMessage(`{"query":"second"}`)}},
			},
		},
	}
	encoded, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	var got gaictx.StoredMessage
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip changed message:\n got %#v\nwant %#v", got, want)
	}
	if _, err := ai.RenderMessages(t.Context(), []ai.Message{got.Message}); err == nil {
		t.Fatal("opaque required extension silently rendered as text")
	}
}

func TestStoredMessageRoundTripPreservesToolResultErrorsAndMedia(t *testing.T) {
	t.Parallel()
	want := gaictx.StoredMessage{SchemaVersion: gaictx.MessageSchemaVersion, Message: ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{
		ToolCallID: "call-b", Name: "search", IsError: true,
		Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "failed"}, {Kind: ai.ContentJSON, JSON: json.RawMessage(`{"retry":true}`)}, {Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", Data: []byte{1, 2, 3}}}},
	}}}}}
	data, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	var got gaictx.StoredMessage
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip changed tool result: %#v", got)
	}
}

func TestStoredMessageRejectsOldAndUnversionedFormats(t *testing.T) {
	t.Parallel()
	for _, payload := range []string{
		`{"ID":"old-id","Role":"user","Content":{"Text":"hello"}}`,
		`{"Role":"assistant","Content":{"ToolName":"search","Args":"{}"}}`,
		`{"Role":"tool","Content":{"ToolName":"search","Result":"ok"}}`,
		`{"message":{"role":"user","parts":[{"kind":"text","text":"hello"}]}}`,
		`{"schema_version":0,"message":{"role":"user","parts":[{"kind":"text","text":"hello"}]}}`,
		`{"schema_version":1,"Role":"user","Content":{"Text":"hello"}}`,
	} {
		var got gaictx.StoredMessage
		if err := json.Unmarshal([]byte(payload), &got); err == nil {
			t.Fatalf("noncanonical stored message accepted: %s", payload)
		}
	}
}

func TestStoredMessageRejectsUnsupportedVersionAndInvalidCanonicalContent(t *testing.T) {
	t.Parallel()
	for _, payload := range []string{
		`{"schema_version":2,"message":{"role":"user","parts":[{"kind":"text","text":"hello"}]}}`,
		`{"schema_version":1,"message":{"role":"user","parts":[{"kind":"unknown"}]}}`,
		`{"schema_version":1,"message":{"role":"assistant","parts":[{"kind":"tool_call","tool_call":{"Name":"search","Type":"function","Args":{}}}]}}`,
	} {
		var got gaictx.StoredMessage
		if err := json.Unmarshal([]byte(payload), &got); err == nil {
			t.Fatalf("invalid envelope accepted: %s", payload)
		}
	}
}

func TestStoredMessageWriterEmitsCurrentSchema(t *testing.T) {
	t.Parallel()
	message := gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "hello")}
	encoded, err := json.Marshal(message)
	if err != nil {
		t.Fatal(err)
	}
	var got gaictx.StoredMessage
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	if got.SchemaVersion != gaictx.MessageSchemaVersion || !reflect.DeepEqual(got.Message, message.Message) {
		t.Fatalf("new message was not written in the canonical schema: %s", encoded)
	}
	if message.SchemaVersion != 0 {
		t.Fatal("serialization mutated the caller's storage envelope")
	}
}
