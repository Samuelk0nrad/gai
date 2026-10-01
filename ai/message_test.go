package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestContentPartValidateTaggedPayloads(t *testing.T) {
	ext := ai.Extension{Namespace: "future", Type: "opaque", Data: json.RawMessage(`"AQID"`)}
	tests := []struct {
		name  string
		part  ai.ContentPart
		valid bool
	}{
		{"empty text", ai.ContentPart{Kind: ai.ContentText}, true},
		{"reasoning", ai.ContentPart{Kind: ai.ContentReasoning, Text: "think"}, true},
		{"call", canonicalCall("first"), true},
		{"result", canonicalResult("first", "search"), true},
		{"json", ai.ContentPart{Kind: ai.ContentJSON, JSON: json.RawMessage(`null`)}, true},
		{"media uri", ai.ContentPart{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", URI: "https://example.test/image"}}, true},
		{"media bytes", ai.ContentPart{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", Data: []byte{1}}}, true},
		{"extension", ai.ContentPart{Kind: ai.ContentExtension, Extensions: []ai.Extension{ext}}, true},
		{"unknown kind", ai.ContentPart{Kind: "future"}, false},
		{"untagged payload", ai.ContentPart{Text: "hello"}, false},
		{"two payloads", ai.ContentPart{Kind: ai.ContentJSON, JSON: json.RawMessage(`{}`), Text: "hello"}, false},
		{"text with call", ai.ContentPart{Kind: ai.ContentText, ToolCall: canonicalCall("first").ToolCall}, false},
		{"missing call", ai.ContentPart{Kind: ai.ContentToolCall}, false},
		{"missing result", ai.ContentPart{Kind: ai.ContentToolResult}, false},
		{"missing json", ai.ContentPart{Kind: ai.ContentJSON}, false},
		{"invalid json", ai.ContentPart{Kind: ai.ContentJSON, JSON: json.RawMessage(`{`)}, false},
		{"missing media", ai.ContentPart{Kind: ai.ContentMedia}, false},
		{"media without mime", ai.ContentPart{Kind: ai.ContentMedia, Media: &ai.MediaPart{Data: []byte{1}}}, false},
		{"media without content", ai.ContentPart{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png"}}, false},
		{"ambiguous media", ai.ContentPart{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", URI: "uri", Data: []byte{1}}}, false},
		{"extension without data", ai.ContentPart{Kind: ai.ContentExtension}, false},
		{"extension invalid json", ai.ContentPart{Kind: ai.ContentExtension, Extensions: []ai.Extension{{Namespace: "provider", Type: "state", Data: json.RawMessage(`{`)}}}, false},
		{"extension without namespace", ai.ContentPart{Kind: ai.ContentText, Extensions: []ai.Extension{{Type: "state", Data: json.RawMessage(`{}`)}}}, false},
		{"extension without type", ai.ContentPart{Kind: ai.ContentText, Extensions: []ai.Extension{{Namespace: "provider", Data: json.RawMessage(`{}`)}}}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.part.Validate(); (err == nil) != tt.valid {
				t.Fatalf("Validate() = %v, valid = %t", err, tt.valid)
			}
		})
	}
}

func TestContentPartRejectsInvalidCallsAndNestedToolResults(t *testing.T) {
	for _, mutate := range []struct {
		name string
		fn   func(*ai.ToolCall)
	}{
		{"missing id", func(c *ai.ToolCall) { c.ID = "" }},
		{"blank id", func(c *ai.ToolCall) { c.ID = " " }},
		{"missing name", func(c *ai.ToolCall) { c.Name = "" }},
		{"wrong type", func(c *ai.ToolCall) { c.Type = "other" }},
		{"invalid arguments", func(c *ai.ToolCall) { c.Args = json.RawMessage(`{`) }},
	} {
		t.Run(mutate.name, func(t *testing.T) {
			p := canonicalCall("first")
			mutate.fn(p.ToolCall)
			if p.Validate() == nil {
				t.Fatal("invalid call accepted")
			}
		})
	}
	for _, kind := range []ai.ContentKind{ai.ContentToolCall, ai.ContentToolResult, ai.ContentReasoning} {
		t.Run("nested "+string(kind), func(t *testing.T) {
			p := canonicalResult("first", "search")
			child := ai.ContentPart{Kind: kind, Text: "reasoning"}
			if kind == ai.ContentToolCall {
				child = canonicalCall("second")
			}
			if kind == ai.ContentToolResult {
				child = canonicalResult("second", "search")
			}
			p.ToolResult.Parts = []ai.ContentPart{child}
			if p.Validate() == nil {
				t.Fatal("invalid nested content accepted")
			}
		})
	}
}

func TestMessageValidateRoleConstraints(t *testing.T) {
	for _, role := range []ai.Role{ai.RoleSystem, ai.RoleUser, ai.RoleAssistant, ai.RoleTool} {
		for _, part := range []ai.ContentPart{{Kind: ai.ContentText, Text: "hello"}, {Kind: ai.ContentReasoning, Text: "think"}, canonicalCall("first"), canonicalResult("first", "search")} {
			t.Run(string(role)+"/"+string(part.Kind), func(t *testing.T) {
				valid := part.Kind == ai.ContentText && role != ai.RoleTool || (part.Kind == ai.ContentReasoning || part.Kind == ai.ContentToolCall) && role == ai.RoleAssistant || part.Kind == ai.ContentToolResult && role == ai.RoleTool
				if err := (ai.Message{Role: role, Parts: []ai.ContentPart{part}}).Validate(); (err == nil) != valid {
					t.Fatalf("Validate() = %v, valid = %t", err, valid)
				}
			})
		}
	}
	for _, m := range []ai.Message{{Role: "future", Parts: ai.TextParts("hello")}, {Role: ai.RoleUser}} {
		if m.Validate() == nil {
			t.Fatalf("invalid message accepted: %#v", m)
		}
	}
}

func TestCanonicalMessagesRoundTripUnknownStateAndOrderedContent(t *testing.T) {
	unknown := ai.Extension{Namespace: "future.provider", Type: "unrecognized", Data: json.RawMessage(`{"binary":"AP8B","nested":[1,true]}`), Required: true}
	call := canonicalCall("first")
	call.ToolCall.ThoughtSignature = []byte{0, 255, 1}
	call.ToolCall.Extensions = []ai.Extension{unknown}
	messages := []ai.Message{
		{Role: ai.RoleAssistant, Extensions: []ai.Extension{unknown}, Parts: []ai.ContentPart{
			{Kind: ai.ContentText, Text: "before"},
			{Kind: ai.ContentReasoning, Text: "thinking", Extensions: []ai.Extension{unknown}},
			call,
			{Kind: ai.ContentText, Text: "between"},
			canonicalCall("second"),
			{Kind: ai.ContentJSON, JSON: json.RawMessage(`{"done":false}`)},
			{Kind: ai.ContentExtension, Extensions: []ai.Extension{unknown}},
		}},
		{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "first", Name: "search", IsError: true, Parts: []ai.ContentPart{
			{Kind: ai.ContentText, Text: "image unavailable"},
			{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", Data: []byte{0, 255, 1}}},
			{Kind: ai.ContentExtension, Extensions: []ai.Extension{unknown}},
		}}}, canonicalResult("second", "search")}},
	}
	if err := (ai.AIRequest{Messages: messages}).Validate(); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(messages)
	if err != nil {
		t.Fatal(err)
	}
	var restored []ai.Message
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored, messages) {
		t.Fatalf("round trip lost semantic content:\ngot %#v\nwant %#v", restored, messages)
	}
}

func TestRenderMessagesPreservesRolesOrderIdentityAndEscaping(t *testing.T) {
	call := canonicalCall("call<&\"")
	call.ToolCall.Args = json.RawMessage(`{"q":"<text>&"}`)
	result := canonicalResult(call.ToolCall.ID, "search")
	result.ToolResult.IsError = true
	result.ToolResult.Parts = []ai.ContentPart{{Kind: ai.ContentText, Text: "failure <&>"}, {Kind: ai.ContentJSON, JSON: json.RawMessage(`{"retry":true}`)}}
	messages := []ai.Message{
		ai.TextMessage(ai.RoleSystem, "instructions"),
		ai.TextMessage(ai.RoleUser, "question"),
		{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "before"}, {Kind: ai.ContentReasoning, Text: "think"}, call, {Kind: ai.ContentText, Text: "after"}}},
		{Role: ai.RoleTool, Parts: []ai.ContentPart{result}},
	}
	got, err := ai.RenderMessages(context.Background(), messages)
	if err != nil {
		t.Fatal(err)
	}
	want := "<system>\ninstructions\n</system>\n<user>\nquestion\n</user>\n<assistant>\nbefore\n<reasoning>think</reasoning>\n<tool_call id=\"call&lt;&amp;&#34;\" name=\"search\">{&#34;q&#34;:&#34;&lt;text&gt;&amp;&#34;}</tool_call>\nafter\n</assistant>\n<tool>\n<tool_result id=\"call&lt;&amp;&#34;\" name=\"search\" is_error=\"true\">failure &lt;&amp;&gt;<json>{&#34;retry&#34;:true}</json></tool_result>\n</tool>\n"
	if got != want {
		t.Fatalf("rendered conversation:\ngot %s\nwant %s", got, want)
	}
}

func TestRenderMessagesRejectsOpaqueAndMediaContent(t *testing.T) {
	ext := ai.Extension{Namespace: "future", Type: "secret", Data: json.RawMessage(`"opaque-state"`)}
	call := canonicalCall("first")
	call.ToolCall.Extensions = []ai.Extension{ext}
	signed := canonicalCall("signed")
	signed.ToolCall.ThoughtSignature = []byte("opaque-state")
	for _, tt := range []struct {
		name    string
		message ai.Message
	}{
		{"message extension", ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("visible"), Extensions: []ai.Extension{ext}}},
		{"part extension", ai.Message{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "visible", Extensions: []ai.Extension{ext}}}}},
		{"standalone extension", ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentExtension, Extensions: []ai.Extension{ext}}}}},
		{"call extension", ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{call}}},
		{"legacy signature", ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{signed}}},
		{"media", ai.Message{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", Data: []byte{1}}}}}},
		{"nested media", ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "first", Name: "search", Parts: []ai.ContentPart{{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", URI: "https://example.test/image"}}}}}}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			text, err := ai.RenderMessages(context.Background(), []ai.Message{tt.message})
			var unsupported *ai.UnsupportedContentError
			if !errors.Is(err, ai.ErrUnsupportedCapability) || !errors.As(err, &unsupported) {
				t.Fatalf("error = %v, want typed unsupported content error", err)
			}
			if text != "" || strings.Contains(err.Error(), "opaque-state") {
				t.Fatalf("renderer leaked partial output or opaque state: text=%q err=%v", text, err)
			}
		})
	}
}

func TestRenderMessagesHonorsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ai.RenderMessages(ctx, []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want cancellation", err)
	}
}
