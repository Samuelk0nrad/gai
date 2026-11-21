package ai_test

import (
	"encoding/json"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
)

func TestHostedToolCallsAndCitationsRoundTrip(t *testing.T) {
	message := ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{
		ai.HostedToolCallPart("p", ai.HostedToolCall{ID: "1", Kind: ai.HostedToolWebSearch, Query: "q", Raw: json.RawMessage(`{"x":1}`)}),
		{Kind: ai.ContentText, Text: "answer", Extensions: ai.CitationsExtension("p", []ai.Citation{{URL: "https://example.test", Start: 0, End: 6, Text: "answer"}})},
	}}
	stored, err := json.Marshal(message)
	if err != nil {
		t.Fatal(err)
	}
	var restored ai.Message
	if err := json.Unmarshal(stored, &restored); err != nil {
		t.Fatal(err)
	}
	calls := restored.HostedToolCalls()
	if len(calls) != 1 || calls[0].Provider != "p" || calls[0].Query != "q" || string(calls[0].Raw) != `{"x":1}` {
		t.Fatalf("calls = %#v", calls)
	}
	citations := restored.Citations()
	if len(citations) != 1 || citations[0].Provider != "p" || citations[0].URL != "https://example.test" || citations[0].End != 6 {
		t.Fatalf("citations = %#v", citations)
	}
	if ai.CitationsExtension("p", nil) != nil {
		t.Fatal("empty citations produced an extension")
	}
}

func TestWithoutOptionalExtensionsKeepsRequiredState(t *testing.T) {
	required := ai.Extension{Namespace: "p", Type: "state", Data: json.RawMessage(`"s"`), Required: true}
	message := ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{
		ai.HostedToolCallPart("p", ai.HostedToolCall{Kind: ai.HostedToolWebSearch}),
		{Kind: ai.ContentText, Text: "answer", Extensions: ai.CitationsExtension("p", []ai.Citation{{URL: "u"}})},
		{Kind: ai.ContentExtension, Extensions: []ai.Extension{required}},
	}}
	plain, ok := message.WithoutOptionalExtensions()
	if !ok || len(plain.Parts) != 2 || len(plain.Parts[0].Extensions) != 0 || !plain.Parts[1].Extensions[0].Required {
		t.Fatalf("plain = %#v", plain)
	}
	if len(message.Parts[1].Extensions) != 1 {
		t.Fatal("source message was changed")
	}
	if _, ok := (ai.Message{Role: ai.RoleAssistant, Parts: message.Parts[:1]}).WithoutOptionalExtensions(); ok {
		t.Fatal("message with only optional state should be empty")
	}
}
