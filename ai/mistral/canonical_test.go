package mistral

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestCanonicalMistralKeepsSystemAndParallelToolCalls(t *testing.T) {
	params, err := buildChatCompletionRequest(ai.AIRequest{Messages: []ai.Message{
		ai.TextMessage(ai.RoleSystem, "rules"),
		{Role: ai.RoleAssistant, Parts: []ai.ContentPart{
			{Kind: ai.ContentText, Text: "checking"},
			{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "one", Type: "function", Name: "lookup", Args: json.RawMessage(`{"x":1}`)}},
			{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "two", Type: "function", Name: "lookup", Args: json.RawMessage(`{"x":2}`)}},
		}},
	}}, "test", false)
	if err != nil {
		t.Fatal(err)
	}
	if len(params.Messages) != 2 || params.Messages[0].Role != "system" || params.Messages[0].Content != "rules" || params.Messages[1].Content != "checking" || len(params.Messages[1].ToolCalls) != 2 {
		t.Fatalf("params = %#v", params)
	}
}

func TestCanonicalMistralRejectsRequiredProviderState(t *testing.T) {
	message := ai.TextMessage(ai.RoleUser, "hello")
	message.Extensions = []ai.Extension{{Namespace: "other", Type: "state", Data: json.RawMessage(`"secret"`), Required: true}}
	_, err := mapNativeMessages([]ai.Message{message})
	if !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("error = %v", err)
	}
}
