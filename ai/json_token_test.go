package ai_test

import (
	"encoding/json"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestNormalizedJSONTokenPreservesCanonicalContentAndVisibleProjection(t *testing.T) {
	part := ai.ContentPart{Kind: ai.ContentJSON, JSON: json.RawMessage(`{"answer":"Paris"}`)}
	token := ai.Token{
		Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "stale", Name: "must_not_run"},
		Text: "stale text", Data: []byte("stale data"), Part: &part,
	}
	normalized := token.Normalized()
	if normalized.Type != ai.TokenTypeText || normalized.Text != string(part.JSON) || normalized.ToolCall != nil || normalized.Data != nil {
		t.Fatalf("visible JSON projection = %#v", normalized)
	}
	if normalized.Part == nil || normalized.Part.Kind != ai.ContentJSON || string(normalized.Part.JSON) != string(part.JSON) {
		t.Fatalf("canonical JSON lost: %#v", normalized.Part)
	}
	normalized.Part.JSON[2] = 'X'
	if string(part.JSON) != `{"answer":"Paris"}` {
		t.Fatal("normalized token aliases source JSON")
	}
}
