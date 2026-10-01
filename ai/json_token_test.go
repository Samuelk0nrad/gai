package ai_test

import (
	"encoding/json"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestJSONTokenPreservesCanonicalContentAndVisibleProjection(t *testing.T) {
	part := ai.ContentPart{Kind: ai.ContentJSON, JSON: json.RawMessage(`{"answer":"Paris"}`)}
	token := ai.Token{

		Part: &part,
	}
	cloned := token.Clone()
	if cloned.Type() != ai.TokenTypeText || cloned.Text() != string(part.JSON) || cloned.ToolCall() != nil {
		t.Fatalf("visible JSON projection = %#v", cloned)
	}
	if cloned.Part == nil || cloned.Part.Kind != ai.ContentJSON || string(cloned.Part.JSON) != string(part.JSON) {
		t.Fatalf("canonical JSON lost: %#v", cloned.Part)
	}
	cloned.Part.JSON[2] = 'X'
	if string(part.JSON) != `{"answer":"Paris"}` {
		t.Fatal("cloned token aliases source JSON")
	}
}
