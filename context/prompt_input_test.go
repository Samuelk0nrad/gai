package context_test

import (
	"errors"
	"github.com/lace-ai/gai/ai"
	"strings"
	"testing"

	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestJSONPartUsesOneStructuredRepresentation(t *testing.T) {
	t.Parallel()

	part, err := gaictx.NewJSONPart("memory_observation", map[string]any{"name": "Sam"})
	if err != nil {
		t.Fatalf("NewJSONPart failed: %v", err)
	}
	node, err := part.Render(t.Context())
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}
	if node.Type != "memory_observation" || node.Value != `{"name":"Sam"}` {
		t.Fatalf("unexpected node: %+v", node)
	}
	counter := &mocks.MockTokenCounter{}
	if _, err := part.Tokens(t.Context(), counter); err != nil {
		t.Fatalf("Tokens failed: %v", err)
	}

	rendered, err := (&gaictx.SimpleRenderer{}).Render(t.Context(), []gaictx.Part{part})
	if err != nil {
		t.Fatalf("renderer failed: %v", err)
	}
	if rendered != "<memory_observation>\n{\"name\":\"Sam\"}\n</memory_observation>" {
		t.Fatalf("unexpected rendered part: %q", rendered)
	}
}

func TestNamedAndJSONPartValidation(t *testing.T) {
	t.Parallel()

	if _, err := gaictx.NewNamedPart("bad name", "value"); !errors.Is(err, gaictx.ErrPromptPartName) {
		t.Fatalf("expected ErrPromptPartName, got %v", err)
	}
	if _, err := gaictx.NewJSONPart("value", make(chan int)); err == nil || !strings.Contains(err.Error(), "marshal value prompt part") {
		t.Fatalf("expected contextual JSON marshal error, got %v", err)
	}
}

func TestPromptInputCloneOwnsContextSlice(t *testing.T) {
	t.Parallel()

	first, _ := gaictx.NewNamedPart("first", "one")
	second, _ := gaictx.NewNamedPart("second", "two")
	input := gaictx.PromptInput{User: ai.TextParts("hello"), Context: []gaictx.Part{first}}
	cloned := input.Clone()
	cloned.Context[0] = second

	if input.Context[0].Name() != "first" {
		t.Fatalf("clone shared context slice: %+v", input.Context)
	}
	if cloned.User[0].Text != "hello" {
		t.Fatalf("clone lost user content: %+v", cloned)
	}
}

func TestPromptInputCloneOwnsCanonicalPayloads(t *testing.T) {
	t.Parallel()
	input := gaictx.PromptInput{User: []ai.ContentPart{{Kind: ai.ContentJSON, JSON: []byte(`{"value":1}`)}, {Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", Data: []byte{1, 2}}}}}
	cloned := input.Clone()
	cloned.User[0].JSON[9] = '2'
	cloned.User[1].Media.Data[0] = 9
	if string(input.User[0].JSON) != `{"value":1}` || input.User[1].Media.Data[0] != 1 {
		t.Fatal("input clone aliases canonical payloads")
	}
}
