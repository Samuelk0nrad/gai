package mistral

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
)

func mediaContentPart(media ai.MediaPart) ai.ContentPart {
	return ai.ContentPart{Kind: ai.ContentMedia, Media: &media}
}

func mappedJSON(t *testing.T, messages ...ai.Message) string {
	t.Helper()
	mapped, err := mapNativeMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(mapped)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func TestUserMediaMapsPDFURLAndTextFiles(t *testing.T) {
	got := mappedJSON(t, ai.Message{Role: ai.RoleUser, Parts: []ai.ContentPart{
		{Kind: ai.ContentText, Text: "read"},
		mediaContentPart(ai.MediaPart{MIMEType: "application/pdf", URI: "https://example.com/a.pdf", Name: "a.pdf"}),
		mediaContentPart(ai.MediaPart{MIMEType: "application/json", Data: []byte(`{"a":1}`)}),
	}})
	fileText, _ := json.Marshal("<file mime_type=\"application/json\">\n{\"a\":1}\n</file>")
	want := `[{"role":"user","content":[{"type":"text","text":"read"},` +
		`{"type":"document_url","document_url":"https://example.com/a.pdf","document_name":"a.pdf"},` +
		`{"type":"text","text":` + string(fileText) + `}]}]`
	if got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
}

func TestToolResultMediaIsSentAsChunks(t *testing.T) {
	call := ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "c1", Type: "function", Name: "open", Args: json.RawMessage(`{}`)}}}}
	result := func(isError bool, parts ...ai.ContentPart) ai.Message {
		return ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "c1", Name: "open", Parts: parts, IsError: isError}}}}
	}
	image := mediaContentPart(ai.MediaPart{MIMEType: "image/png", Data: []byte("png")})
	callJSON := `{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"open","arguments":"{}"}}]}`

	got := mappedJSON(t, call, result(false, ai.ContentPart{Kind: ai.ContentText, Text: "here"}, image))
	want := `[` + callJSON + `,{"role":"tool","content":[{"type":"text","text":"here"},{"type":"image_url","image_url":"data:image/png;base64,cG5n"}],"tool_call_id":"c1"}]`
	if got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
	got = mappedJSON(t, call, result(true, ai.ContentPart{Kind: ai.ContentText, Text: "bad"}, image))
	want = `[` + callJSON + `,{"role":"tool","content":[{"type":"text","text":"{\"error\":\"bad\"}"},{"type":"image_url","image_url":"data:image/png;base64,cG5n"}],"tool_call_id":"c1"}]`
	if got != want {
		t.Fatalf("error result:\ngot  %s\nwant %s", got, want)
	}
	// Text-only results stay a plain string.
	got = mappedJSON(t, call, result(false, ai.ContentPart{Kind: ai.ContentText, Text: "plain"}))
	want = `[` + callJSON + `,{"role":"tool","content":"plain","tool_call_id":"c1"}]`
	if got != want {
		t.Fatalf("text result:\ngot  %s\nwant %s", got, want)
	}
	if !containsImageInput([]ai.Message{call, result(false, image)}) {
		t.Fatal("tool-result image not seen by the vision check")
	}
}

func TestInlinePDFIsRejected(t *testing.T) {
	_, err := mapNativeMessages([]ai.Message{{Role: ai.RoleUser, Parts: []ai.ContentPart{
		mediaContentPart(ai.MediaPart{MIMEType: "application/pdf", Data: []byte("%PDF")}),
	}}})
	var unsupported *ai.UnsupportedContentError
	if !errors.As(err, &unsupported) || unsupported.Provider != "mistral" {
		t.Fatalf("err = %v", err)
	}
}
