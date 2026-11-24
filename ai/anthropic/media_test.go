package anthropic

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
)

func mediaPart(media ai.MediaPart) ai.ContentPart {
	return ai.ContentPart{Kind: ai.ContentMedia, Media: &media}
}

// messagesJSON maps messages and returns them as compact JSON.
func messagesJSON(t *testing.T, messages ...ai.Message) string {
	t.Helper()
	mapped, err := mapNativeMessages(messages, false)
	if err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	encoder := json.NewEncoder(&out)
	encoder.SetEscapeHTML(false)
	raw, err := json.Marshal(mapped)
	if err != nil {
		t.Fatal(err)
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatal(err)
	}
	if err := encoder.Encode(value); err != nil {
		t.Fatal(err)
	}
	return strings.TrimSuffix(out.String(), "\n")
}

func TestUserMediaMapsToImageAndDocumentBlocks(t *testing.T) {
	tests := []struct {
		name  string
		media ai.MediaPart
		want  string
	}{
		{"image URL", ai.MediaPart{MIMEType: "image/png", URI: "https://example.com/cat.png"},
			`{"source":{"type":"url","url":"https://example.com/cat.png"},"type":"image"}`},
		{"image bytes with alias MIME", ai.MediaPart{MIMEType: "image/jpg", Data: []byte("jpg")},
			`{"source":{"data":"anBn","media_type":"image/jpeg","type":"base64"},"type":"image"}`},
		{"image data URI", ai.MediaPart{MIMEType: "image/webp", URI: "data:image/webp;base64,d2VicA=="},
			`{"source":{"data":"d2VicA==","media_type":"image/webp","type":"base64"},"type":"image"}`},
		{"PDF URL with name", ai.MediaPart{MIMEType: "application/pdf", URI: "https://example.com/a.pdf", Name: "a.pdf"},
			`{"source":{"type":"url","url":"https://example.com/a.pdf"},"title":"a.pdf","type":"document"}`},
		{"PDF bytes", ai.MediaPart{MIMEType: "application/pdf", Data: []byte("%PDF")},
			`{"source":{"data":"JVBERg==","media_type":"application/pdf","type":"base64"},"type":"document"}`},
		{"text bytes", ai.MediaPart{MIMEType: "text/csv", Data: []byte("a,b"), Name: "t.csv"},
			`{"source":{"data":"a,b","media_type":"text/plain","type":"text"},"title":"t.csv","type":"document"}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := messagesJSON(t, ai.Message{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "look"}, mediaPart(tc.media)}})
			want := `[{"content":[{"text":"look","type":"text"},` + tc.want + `],"role":"user"}]`
			if got != want {
				t.Fatalf("got  %s\nwant %s", got, want)
			}
		})
	}
}

func TestToolResultMediaKeepsOrder(t *testing.T) {
	result := func(isError bool, parts ...ai.ContentPart) ai.Message {
		return ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "c1", Name: "open", Parts: parts, IsError: isError}}}}
	}
	got := messagesJSON(t, result(false,
		ai.ContentPart{Kind: ai.ContentText, Text: "here"},
		mediaPart(ai.MediaPart{MIMEType: "image/png", Data: []byte("png")}),
		mediaPart(ai.MediaPart{MIMEType: "application/pdf", URI: "https://example.com/r.pdf"}),
	))
	want := `[{"content":[{"content":[{"text":"here","type":"text"},` +
		`{"source":{"data":"cG5n","media_type":"image/png","type":"base64"},"type":"image"},` +
		`{"source":{"type":"url","url":"https://example.com/r.pdf"},"type":"document"}],` +
		`"is_error":false,"tool_use_id":"c1","type":"tool_result"}],"role":"user"}]`
	if got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
	// Text-only results keep their single text block.
	got = messagesJSON(t, result(true, ai.ContentPart{Kind: ai.ContentText, Text: "boom"}))
	want = `[{"content":[{"content":[{"text":"boom","type":"text"}],"is_error":true,"tool_use_id":"c1","type":"tool_result"}],"role":"user"}]`
	if got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
}

func TestUnsupportedMediaIsTyped(t *testing.T) {
	for name, message := range map[string]ai.Message{
		"audio":        {Role: ai.RoleUser, Parts: []ai.ContentPart{mediaPart(ai.MediaPart{MIMEType: "audio/mpeg", Data: []byte{1}})}},
		"s3 scheme":    {Role: ai.RoleUser, Parts: []ai.ContentPart{mediaPart(ai.MediaPart{MIMEType: "image/png", URI: "s3://bucket/a.png"})}},
		"text by URL":  {Role: ai.RoleUser, Parts: []ai.ContentPart{mediaPart(ai.MediaPart{MIMEType: "text/plain", URI: "https://example.com/a.txt"})}},
		"invalid UTF8": {Role: ai.RoleUser, Parts: []ai.ContentPart{mediaPart(ai.MediaPart{MIMEType: "text/plain", Data: []byte{0xff}})}},
		"assistant":    {Role: ai.RoleAssistant, Parts: []ai.ContentPart{mediaPart(ai.MediaPart{MIMEType: "image/png", Data: []byte{1}})}},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := mapNativeMessages([]ai.Message{message}, false)
			var unsupported *ai.UnsupportedContentError
			if !errors.As(err, &unsupported) || unsupported.Provider != "anthropic" || !errors.Is(err, ai.ErrUnsupportedCapability) {
				t.Fatalf("err = %v", err)
			}
		})
	}
}

func TestAcceptsMedia(t *testing.T) {
	model := &Model{}
	if err := model.AcceptsMedia(ai.MediaPart{MIMEType: "application/pdf", URI: "https://example.com/a.pdf"}, ai.MediaInToolResult); err != nil {
		t.Fatal(err)
	}
	for _, rejected := range []string{"audio/mpeg", "video/mp4", "image/heic", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"} {
		if err := model.AcceptsMedia(ai.MediaPart{MIMEType: rejected, Data: []byte{1}}, ai.MediaInUserMessage); !errors.Is(err, ai.ErrUnsupportedCapability) {
			t.Errorf("%s: %v", rejected, err)
		}
	}
}
