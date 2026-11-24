package gemini

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
)

func media(part ai.MediaPart) ai.ContentPart {
	return ai.ContentPart{Kind: ai.ContentMedia, Media: &part}
}

func contentsJSON(t *testing.T, model string, messages ...ai.Message) string {
	t.Helper()
	contents, err := nativeContentsForModel(model, ai.AIRequest{Messages: messages})
	if err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	encoder := json.NewEncoder(&out)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(contents); err != nil {
		t.Fatal(err)
	}
	return strings.TrimSuffix(out.String(), "\n")
}

func TestUserMediaMapsToInlineAndFileData(t *testing.T) {
	tests := []struct {
		name  string
		media ai.MediaPart
		want  string
	}{
		{"image bytes", ai.MediaPart{MIMEType: "image/jpg", Data: []byte("jpg")}, `{"inlineData":{"data":"anBn","mimeType":"image/jpeg"}}`},
		{"image data URI", ai.MediaPart{MIMEType: "image/png", URI: "data:image/png;base64,cG5n"}, `{"inlineData":{"data":"cG5n","mimeType":"image/png"}}`},
		{"PDF URL", ai.MediaPart{MIMEType: "application/pdf", URI: "https://example.com/a.pdf"}, `{"fileData":{"fileUri":"https://example.com/a.pdf","mimeType":"application/pdf"}}`},
		{"text bytes", ai.MediaPart{MIMEType: "text/plain", Data: []byte("hi"), Name: "a.txt"}, `{"text":"<file name=\"a.txt\" mime_type=\"text/plain\">\nhi\n</file>"}`},
		{"audio bytes with alias MIME", ai.MediaPart{MIMEType: "audio/mp3", Data: []byte("mp3")}, `{"inlineData":{"data":"bXAz","mimeType":"audio/mpeg"}}`},
		{"video URL", ai.MediaPart{MIMEType: "video/mp4", URI: "https://www.youtube.com/watch?v=x"}, `{"fileData":{"fileUri":"https://www.youtube.com/watch?v=x","mimeType":"video/mp4"}}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := contentsJSON(t, "", ai.Message{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "look"}, media(tc.media)}})
			want := `[{"parts":[{"text":"look"},` + tc.want + `],"role":"user"}]`
			if got != want {
				t.Fatalf("got  %s\nwant %s", got, want)
			}
		})
	}
}

func TestToolResultMediaUsesFunctionResponseParts(t *testing.T) {
	message := ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "c1", Name: "open", Parts: []ai.ContentPart{
		{Kind: ai.ContentText, Text: "here"},
		media(ai.MediaPart{MIMEType: "image/png", Data: []byte("png")}),
		media(ai.MediaPart{MIMEType: "application/pdf", URI: "https://example.com/r.pdf"}),
		media(ai.MediaPart{MIMEType: "text/csv", Data: []byte("a,b")}),
	}}}}}
	call := ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "c1", Type: "function", Name: "open", Args: json.RawMessage(`{}`)}}}}
	callJSON := `{"parts":[{"functionCall":{"id":"c1","name":"open"}}],"role":"model"},`
	got := contentsJSON(t, Gemini3FlashPreview, call, message)
	want := `[` + callJSON + `{"parts":[{"functionResponse":{"parts":[{"inlineData":{"mimeType":"image/png","data":"cG5n"}},` +
		`{"fileData":{"fileUri":"https://example.com/r.pdf","mimeType":"application/pdf"}}],` +
		`"id":"c1","name":"open","response":{"output":"here\n<file mime_type=\"text/csv\">\na,b\n</file>"}}}],"role":"user"}]`
	if got != want {
		t.Fatalf("Gemini 3:\ngot  %s\nwant %s", got, want)
	}
	// Gemini 2.x does not accept functionResponse.parts: media follows as siblings.
	got = contentsJSON(t, Gemini2_5Flash, call, message)
	want = `[` + callJSON + `{"parts":[{"functionResponse":{"id":"c1","name":"open","response":{"output":"here\n<file mime_type=\"text/csv\">\na,b\n</file>"}}},` +
		`{"inlineData":{"data":"cG5n","mimeType":"image/png"}},` +
		`{"fileData":{"fileUri":"https://example.com/r.pdf","mimeType":"application/pdf"}}],"role":"user"}]`
	if got != want {
		t.Fatalf("Gemini 2.5:\ngot  %s\nwant %s", got, want)
	}
}

func TestUnsupportedMediaIsTyped(t *testing.T) {
	for name, message := range map[string]ai.Message{
		"zip":         {Role: ai.RoleUser, Parts: []ai.ContentPart{media(ai.MediaPart{MIMEType: "application/zip", Data: []byte{1}})}},
		"gif":         {Role: ai.RoleUser, Parts: []ai.ContentPart{media(ai.MediaPart{MIMEType: "image/gif", Data: []byte{1}})}},
		"file scheme": {Role: ai.RoleUser, Parts: []ai.ContentPart{media(ai.MediaPart{MIMEType: "image/png", URI: "file:///a.png"})}},
		"text by URL": {Role: ai.RoleUser, Parts: []ai.ContentPart{media(ai.MediaPart{MIMEType: "text/plain", URI: "https://example.com/a.txt"})}},
		"assistant":   {Role: ai.RoleAssistant, Parts: []ai.ContentPart{media(ai.MediaPart{MIMEType: "image/png", Data: []byte{1}})}},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := nativeContents(ai.AIRequest{Messages: []ai.Message{message}})
			var unsupported *ai.UnsupportedContentError
			if !errors.As(err, &unsupported) || unsupported.Provider != "gemini" || !errors.Is(err, ai.ErrUnsupportedCapability) {
				t.Fatalf("err = %v", err)
			}
		})
	}
}

func TestToolResultAudioFollowsFunctionResponse(t *testing.T) {
	call := ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "c1", Type: "function", Name: "record", Args: json.RawMessage(`{}`)}}}}
	result := ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "c1", Name: "record", Parts: []ai.ContentPart{
		media(ai.MediaPart{MIMEType: "audio/wav", Data: []byte("wav")}),
	}}}}}
	got := contentsJSON(t, Gemini3FlashPreview, call, result)
	want := `[{"parts":[{"functionCall":{"id":"c1","name":"record"}}],"role":"model"},` +
		`{"parts":[{"functionResponse":{"id":"c1","name":"record","response":{"output":""}}},{"inlineData":{"data":"d2F2","mimeType":"audio/wav"}}],"role":"user"}]`
	if got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
}

func TestAcceptsMedia(t *testing.T) {
	model := &Model{name: Gemini3FlashPreview}
	for _, accepted := range []ai.MediaPart{
		{MIMEType: "audio/x-wav", Data: []byte{1}},
		{MIMEType: "video/quicktime", URI: "https://example.com/a.mov"},
		{MIMEType: "application/pdf", Data: []byte{1}},
	} {
		if err := model.AcceptsMedia(accepted, ai.MediaInToolResult); err != nil {
			t.Errorf("%s: %v", accepted.MIMEType, err)
		}
	}
	if err := model.AcceptsMedia(ai.MediaPart{MIMEType: "application/msword", Data: []byte{1}}, ai.MediaInUserMessage); !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Errorf("msword: %v", err)
	}
}

func TestSiblingToolResultMediaReadsTheModelID(t *testing.T) {
	for model, want := range map[string]bool{
		"gemini-2.5-flash":                          true,
		"models/gemini-2.5-flash":                   true,
		"publishers/google/models/gemini-2.5-flash": true,
		"projects/p/locations/us-central1/publishers/google/models/gemini-1.5-pro": true,
		"gemini-3-flash-preview": false,
		"projects/p/locations/global/publishers/google/models/gemini-3-pro-preview": false,
	} {
		if got := siblingToolResultMedia(model); got != want {
			t.Errorf("%s: got %t, want %t", model, got, want)
		}
	}
}
