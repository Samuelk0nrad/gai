package openai

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/obstest"
)

func mediaPart(media ai.MediaPart) ai.ContentPart {
	return ai.ContentPart{Kind: ai.ContentMedia, Media: &media}
}

// responsesInputJSON maps messages and returns the input array as JSON.
func responsesInputJSON(t *testing.T, messages ...ai.Message) string {
	t.Helper()
	params, err := buildResponsesParams("gpt-test", ai.AIRequest{Messages: messages})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		Input any `json:"input"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	// Re-encode without HTML escaping so expectations stay readable.
	var out strings.Builder
	encoder := json.NewEncoder(&out)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(body.Input); err != nil {
		t.Fatal(err)
	}
	return strings.TrimSuffix(out.String(), "\n")
}

func TestResponsesMapsUserMediaInOrder(t *testing.T) {
	tests := []struct {
		name  string
		media ai.MediaPart
		want  string
	}{
		{
			name:  "image URL",
			media: ai.MediaPart{MIMEType: "image/png", URI: "https://example.com/cat.png"},
			want:  `{"detail":"auto","image_url":"https://example.com/cat.png","type":"input_image"}`,
		},
		{
			name:  "image bytes with alias MIME",
			media: ai.MediaPart{MIMEType: "image/JPG", Data: []byte{0xff, 0xd8, 0xff}},
			want:  `{"detail":"auto","image_url":"data:image/jpeg;base64,/9j/","type":"input_image"}`,
		},
		{
			name:  "image data URI",
			media: ai.MediaPart{MIMEType: "image/webp", URI: "data:image/webp;base64,UklG"},
			want:  `{"detail":"auto","image_url":"data:image/webp;base64,UklG","type":"input_image"}`,
		},
		{
			name:  "PDF URL",
			media: ai.MediaPart{MIMEType: "application/pdf", URI: "https://example.com/a.pdf"},
			want:  `{"file_url":"https://example.com/a.pdf","type":"input_file"}`,
		},
		{
			name:  "named PDF URL",
			media: ai.MediaPart{MIMEType: "application/pdf", URI: "https://example.com/a.pdf", Name: "invoice.pdf"},
			want:  `{"file_url":"https://example.com/a.pdf","filename":"invoice.pdf","type":"input_file"}`,
		},
		{
			name:  "PDF bytes",
			media: ai.MediaPart{MIMEType: "application/pdf", Data: []byte("%PDF")},
			want:  `{"file_data":"data:application/pdf;base64,JVBERg==","filename":"file.pdf","type":"input_file"}`,
		},
		{
			name:  "named PDF data URI",
			media: ai.MediaPart{MIMEType: "application/pdf", URI: "data:application/pdf;base64,JVBERg==", Name: "invoice.pdf"},
			want:  `{"file_data":"data:application/pdf;base64,JVBERg==","filename":"invoice.pdf","type":"input_file"}`,
		},
		{
			name:  "docx bytes named without extension",
			media: ai.MediaPart{MIMEType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document", Data: []byte("PK"), Name: "report"},
			want:  `{"file_data":"data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,UEs=","filename":"report.docx","type":"input_file"}`,
		},
		{
			name:  "xlsx URL",
			media: ai.MediaPart{MIMEType: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", URI: "https://example.com/q.xlsx"},
			want:  `{"file_url":"https://example.com/q.xlsx","type":"input_file"}`,
		},
		{
			name:  "text bytes",
			media: ai.MediaPart{MIMEType: "text/markdown; charset=utf-8", Data: []byte("# Plan"), Name: `a&"b".md`},
			want:  `{"text":"<file name=\"a&amp;&#34;b&#34;.md\" mime_type=\"text/markdown\">\n# Plan\n</file>","type":"input_text"}`,
		},
		{
			name:  "JSON data URI",
			media: ai.MediaPart{MIMEType: "application/json", URI: "data:application/json;base64,eyJhIjoxfQ=="},
			want:  `{"text":"<file mime_type=\"application/json\">\n{\"a\":1}\n</file>","type":"input_text"}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := responsesInputJSON(t, ai.Message{Role: ai.RoleUser, Parts: []ai.ContentPart{
				{Kind: ai.ContentText, Text: "look"},
				mediaPart(tc.media),
				{Kind: ai.ContentText, Text: "thanks"},
			}})
			want := `[{"content":[{"text":"look","type":"input_text"},` + tc.want + `,{"text":"thanks","type":"input_text"}],"role":"user"}]`
			if got != want {
				t.Fatalf("input\n got: %s\nwant: %s", got, want)
			}
		})
	}
}

func TestResponsesTextOnlyMessagesKeepStringContent(t *testing.T) {
	got := responsesInputJSON(t,
		ai.TextMessage(ai.RoleSystem, "rules"),
		ai.Message{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "a"}, {Kind: ai.ContentJSON, JSON: json.RawMessage(`{"b":1}`)}}},
	)
	want := `[{"content":"rules","role":"system"},{"content":"a","role":"user"},{"content":"{\"b\":1}","role":"user"}]`
	if got != want {
		t.Fatalf("input\n got: %s\nwant: %s", got, want)
	}
}

func TestResponsesRejectsUnsupportedMedia(t *testing.T) {
	user := func(media ai.MediaPart) ai.Message {
		return ai.Message{Role: ai.RoleUser, Parts: []ai.ContentPart{mediaPart(media)}}
	}
	image := mediaPart(ai.MediaPart{MIMEType: "image/png", URI: "https://example.com/a.png"})
	for name, message := range map[string]ai.Message{
		"audio":                 user(ai.MediaPart{MIMEType: "audio/mpeg", Data: []byte{1}}),
		"tiff":                  user(ai.MediaPart{MIMEType: "image/tiff", Data: []byte{1}}),
		"s3 scheme":             user(ai.MediaPart{MIMEType: "image/png", URI: "s3://bucket/a.png"}),
		"file scheme PDF":       user(ai.MediaPart{MIMEType: "application/pdf", URI: "file:///tmp/a.pdf"}),
		"hostless URL":          user(ai.MediaPart{MIMEType: "image/png", URI: "https:///a.png"}),
		"mismatched data URI":   user(ai.MediaPart{MIMEType: "image/png", URI: "data:image/gif;base64,R0lG"}),
		"unencoded data URI":    user(ai.MediaPart{MIMEType: "application/pdf", URI: "data:application/pdf,raw"}),
		"text by URL":           user(ai.MediaPart{MIMEType: "text/plain", URI: "https://example.com/a.txt"}),
		"text not UTF-8":        user(ai.MediaPart{MIMEType: "text/plain", Data: []byte{0xff, 0xfe}}),
		"system media":          {Role: ai.RoleSystem, Parts: []ai.ContentPart{image}},
		"assistant media":       {Role: ai.RoleAssistant, Parts: []ai.ContentPart{image}},
		"tool result bad media": toolMessage(ai.ToolResult{ToolCallID: "c", Name: "open_file", Parts: []ai.ContentPart{mediaPart(ai.MediaPart{MIMEType: "video/mp4", Data: []byte{1}})}}),
	} {
		t.Run(name, func(t *testing.T) {
			messages := []ai.Message{message}
			if message.Role == ai.RoleTool {
				messages = []ai.Message{openFileCall("c"), message}
			}
			_, err := buildResponsesParams("gpt-test", ai.AIRequest{Messages: messages})
			var unsupported *ai.UnsupportedContentError
			if !errors.As(err, &unsupported) || unsupported.Provider != "openai" || unsupported.Kind != ai.ContentMedia || !errors.Is(err, ai.ErrUnsupportedCapability) {
				t.Fatalf("error = %v, want *ai.UnsupportedContentError for openai media", err)
			}
		})
	}
}

func openFileCall(id string) ai.Message {
	return ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: id, Type: "function", Name: "open_file", Args: json.RawMessage(`{"id":"f1"}`)}}}}
}

func toolMessage(result ai.ToolResult) ai.Message {
	return ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &result}}}
}

func TestResponsesToolResultOutput(t *testing.T) {
	call := openFileCall("call_1")
	callJSON := `{"arguments":"{\"id\":\"f1\"}","call_id":"call_1","name":"open_file","type":"function_call"}`
	png := ai.MediaPart{MIMEType: "image/png", Data: []byte{0x89, 'P', 'N', 'G'}}
	tests := []struct {
		name   string
		result ai.ToolResult
		output string
	}{
		{
			name:   "text keeps string output",
			result: ai.ToolResult{ToolCallID: "call_1", Name: "open_file", Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "plain "}, {Kind: ai.ContentJSON, JSON: json.RawMessage(`{"ok":true}`)}}},
			output: `"plain {\"ok\":true}"`,
		},
		{
			name:   "error text keeps string output",
			result: ai.ToolResult{ToolCallID: "call_1", Name: "open_file", Parts: ai.TextParts("missing"), IsError: true},
			output: `"{\"error\":\"missing\"}"`,
		},
		{
			name: "media uses content array",
			result: ai.ToolResult{ToolCallID: "call_1", Name: "open_file", Parts: []ai.ContentPart{
				{Kind: ai.ContentText, Text: "receipt.png "},
				{Kind: ai.ContentJSON, JSON: json.RawMessage(`{"size":4}`)},
				mediaPart(png),
				mediaPart(ai.MediaPart{MIMEType: "application/pdf", URI: "https://example.com/r.pdf", Name: "r.pdf"}),
				{Kind: ai.ContentText, Text: "done"},
			}},
			output: `[{"text":"receipt.png {\"size\":4}","type":"input_text"},{"detail":"auto","image_url":"data:image/png;base64,iVBORw==","type":"input_image"},{"file_url":"https://example.com/r.pdf","filename":"r.pdf","type":"input_file"},{"text":"done","type":"input_text"}]`,
		},
		{
			name: "error with media sends error text first",
			result: ai.ToolResult{ToolCallID: "call_1", Name: "open_file", IsError: true, Parts: []ai.ContentPart{
				mediaPart(png), {Kind: ai.ContentText, Text: "partial"},
			}},
			output: `[{"text":"{\"error\":\"partial\"}","type":"input_text"},{"detail":"auto","image_url":"data:image/png;base64,iVBORw==","type":"input_image"}]`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := responsesInputJSON(t, call, toolMessage(tc.result))
			want := `[` + callJSON + `,{"call_id":"call_1","output":` + tc.output + `,"type":"function_call_output"}]`
			if got != want {
				t.Fatalf("input\n got: %s\nwant: %s", got, want)
			}
		})
	}
}

func TestChatCompletionsRejectsMediaWithTypedError(t *testing.T) {
	image := mediaPart(ai.MediaPart{MIMEType: "image/png", URI: "https://example.com/a.png"})
	for name, message := range map[string]ai.Message{
		"user media":        {Role: ai.RoleUser, Parts: []ai.ContentPart{image}},
		"tool result media": toolMessage(ai.ToolResult{ToolCallID: "c", Name: "open_file", Parts: []ai.ContentPart{image}}),
	} {
		t.Run(name, func(t *testing.T) {
			_, err := mapNativeMessages([]ai.Message{message})
			var unsupported *ai.UnsupportedContentError
			if !errors.As(err, &unsupported) || !strings.Contains(err.Error(), "Responses transport") {
				t.Fatalf("error = %v, want typed media error naming the Responses transport", err)
			}
		})
	}
}

// TestResponsesMediaRequestOnTheWire checks the body the SDK sends, including
// the tool-result content array placed through extra fields.
func TestResponsesMediaRequestOnTheWire(t *testing.T) {
	var body []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"a cat","annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()
	provider := New("test-key", nil, WithResponsesTransport())
	provider.baseURL = server.URL
	model, err := provider.Model("gpt-test")
	if err != nil {
		t.Fatal(err)
	}
	secret := []byte("very-secret-image-bytes")
	req := ai.AIRequest{Messages: []ai.Message{
		{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "what is this?"}, mediaPart(ai.MediaPart{MIMEType: "image/png", Data: secret, Name: "cat.png"})}},
		{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "call_1", Type: "function", Name: "open_file", Args: json.RawMessage(`{}`)}}}},
		toolMessage(ai.ToolResult{ToolCallID: "call_1", Name: "open_file", Parts: []ai.ContentPart{mediaPart(ai.MediaPart{MIMEType: "application/pdf", URI: "https://files.example.com/a.pdf?X-Amz-Signature=sig"})}}),
	}}
	recorder := obstest.Install(t)
	ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{Prompt: gai.CaptureEnabled, Completion: gai.CaptureEnabled, MaxBytes: gai.MaxContentCaptureBytes})
	response, err := model.(ai.ModelGenerator).Generate(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if response.Text() != "a cat" {
		t.Fatalf("response text = %q", response.Text())
	}
	var sent struct {
		Input []json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(body, &sent); err != nil {
		t.Fatalf("decode request %s: %v", body, err)
	}
	if len(sent.Input) != 3 {
		t.Fatalf("input items = %d in %s", len(sent.Input), body)
	}
	if want := `{"content":[{"text":"what is this?","type":"input_text"},{"detail":"auto","image_url":"data:image/png;base64,dmVyeS1zZWNyZXQtaW1hZ2UtYnl0ZXM=","type":"input_image"}],"role":"user"}`; string(sent.Input[0]) != want {
		t.Fatalf("user item = %s, want %s", sent.Input[0], want)
	}
	if want := `{"call_id":"call_1","output":[{"file_url":"https://files.example.com/a.pdf?X-Amz-Signature=sig","type":"input_file"}],"type":"function_call_output"}`; string(sent.Input[2]) != want {
		t.Fatalf("tool output item = %s, want %s", sent.Input[2], want)
	}

	// Captured prompt content keeps the text but never the media bytes or URL.
	attrs := obstest.Attributes(obstest.RequireGenerationSpans(t, recorder, 1)[0])
	input := attrs["langfuse.observation.input"].AsString()
	if !strings.Contains(input, "what is this?") {
		t.Fatalf("captured input lacks prompt text: %s", input)
	}
	for _, leak := range []string{"dmVyeS1zZWNyZXQtaW1hZ2UtYnl0ZXM", "very-secret", "X-Amz-Signature", "files.example.com"} {
		for key, value := range attrs {
			if strings.Contains(value.String(), leak) {
				t.Fatalf("span attribute %s leaks %q: %s", key, leak, value.String())
			}
		}
	}
}

// TestTokenCounterEstimatesMediaRequests checks that a request budget with
// the OpenAI counter neither tokenizes media bytes nor fails on media.
func TestTokenCounterEstimatesMediaRequests(t *testing.T) {
	counter, err := NewTokenCounter(GPT56Luna)
	if err != nil {
		t.Fatal(err)
	}
	image := make([]byte, 4<<20)
	for i := range image {
		image[i] = byte(i * 31)
	}
	req := ai.AIRequest{Messages: []ai.Message{
		{Role: ai.RoleUser, Parts: []ai.ContentPart{
			{Kind: ai.ContentText, Text: "what is in these files?"},
			mediaPart(ai.MediaPart{MIMEType: "image/png", Data: image, Name: "photo.png"}),
			mediaPart(ai.MediaPart{MIMEType: "application/pdf", URI: "https://example.com/a.pdf"}),
			mediaPart(ai.MediaPart{MIMEType: "text/csv", Data: []byte("a,b\n1,2\n")}),
		}},
		openFileCall("call_1"),
		toolMessage(ai.ToolResult{ToolCallID: "call_1", Name: "open_file", Parts: []ai.ContentPart{mediaPart(ai.MediaPart{MIMEType: "image/jpeg", Data: image})}}),
	}}
	result, err := ai.EstimateRequestTokens(t.Context(), req, counter)
	if err != nil {
		t.Fatal(err)
	}
	// Two images (1,600 each) and a PDF by URL (10 pages at 1,500) dominate;
	// the 4 MiB of image bytes would be millions of tokens if tokenized.
	if result.MessageTokens < 18_200 || result.MessageTokens > 18_600 {
		t.Fatalf("message tokens = %d, want about 18,200 plus envelopes", result.MessageTokens)
	}
}

func TestAcceptsMediaFollowsTransport(t *testing.T) {
	image := ai.MediaPart{MIMEType: "image/png", Data: []byte{1}}
	responsesModel, err := New("k", nil, WithResponsesTransport()).TypedModel("gpt-test")
	if err != nil {
		t.Fatal(err)
	}
	chatModel, err := New("k", nil).TypedModel("gpt-test")
	if err != nil {
		t.Fatal(err)
	}
	var acceptor ai.MediaAcceptor = responsesModel
	if err := acceptor.AcceptsMedia(image, ai.MediaInToolResult); err != nil {
		t.Fatalf("Responses image: %v", err)
	}
	for name, check := range map[string]error{
		"chat transport": chatModel.AcceptsMedia(image, ai.MediaInUserMessage),
		"audio":          responsesModel.AcceptsMedia(ai.MediaPart{MIMEType: "audio/mpeg", Data: []byte{1}}, ai.MediaInUserMessage),
		"video URL":      responsesModel.AcceptsMedia(ai.MediaPart{MIMEType: "video/mp4", URI: "https://example.com/a.mp4"}, ai.MediaInUserMessage),
	} {
		if !errors.Is(check, ai.ErrUnsupportedCapability) {
			t.Errorf("%s: err = %v", name, check)
		}
	}
}
