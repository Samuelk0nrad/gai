package ai

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestRequestEstimateCountsMediaWithoutEncodedBytes(t *testing.T) {
	image := bytes.Repeat([]byte{0x89}, 1<<20)
	pdf := []byte("%PDF-1.7 1 0 obj << /Type /Pages /Kids [2 0 R 3 0 R 4 0 R] /Count 3 >> endobj 2 0 obj << /Type /Page >> endobj 3 0 obj << /Type/Page >> endobj")
	compressedPDF := bytes.Repeat([]byte{0}, 3*mediaPDFBytesPerPage+1)
	media := func(m MediaPart) ContentPart { return ContentPart{Kind: ContentMedia, Media: &m} }
	tests := []struct {
		name  string
		part  ContentPart
		extra int
	}{
		{"image bytes", media(MediaPart{MIMEType: "image/png", Data: image, Name: "cat.png"}), mediaImageTokens},
		{"image URL", media(MediaPart{MIMEType: "image/jpeg", URI: "https://example.com/cat.jpg"}), mediaImageTokens},
		{"PDF page tree", media(MediaPart{MIMEType: "application/pdf", Data: pdf}), 3 * mediaPDFPageTokens},
		{"compressed PDF", media(MediaPart{MIMEType: "application/pdf", Data: compressedPDF}), 4 * mediaPDFPageTokens},
		{"PDF URL", media(MediaPart{MIMEType: "application/pdf", URI: "https://example.com/a.pdf"}), mediaPDFUnknownPages * mediaPDFPageTokens},
		{"text bytes", media(MediaPart{MIMEType: "text/plain; charset=utf-8", Data: []byte("hello file")}), len("hello file")},
		{"other bytes", media(MediaPart{MIMEType: "audio/mpeg", Data: bytes.Repeat([]byte{1}, 4*(mediaImageTokens+10))}), mediaImageTokens + 10},
		{"tool result media", ContentPart{Kind: ContentToolResult, ToolResult: &ToolResult{ToolCallID: "c", Name: "open_file", Parts: []ContentPart{
			{Kind: ContentText, Text: "here"}, media(MediaPart{MIMEType: "image/webp", Data: image}),
		}}}, mediaImageTokens},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			message := Message{Role: RoleUser, Parts: []ContentPart{{Kind: ContentText, Text: "look"}, tc.part}}
			if tc.part.Kind == ContentToolResult {
				message.Role = RoleTool
				message.Parts = message.Parts[1:]
			}
			before, _ := json.Marshal(message)
			counter := &requestTestCounter{}
			got, err := EstimateMessageTokens(t.Context(), []Message{message}, counter)
			if err != nil {
				t.Fatal(err)
			}
			envelope, _ := json.Marshal(Message{Role: message.Role, Parts: withoutMediaData(message.Parts)})
			if want := len(envelope) + tc.extra; got != want {
				t.Fatalf("tokens = %d, want envelope %d + media %d", got, len(envelope), tc.extra)
			}
			for _, text := range counter.texts {
				if strings.Contains(text, `"data"`) {
					t.Fatalf("counted encoded media bytes: %.80s", text)
				}
			}
			if after, _ := json.Marshal(message); !bytes.Equal(before, after) {
				t.Fatal("estimate modified the message")
			}
		})
	}
}

func TestIsTextMIME(t *testing.T) {
	for mimeType, want := range map[string]bool{
		"text/plain": true, "TEXT/CSV": true, "text/markdown; charset=utf-8": true,
		"application/json": true, "application/ld+json": true,
		"application/pdf": false, "image/png": false, "": false,
	} {
		if got := IsTextMIME(mimeType); got != want {
			t.Errorf("IsTextMIME(%q) = %v, want %v", mimeType, got, want)
		}
	}
}

func TestMediaPlaceholder(t *testing.T) {
	for want, media := range map[string]MediaPart{
		"[image: board.jpg, image/png]":       {MIMEType: "image/png", Data: []byte{1}, Name: "board.jpg"},
		"[file: report.pdf, application/pdf]": {MIMEType: "application/pdf", URI: "https://example.com/secret", Name: " report.pdf "},
		"[file: text/csv]":                    {MIMEType: "text/csv; charset=utf-8", Data: []byte("a,b")},
		"[audio: audio/mpeg]":                 {MIMEType: "Audio/MPEG", URI: "https://example.com/a.mp3"},
		"[video: two lines.mp4, video/mp4]":   {MIMEType: "video/mp4", Data: []byte{1}, Name: "two\nlines.mp4"},
	} {
		if got := media.Placeholder(); got != want {
			t.Errorf("Placeholder() = %q, want %q", got, want)
		}
	}
}
