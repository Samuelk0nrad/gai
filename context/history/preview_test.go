package history_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/context/history"
)

func TestToolResultPreviewUsesOneMarkerAcrossTextParts(t *testing.T) {
	t.Parallel()
	const marker = "\n[tool result truncated]"
	manyTexts := []string{strings.Repeat("界", 251), strings.Repeat("語", 249), "overflow"}
	manyWant := []string{manyTexts[0], manyTexts[1], marker}
	for i := 0; i < 64; i++ {
		manyTexts = append(manyTexts, "later text")
		manyWant = append(manyWant, "")
	}
	for _, tt := range []struct {
		name    string
		text    []string
		want    []string
		markers int
	}{
		{
			name: "exact boundary without omission",
			text: []string{strings.Repeat("界", 500), ""},
			want: []string{strings.Repeat("界", 500), ""},
		},
		{
			name:    "overflow followed by tail",
			markers: 1,
			text:    []string{strings.Repeat("界", 501), "tail"},
			want:    []string{strings.Repeat("界", 500) + marker, ""},
		},
		{
			name:    "exact boundary followed by tails",
			markers: 1,
			text:    []string{strings.Repeat("界", 500), "tail", "later"},
			want:    []string{strings.Repeat("界", 500), marker, ""},
		},
		{
			name:    "many segmented parts",
			markers: 1,
			text:    manyTexts,
			want:    manyWant,
		},
		{
			name:    "invalid UTF-8 in truncated prefix",
			markers: 1,
			text:    []string{strings.Repeat("a", 499) + "\xff" + "tail"},
			want:    []string{strings.Repeat("a", 499) + "\ufffd" + marker},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			parts := make([]ai.ContentPart, len(tt.text))
			for i, text := range tt.text {
				parts[i] = ai.ContentPart{Kind: ai.ContentText, Text: text}
			}
			source := previewHistoryPart(parts)
			before := ai.CloneMessages(source.Messages)
			messages := source.ConversationMessages()
			gotParts := messages[0].Parts[0].ToolResult.Parts
			if len(gotParts) != len(tt.want) {
				t.Fatalf("preview changed part ordering/count: got %d, want %d", len(gotParts), len(tt.want))
			}
			var rendered strings.Builder
			for _, part := range gotParts {
				rendered.WriteString(part.Text)
			}
			if got := strings.Count(rendered.String(), marker); got != tt.markers {
				t.Errorf("preview contains %d truncation markers, want %d", got, tt.markers)
			}
			if got := len([]rune(rendered.String())); got != 500+tt.markers*len([]rune(marker)) {
				t.Errorf("preview has %d runes, want 500 content runes and %d markers", got, tt.markers)
			}
			for i, want := range tt.want {
				if gotParts[i].Text != want {
					t.Errorf("part %d text = %q, want %q", i, gotParts[i].Text, want)
					break
				}
			}
			if !reflect.DeepEqual(source.Messages, before) {
				t.Fatal("preview changed the source history")
			}
		})
	}
}

func TestToolResultPreviewPreservesNonPlainPartsAndSource(t *testing.T) {
	t.Parallel()
	const marker = "\n[tool result truncated]"
	protected := []ai.ContentPart{
		{Kind: ai.ContentJSON, JSON: json.RawMessage(`{"complete":"value"}`)},
		{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", Data: []byte{1, 2, 3}}},
		{Kind: ai.ContentText, Text: strings.Repeat("signed", 150), Extensions: []ai.Extension{{Namespace: "test", Type: "signature", Data: json.RawMessage(`"opaque"`), Required: true}}},
		{Kind: ai.ContentExtension, Extensions: []ai.Extension{{Namespace: "test", Type: "continuation", Data: json.RawMessage(`{"state":1}`), Required: true}}},
	}
	parts := ai.TextParts(strings.Repeat("界", 501))
	parts = append(parts, protected...)
	parts = append(parts, ai.TextParts("tail")...)
	parts = append(parts, ai.TextParts("last tail")...)
	source := previewHistoryPart(parts)
	before := ai.CloneMessages(source.Messages)
	messages := source.ConversationMessages()
	got := messages[0].Parts[0].ToolResult.Parts
	if got[0].Text != strings.Repeat("界", 500)+marker || got[5].Text != "" || got[6].Text != "" {
		t.Error("plain text was not bounded to one preview followed by empty plain text parts")
	}
	if !reflect.DeepEqual(got[1:5], protected) {
		t.Error("preview changed JSON, media, or extension-bearing parts")
	}
	if !reflect.DeepEqual(source.Messages, before) {
		t.Fatal("preview changed the source history")
	}
	// The preserved mutable payloads must also belong to the projection itself.
	got[1].JSON[2] = 'X'
	got[2].Media.Data[0] = 9
	got[3].Extensions[0].Data[1] = 'X'
	got[4].Extensions[0].Data[2] = 'X'
	if !reflect.DeepEqual(source.Messages, before) {
		t.Fatal("projection aliases protected source payloads")
	}
}

func previewHistoryPart(parts []ai.ContentPart) *history.Part {
	return &history.Part{Messages: []ai.Message{{Role: ai.RoleTool, Parts: []ai.ContentPart{{
		Kind:       ai.ContentToolResult,
		ToolResult: &ai.ToolResult{ToolCallID: "call-preview", Name: "search", Parts: parts},
	}}}}}
}
