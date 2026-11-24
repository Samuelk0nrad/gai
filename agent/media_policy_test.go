package agent_test

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/Samuelk0nrad/gai/agent"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/loop"
	"github.com/Samuelk0nrad/gai/modelinput"
)

// pngOnlyModel is a durableModel whose adapter accepts only image/png.
type pngOnlyModel struct{ *durableModel }

func (pngOnlyModel) AcceptsMedia(media ai.MediaPart, _ ai.MediaPlacement) error {
	if media.MIMEType == "image/png" {
		return nil
	}
	return &ai.UnsupportedContentError{Provider: "test", Kind: ai.ContentMedia, Detail: media.MIMEType}
}

const docxMIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

func docxTool(t *testing.T) loop.Tool {
	t.Helper()
	tool, err := loop.NewResultTool("open_file", "Open a stored file.", ai.ToolParameters{}, func(context.Context, ai.ToolCall) (loop.ToolResult, error) {
		return loop.ToolResult{Text: "report.docx", Media: []ai.MediaPart{
			{MIMEType: docxMIME, Data: []byte("docx-bytes"), Name: "report.docx"},
			{MIMEType: "image/png", Data: []byte("png")},
		}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return tool
}

// textOfDocx converts .docx files to text and declines everything else.
func textOfDocx(calls *atomic.Int32) ai.MediaConverter {
	return ai.MediaConverterFunc(func(_ context.Context, media ai.MediaPart, _ func(ai.MediaPart) error) ([]ai.ContentPart, error) {
		calls.Add(1)
		if media.MIMEType != docxMIME {
			return nil, ai.ErrMediaNotConverted
		}
		return ai.TextParts("<text of " + media.Name + ">"), nil
	})
}

func TestMediaPolicyConvertsToolAndUserMedia(t *testing.T) {
	model := pngOnlyModel{&durableModel{responses: [][]ai.Token{calls("open_file"), response("done")}}}
	var conversions atomic.Int32
	def := durableDefinition(model, docxTool(t))
	def.Media = ai.MediaPolicy{Converter: textOfDocx(&conversions)}
	input := agent.RunInput{Prompt: modelinput.PromptInput{User: []ai.ContentPart{
		{Kind: ai.ContentText, Text: "summarize"},
		{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: docxMIME, Data: []byte("brief"), Name: "brief.docx"}},
	}}}
	w, err := agent.New(def).NewRun(t.Context(), input)
	if err != nil {
		t.Fatal(err)
	}
	result, err := w.Run(t.Context())
	if err != nil || result.Status != agent.StatusSucceeded {
		t.Fatalf("result = %+v, error = %v", result, err)
	}
	// The user's .docx is converted once although two requests carry it.
	for i, request := range model.requests {
		user := request.Messages[1]
		if user.Role != ai.RoleUser || user.Text() != "summarize<text of brief.docx>" {
			t.Fatalf("request %d user message = %+v", i, user)
		}
	}
	text, media := toolResultMedia(t, model.requests[1].Messages)
	if text != "report.docx\n<text of report.docx>" || len(media) != 1 || media[0].MIMEType != "image/png" {
		t.Fatalf("tool result text %q media %+v", text, media)
	}
	if conversions.Load() != 2 {
		t.Fatalf("conversions = %d, want 2", conversions.Load())
	}
}

func TestUnconvertedToolMediaBecomesToolError(t *testing.T) {
	model := pngOnlyModel{&durableModel{responses: [][]ai.Token{calls("open_file"), response("sorry")}}}
	w, err := agent.New(durableDefinition(model, docxTool(t))).NewRun(t.Context(), textRunInput("open it"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := w.Run(t.Context())
	if err != nil || result.Status != agent.StatusSucceeded {
		t.Fatalf("result = %+v, error = %v", result, err)
	}
	results := model.requests[1].Messages[len(model.requests[1].Messages)-1].ToolResults()
	if len(results) != 1 || !results[0].IsError || !strings.Contains(results[0].Text(), "[file: report.docx, "+docxMIME+"]") {
		t.Fatalf("tool results = %+v", results)
	}
}

func TestUnsupportedUserMediaFailsBeforeTheRequest(t *testing.T) {
	model := pngOnlyModel{&durableModel{responses: [][]ai.Token{response("unused")}}}
	def := durableDefinition(model)
	def.Media = ai.MediaPolicy{Converter: ai.MediaConverterFunc(func(context.Context, ai.MediaPart, func(ai.MediaPart) error) ([]ai.ContentPart, error) {
		return nil, errors.New("converter down")
	})}
	input := agent.RunInput{Prompt: modelinput.PromptInput{User: []ai.ContentPart{
		{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "audio/mpeg", Data: []byte{1}}},
	}}}
	w, err := agent.New(def).NewRun(t.Context(), input)
	if err != nil {
		t.Fatal(err)
	}
	_, err = w.Run(t.Context())
	var conversion *ai.MediaConversionError
	if !errors.As(err, &conversion) || model.count() != 0 {
		t.Fatalf("error = %v, requests = %d", err, model.count())
	}
}
