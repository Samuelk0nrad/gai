package agent_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/agent"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/obstest"
	"github.com/Samuelk0nrad/gai/loop"
)

var receiptPNG = []byte("\x89PNG-receipt-secret-bytes")

func openFileTool(t *testing.T, runs *atomic.Int32) loop.Tool {
	t.Helper()
	tool, err := loop.NewResultTool("open_file", "Open a stored file.", ai.ToolParameters{}, func(context.Context, ai.ToolCall) (loop.ToolResult, error) {
		runs.Add(1)
		return loop.ToolResult{Text: "receipt.png", Media: []ai.MediaPart{
			{MIMEType: "image/png", Data: append([]byte(nil), receiptPNG...), Name: "receipt.png"},
			{MIMEType: "application/pdf", URI: "https://files.example.test/r.pdf?X-Amz-Signature=secret"},
		}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	// Registration options wrap the tool; the wrapper must keep its media.
	tool, err = loop.WithToolOptions(tool, loop.ToolOptions{Serial: true})
	if err != nil {
		t.Fatal(err)
	}
	return tool
}

// toolResultMedia returns the media parts of the tool result in messages.
func toolResultMedia(t *testing.T, messages []ai.Message) (string, []ai.MediaPart) {
	t.Helper()
	for _, message := range messages {
		results := message.ToolResults()
		if len(results) == 0 {
			continue
		}
		result := results[0]
		var media []ai.MediaPart
		for _, part := range result.Parts {
			if part.Media != nil {
				media = append(media, *part.Media)
			}
		}
		return result.Text(), media
	}
	t.Fatalf("no tool result in %+v", messages)
	return "", nil
}

func requireOpenFileMedia(t *testing.T, messages []ai.Message) {
	t.Helper()
	text, media := toolResultMedia(t, messages)
	if text != "receipt.png" || len(media) != 2 || media[0].MIMEType != "image/png" || media[0].Name != "receipt.png" || !bytes.Equal(media[0].Data, receiptPNG) || media[1].URI != "https://files.example.test/r.pdf?X-Amz-Signature=secret" {
		t.Fatalf("tool result text %q media %+v", text, media)
	}
}

func TestResultToolMediaReachesModelAndSurvivesResume(t *testing.T) {
	var runs atomic.Int32
	tool := openFileTool(t, &runs)
	firstModel := &durableModel{responses: [][]ai.Token{calls("open_file")}}
	def := durableDefinition(firstModel, tool)
	store := &checkpointStore{}
	w := newDurableRun(t, agent.New(def), store)
	// Pause once the processed result is saved, before the next model call.
	store.hook = func(cp agent.Checkpoint) error {
		if p := pending(cp); p != nil && len(p.Invocations) == 1 && p.Invocations[0].Result != nil {
			return w.RequestPause()
		}
		return nil
	}
	result := runDurable(t, w)
	if result.Status != agent.StatusPaused || result.Checkpoint == nil {
		t.Fatalf("result = %+v", result)
	}
	checkpoint := store.load()
	if err := checkpoint.Validate(); err != nil {
		t.Fatal(err)
	}
	saved := savedToolResult(t, checkpoint)
	requireOpenFileMedia(t, []ai.Message{{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: saved}}}})

	nextModel := &durableModel{responses: [][]ai.Token{response("a receipt")}}
	def.Model = nextModel
	store.hook = nil
	resumed, err := agent.New(def).Resume(context.Background(), checkpoint, agent.ResumeInput{}, store.durability())
	if err != nil {
		t.Fatal(err)
	}
	final, err := resumed.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if final.Status != agent.StatusSucceeded || final.Text != "a receipt" || nextModel.count() != 1 || runs.Load() != 1 {
		t.Fatalf("final = %+v", final)
	}
	requireOpenFileMedia(t, nextModel.requests[0].Messages)
	requireOpenFileMedia(t, final.Primary.Messages)
}

// savedToolResult returns the only processed tool result in a checkpoint,
// pending or already accepted.
func savedToolResult(t *testing.T, cp agent.Checkpoint) *ai.ToolResult {
	t.Helper()
	for _, stage := range cp.Stages {
		if stage.Continuation == nil {
			continue
		}
		generations := stage.Continuation.Generations
		if stage.Continuation.Pending != nil {
			generations = append(generations, *stage.Continuation.Pending)
		}
		for _, generation := range generations {
			for _, inv := range generation.Invocations {
				if inv.Result != nil {
					return inv.Result
				}
			}
		}
	}
	t.Fatal("no saved tool result")
	return nil
}

func TestResultToolMediaIsNotCapturedInTraces(t *testing.T) {
	recorder := obstest.Install(t)
	model := &durableModel{responses: [][]ai.Token{calls("open_file"), response("done")}}
	ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{
		Prompt: gai.CaptureEnabled, Completion: gai.CaptureEnabled, ToolInput: gai.CaptureEnabled, ToolOutput: gai.CaptureEnabled,
		MaxBytes: gai.MaxContentCaptureBytes,
	})
	w, err := agent.New(durableDefinition(model, openFileTool(t, new(atomic.Int32)))).NewRun(ctx, textRunInput("open it"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := w.Run(ctx)
	if err != nil || result.Status != agent.StatusSucceeded {
		t.Fatalf("result = %+v, error = %v", result, err)
	}
	requireOpenFileMedia(t, model.requests[1].Messages)

	var toolOutput string
	for _, span := range recorder.Ended() {
		for key, value := range obstest.Attributes(span) {
			text := value.String()
			for _, leak := range []string{"receipt-secret", base64.StdEncoding.EncodeToString(receiptPNG), "X-Amz-Signature", "files.example.test"} {
				if strings.Contains(text, leak) {
					t.Fatalf("span %q attribute %s leaks %q: %s", span.Name(), key, leak, text)
				}
			}
			if key == "langfuse.observation.output" && strings.Contains(text, "[image:") {
				toolOutput = text
			}
		}
	}
	want := "receipt.png\n[image: receipt.png, image/png, " + strconv.Itoa(len(receiptPNG)) + " bytes]\n[file: application/pdf]"
	if toolOutput != want {
		t.Fatalf("captured tool output = %q, want %q", toolOutput, want)
	}
}

func TestResultToolRejectsInvalidMedia(t *testing.T) {
	tool, err := loop.NewResultTool("open_file", "Open a stored file.", ai.ToolParameters{}, func(context.Context, ai.ToolCall) (loop.ToolResult, error) {
		return loop.ToolResult{Media: []ai.MediaPart{{MIMEType: "image/png"}}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	result := loop.CallTool(t.Context(), ai.ToolCall{ID: "a", Type: "function", Name: "open_file", Args: []byte(`{}`)}, []loop.Tool{tool})
	if result.Err == nil || len(result.Media) != 0 {
		t.Fatalf("invalid media accepted: %+v", result)
	}
	failing, err := loop.NewResultTool("fail", "Fail.", ai.ToolParameters{}, func(context.Context, ai.ToolCall) (loop.ToolResult, error) {
		return loop.ToolResult{Text: "ignored", Media: []ai.MediaPart{{MIMEType: "image/png", Data: []byte{1}}}}, errors.New("not found")
	})
	if err != nil {
		t.Fatal(err)
	}
	result = loop.CallTool(t.Context(), ai.ToolCall{ID: "a", Type: "function", Name: "fail", Args: []byte(`{}`)}, []loop.Tool{failing})
	if result.Err == nil || result.Text != "" || result.Media != nil {
		t.Fatalf("error result kept output: %+v", result)
	}
	if text, err := failing.Function(t.Context(), ai.ToolCall{ID: "a", Type: "function", Name: "fail", Args: []byte(`{}`)}); err == nil || text != "" {
		t.Fatalf("Function = %q, %v", text, err)
	}
}
