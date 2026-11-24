package ai_test

import (
	"context"
	"errors"
	"reflect"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Samuelk0nrad/gai/ai"
)

// pngOnlyModel accepts only image/png media, and nothing in tool results
// except PNG either.
type pngOnlyModel struct{}

func (pngOnlyModel) GenerateStream(context.Context, ai.AIRequest) <-chan ai.Token { return nil }

func (pngOnlyModel) AcceptsMedia(media ai.MediaPart, _ ai.MediaPlacement) error {
	if media.MIMEType == "image/png" {
		return nil
	}
	return &ai.UnsupportedContentError{Provider: "test", Kind: ai.ContentMedia, Detail: media.MIMEType}
}

func mediaContent(media ai.MediaPart) ai.ContentPart {
	return ai.ContentPart{Kind: ai.ContentMedia, Media: &media}
}

func TestMediaAdapterConvertsOnceAndKeepsAcceptedMedia(t *testing.T) {
	var calls atomic.Int32
	converter := ai.MediaConverterFunc(func(_ context.Context, media ai.MediaPart, accepts func(ai.MediaPart) error) ([]ai.ContentPart, error) {
		calls.Add(1)
		if media.MIMEType != "image/heic" {
			return nil, ai.ErrMediaNotConverted
		}
		jpeg := ai.MediaPart{MIMEType: "image/jpeg", Data: []byte{1}}
		if accepts(jpeg) == nil {
			return []ai.ContentPart{mediaContent(jpeg)}, nil
		}
		png := ai.MediaPart{MIMEType: "image/png", Data: []byte{2}}
		return []ai.ContentPart{{Kind: ai.ContentText, Text: "converted photo:"}, mediaContent(png)}, nil
	})
	adapter := ai.NewMediaAdapter(pngOnlyModel{}, ai.MediaPolicy{Converter: converter})
	heic := mediaContent(ai.MediaPart{MIMEType: "image/heic", Data: []byte{9}, Name: "photo.heic"})
	png := mediaContent(ai.MediaPart{MIMEType: "image/png", Data: []byte{3}})
	messages := []ai.Message{
		{Role: ai.RoleSystem, Parts: ai.TextParts("system")},
		{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "look"}, heic, png}},
	}
	for range 3 {
		got, err := adapter.Messages(t.Context(), messages)
		if err != nil {
			t.Fatal(err)
		}
		want := []ai.ContentPart{{Kind: ai.ContentText, Text: "look"}, {Kind: ai.ContentText, Text: "converted photo:"}, mediaContent(ai.MediaPart{MIMEType: "image/png", Data: []byte{2}}), png}
		if !reflect.DeepEqual(got[1].Parts, want) {
			t.Fatalf("parts = %#v", got[1].Parts)
		}
		if !reflect.DeepEqual(got[0], messages[0]) {
			t.Fatal("system message changed")
		}
	}
	if calls.Load() != 1 {
		t.Fatalf("converter calls = %d, want 1", calls.Load())
	}
	if messages[1].Parts[1].Media.MIMEType != "image/heic" {
		t.Fatal("input modified")
	}
}

func TestMediaAdapterToolResultsAndFallback(t *testing.T) {
	docx := ai.MediaPart{MIMEType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document", Data: []byte{1}, Name: "report.docx"}
	message := ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{
		ToolCallID: "c1", Name: "open", Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "file:"}, mediaContent(docx)},
	}}}}
	// The default fallback keeps the media, so the adapter's error surfaces.
	_, err := ai.NewMediaAdapter(pngOnlyModel{}, ai.MediaPolicy{}).Messages(t.Context(), []ai.Message{message})
	var unsupported *ai.UnsupportedContentError
	if !errors.As(err, &unsupported) {
		t.Fatalf("err = %v", err)
	}
	got, err := ai.NewMediaAdapter(pngOnlyModel{}, ai.MediaPolicy{Fallback: ai.MediaFallbackPlaceholder}).Messages(t.Context(), []ai.Message{message})
	if err != nil {
		t.Fatal(err)
	}
	parts := got[0].Parts[0].ToolResult.Parts
	if len(parts) != 2 || parts[1].Text != "[file: report.docx, application/vnd.openxmlformats-officedocument.wordprocessingml.document]" {
		t.Fatalf("parts = %#v", parts)
	}
}

func TestMediaAdapterConverterErrors(t *testing.T) {
	audio := mediaContent(ai.MediaPart{MIMEType: "audio/mpeg", URI: "https://secret.example/a.mp3?sig=x"})
	messages := []ai.Message{{Role: ai.RoleUser, Parts: []ai.ContentPart{audio}}}
	failing := ai.MediaConverterFunc(func(context.Context, ai.MediaPart, func(ai.MediaPart) error) ([]ai.ContentPart, error) {
		return nil, errors.New("transcriber down")
	})
	_, err := ai.NewMediaAdapter(pngOnlyModel{}, ai.MediaPolicy{Converter: failing, Fallback: ai.MediaFallbackPlaceholder}).Messages(t.Context(), messages)
	var conversion *ai.MediaConversionError
	if !errors.As(err, &conversion) || strings.Contains(err.Error(), "secret") {
		t.Fatalf("err = %v", err)
	}
	// Converted media must itself be accepted.
	bad := ai.MediaConverterFunc(func(context.Context, ai.MediaPart, func(ai.MediaPart) error) ([]ai.ContentPart, error) {
		return []ai.ContentPart{mediaContent(ai.MediaPart{MIMEType: "audio/wav", Data: []byte{1}})}, nil
	})
	if _, err := ai.NewMediaAdapter(pngOnlyModel{}, ai.MediaPolicy{Converter: bad}).Messages(t.Context(), messages); !errors.As(err, &conversion) {
		t.Fatalf("err = %v", err)
	}
}

func TestMediaAdapterWithoutAcceptorChangesNothing(t *testing.T) {
	messages := []ai.Message{{Role: ai.RoleUser, Parts: []ai.ContentPart{mediaContent(ai.MediaPart{MIMEType: "audio/mpeg", Data: []byte{1}})}}}
	got, err := ai.NewMediaAdapter(nil, ai.MediaPolicy{Fallback: ai.MediaFallbackPlaceholder}).Messages(t.Context(), messages)
	if err != nil || !reflect.DeepEqual(got, messages) {
		t.Fatalf("got %#v, %v", got, err)
	}
}

func TestMediaAdapterSharesConcurrentConversions(t *testing.T) {
	var calls atomic.Int32
	release := make(chan struct{})
	converter := ai.MediaConverterFunc(func(context.Context, ai.MediaPart, func(ai.MediaPart) error) ([]ai.ContentPart, error) {
		calls.Add(1)
		<-release
		return ai.TextParts("transcript"), nil
	})
	adapter := ai.NewMediaAdapter(pngOnlyModel{}, ai.MediaPolicy{Converter: converter})
	parts := []ai.ContentPart{mediaContent(ai.MediaPart{MIMEType: "audio/mpeg", Data: []byte{1, 2}})}
	const callers = 8
	results := make(chan []ai.ContentPart, callers)
	errs := make(chan error, callers)
	for range callers {
		go func() {
			got, err := adapter.Parts(t.Context(), parts, ai.MediaInToolResult)
			results <- got
			errs <- err
		}()
	}
	// Let every caller reach the adapter before the conversion finishes.
	for calls.Load() == 0 {
		runtime.Gosched()
	}
	time.Sleep(10 * time.Millisecond)
	close(release)
	for range callers {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
		if got := <-results; len(got) != 1 || got[0].Text != "transcript" {
			t.Fatalf("parts = %#v", got)
		}
	}
	if calls.Load() != 1 {
		t.Fatalf("converter calls = %d, want 1", calls.Load())
	}
}

func TestMediaAdapterEmptyConversionRemovesMedia(t *testing.T) {
	empty := ai.MediaConverterFunc(func(context.Context, ai.MediaPart, func(ai.MediaPart) error) ([]ai.ContentPart, error) {
		return nil, nil
	})
	parts := []ai.ContentPart{{Kind: ai.ContentText, Text: "see"}, mediaContent(ai.MediaPart{MIMEType: "application/zip", Data: []byte{1}})}
	got, err := ai.NewMediaAdapter(pngOnlyModel{}, ai.MediaPolicy{Converter: empty}).Parts(t.Context(), parts, ai.MediaInUserMessage)
	if err != nil || len(got) != 1 || got[0].Text != "see" {
		t.Fatalf("parts = %#v, %v", got, err)
	}
}

func TestMediaAdapterKeepsRequiredExtensions(t *testing.T) {
	part := mediaContent(ai.MediaPart{MIMEType: "audio/mpeg", Data: []byte{1}})
	part.Extensions = []ai.Extension{{Namespace: "x", Type: "state", Data: []byte(`1`), Required: true}}
	_, err := ai.NewMediaAdapter(pngOnlyModel{}, ai.MediaPolicy{Fallback: ai.MediaFallbackPlaceholder}).Parts(t.Context(), []ai.ContentPart{part}, ai.MediaInUserMessage)
	var unsupported *ai.UnsupportedContentError
	if !errors.As(err, &unsupported) || unsupported.Kind != ai.ContentExtension {
		t.Fatalf("err = %v", err)
	}
}
