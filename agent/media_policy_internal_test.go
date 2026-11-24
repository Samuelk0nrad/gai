package agent

import (
	"context"
	"errors"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
)

type recordingAcceptorModel struct{ requests []ai.AIRequest }

func (m *recordingAcceptorModel) GenerateStream(_ context.Context, request ai.AIRequest) <-chan ai.Token {
	m.requests = append(m.requests, request)
	tokens := make(chan ai.Token)
	close(tokens)
	return tokens
}

func (*recordingAcceptorModel) AcceptsMedia(media ai.MediaPart, _ ai.MediaPlacement) error {
	if media.MIMEType == "image/png" {
		return nil
	}
	return &ai.UnsupportedContentError{Provider: "test", Kind: ai.ContentMedia, Detail: media.MIMEType}
}

// Compaction summaries call the model directly, so the wrapper must apply
// the media policy to them.
func TestMediaPolicyModelAdaptsCompactionRequests(t *testing.T) {
	inner := &recordingAcceptorModel{}
	heic := ai.MediaPart{MIMEType: "image/heic", Data: []byte{1}, Name: "a.heic"}
	request := ai.AIRequest{Messages: []ai.Message{{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentMedia, Media: &heic}}}}}

	model := mediaPolicyModel{Model: inner, media: ai.NewMediaAdapter(inner, ai.MediaPolicy{Fallback: ai.MediaFallbackPlaceholder})}
	for range model.GenerateStream(t.Context(), request) {
	}
	if len(inner.requests) != 1 || inner.requests[0].Messages[0].Text() != "[image: a.heic, image/heic]" {
		t.Fatalf("requests = %+v", inner.requests)
	}

	failing := mediaPolicyModel{Model: inner, media: ai.NewMediaAdapter(inner, ai.MediaPolicy{})}
	var errs []error
	for token := range failing.GenerateStream(t.Context(), request) {
		errs = append(errs, token.Err)
	}
	var unsupported *ai.UnsupportedContentError
	if len(errs) != 1 || !errors.As(errs[0], &unsupported) || len(inner.requests) != 1 {
		t.Fatalf("errors = %v, requests = %d", errs, len(inner.requests))
	}
}
