package loop_test

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

type customRequestBuilder struct {
	stubPromptBuilder
	request ai.AIRequest
}

func (b *customRequestBuilder) BuildRequest(context.Context, gaictx.Conversation) (ai.AIRequest, error) {
	return b.request.Copy(), nil
}

func TestLoopNormalizesPromptOnlyBuilderForEveryTransport(t *testing.T) {
	for _, transport := range []struct {
		name string
		mode loop.ToolTransportMode
	}{{"native", loop.ToolTransportNative}, {"text", loop.ToolTransportText}} {
		t.Run(transport.name, func(t *testing.T) {
			model := &scriptedStreamModel{sequences: [][]ai.Token{{{Type: ai.TokenTypeText, Text: "done"}}}}
			builder := &customRequestBuilder{request: ai.AIRequest{Prompt: "rendered"}}
			l := loop.New(model, nil, builder, nil)
			l.ToolTransport = transport.mode
			l.MaxTokens = 42
			if err := loopError(collectLoopEvents(t, l, t.Context())); err != nil {
				t.Fatal(err)
			}
			requests := model.Requests()
			if len(requests) != 1 {
				t.Fatalf("requests = %d, want one", len(requests))
			}
			request := requests[0]
			if !reflect.DeepEqual(request.Messages, []ai.Message{ai.TextMessage(ai.RoleUser, "rendered")}) || request.MaxTokens != 42 {
				t.Fatalf("prompt-only request was not normalized: %#v", request)
			}
			if transport.mode == loop.ToolTransportText {
				want, err := ai.RenderMessages(t.Context(), request.Messages)
				if err != nil || request.Prompt != want || !strings.Contains(request.Prompt, "rendered") {
					t.Fatalf("fallback lost prompt: %q, err=%v", request.Prompt, err)
				}
			} else if request.Prompt != "" {
				t.Fatalf("native prompt was not consumed by normalization: %q", request.Prompt)
			}
		})
	}
}

func TestLoopNormalizesAuthoritativeMessagesAndRejectsMalformedRequests(t *testing.T) {
	for _, transport := range []struct {
		name string
		mode loop.ToolTransportMode
	}{{"native", loop.ToolTransportNative}, {"text", loop.ToolTransportText}} {
		t.Run(transport.name+"/authoritative", func(t *testing.T) {
			model := &scriptedStreamModel{sequences: [][]ai.Token{{{Type: ai.TokenTypeText, Text: "done"}}}}
			builder := &customRequestBuilder{request: ai.AIRequest{Prompt: "stale prompt", Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "authoritative")}}}
			l := loop.New(model, nil, builder, nil)
			l.ToolTransport = transport.mode
			if err := loopError(collectLoopEvents(t, l, t.Context())); err != nil {
				t.Fatal(err)
			}
			requests := model.Requests()
			if len(requests) != 1 || !reflect.DeepEqual(requests[0].Messages, builder.request.Messages) || strings.Contains(requests[0].Prompt, "stale") {
				t.Fatalf("stale prompt affected request: %#v", requests)
			}
		})
		t.Run(transport.name+"/invalid", func(t *testing.T) {
			model := &scriptedStreamModel{sequences: [][]ai.Token{{{Type: ai.TokenTypeText, Text: "must not run"}}}}
			builder := &customRequestBuilder{request: ai.AIRequest{Messages: []ai.Message{{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "missing", Name: "tool", Parts: ai.TextParts("orphaned")}}}}}}}
			l := loop.New(model, nil, builder, nil)
			l.ToolTransport = transport.mode
			err := loopError(collectLoopEvents(t, l, t.Context()))
			if !errors.Is(err, loop.ErrBuildPrompt) {
				t.Fatalf("malformed request error = %v, want build failure", err)
			}
			if len(model.Requests()) != 0 {
				t.Fatal("malformed canonical request reached model")
			}
		})
	}
}
