package loop_test

import (
	"context"
	"errors"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
	"testing"
)

type customRequestBuilder struct {
	stubPromptBuilder
	request ai.AIRequest
}

func (b *customRequestBuilder) BuildRequest(context.Context, gaictx.Conversation) (ai.AIRequest, error) {
	return b.request.Copy(), nil
}

func TestLoopRejectsMalformedCanonicalRequestsBeforeModel(t *testing.T) {
	for _, transport := range []loop.ToolTransportMode{loop.ToolTransportNative, loop.ToolTransportText} {
		for _, request := range []ai.AIRequest{
			{},
			{Messages: []ai.Message{{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "missing", Name: "tool", Parts: ai.TextParts("orphaned")}}}}}},
		} {
			model := &scriptedStreamModel{sequences: [][]ai.Token{{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "must not run"}}}}}
			l := loop.New(model, nil, &customRequestBuilder{request: request}, nil)
			l.ToolTransport = transport
			err := loopError(collectLoopEvents(t, l, t.Context()))
			if !errors.Is(err, loop.ErrBuildPrompt) {
				t.Fatalf("malformed request error=%v, want build failure", err)
			}
			if len(model.Requests()) != 0 {
				t.Fatal("malformed canonical request reached model")
			}
		}
	}
}
