package context

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

type previewTestRenderer = Renderer

type previewObservationSink struct{ events []gai.Observation }

func (s *previewObservationSink) Emit(_ context.Context, event gai.Observation) {
	s.events = append(s.events, event)
}

type previewNodePart struct {
	name string
	node RenderNode
}

func (p previewNodePart) Name() string { return p.name }
func (p previewNodePart) Tokens(context.Context, ai.TokenCounter) (int, error) {
	return 0, nil
}
func (p previewNodePart) Render(context.Context) (RenderNode, error) {
	return p.node, nil
}

func requirePreviewTruncation(t *testing.T, fields map[string]any, key string) {
	t.Helper()
	head, headOK := fields[key+"_head"].(string)
	tail, tailOK := fields[key+"_tail"].(string)
	if fields[key+"_mode"] != "truncated" || !headOK || !tailOK || len([]rune(head)) != 5 || len([]rune(tail)) != 5 {
		t.Fatalf("preview size configuration changed: %v", fields)
	}
}

func previewRendererCases() []struct {
	name string
	new  func(*previewObservationSink, RenderResultCallback) previewTestRenderer
} {
	return []struct {
		name string
		new  func(*previewObservationSink, RenderResultCallback) previewTestRenderer
	}{
		{"XML pointer", func(sink *previewObservationSink, callback RenderResultCallback) previewTestRenderer {
			r := &XMLRenderer{ObservationSink: sink, ObservationPreviewChars: 5}
			_ = r.SetRenderResultCallback(context.Background(), callback)
			return r
		}},
		{"XML value", func(sink *previewObservationSink, callback RenderResultCallback) previewTestRenderer {
			r := &XMLRenderer{ObservationSink: sink, ObservationPreviewChars: 5}
			_ = r.SetRenderResultCallback(context.Background(), callback)
			return *r
		}},
		{"simple pointer", func(sink *previewObservationSink, callback RenderResultCallback) previewTestRenderer {
			r := &SimpleRenderer{ObservationSink: sink, ObservationPreviewChars: 5}
			_ = r.SetRenderResultCallback(context.Background(), callback)
			return r
		}},
		{"simple value", func(sink *previewObservationSink, callback RenderResultCallback) previewTestRenderer {
			r := &SimpleRenderer{ObservationSink: sink, ObservationPreviewChars: 5}
			_ = r.SetRenderResultCallback(context.Background(), callback)
			return *r
		}},
	}
}

type previewFailurePart struct{ err error }

func (previewFailurePart) Name() string { return "preview-failure" }
func (previewFailurePart) Tokens(ctx context.Context, _ ai.TokenCounter) (int, error) {
	return 0, ctx.Err()
}
func (p previewFailurePart) Render(ctx context.Context) (RenderNode, error) {
	if err := ctx.Err(); err != nil {
		return RenderNode{}, err
	}
	return RenderNode{}, p.err
}

func TestRendererPreviewMatchesNormalTextAndErrorsWithoutPublishing(t *testing.T) {
	failure := errors.New("part could not render")
	for _, rendererCase := range previewRendererCases() {
		for _, partCase := range []struct {
			name   string
			parts  []Part
			cancel bool
			want   error
		}{
			{name: "empty"},
			{name: "nil part", parts: []Part{nil}},
			{name: "escaped Unicode text", parts: []Part{NewTextPart(`< & " 日本 😀`)}},
			{name: "grouped instructions", parts: []Part{NewSystemPart([]Part{NewTextPart("first"), NewTextPart("second")})}},
			{name: "nested generic node", parts: []Part{previewNodePart{name: "custom", node: RenderNode{Type: "memo", Fields: []RenderField{{Key: "source", Value: `< & "`}}, Children: []RenderNode{{Type: "text", Value: "nested content"}}}}}},
			{name: "part failure", parts: []Part{previewFailurePart{err: failure}}, want: failure},
			{name: "cancellation-aware part", parts: []Part{previewFailurePart{}}, cancel: true, want: context.Canceled},
		} {
			t.Run(rendererCase.name+"/"+partCase.name, func(t *testing.T) {
				sink := &previewObservationSink{}
				var callbacks []string
				renderer := rendererCase.new(sink, func(_ []Part, prompt string) { callbacks = append(callbacks, prompt) })
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				if partCase.cancel {
					cancel()
				}
				preview, previewErr := renderPreview(ctx, renderer, partCase.parts)
				if !errors.Is(previewErr, partCase.want) || len(callbacks) != 0 || len(sink.events) != 0 {
					t.Fatalf("preview = %q, %v; callbacks %v, observations %v", preview, previewErr, callbacks, sink.events)
				}
				normal, normalErr := renderer.Render(ctx, partCase.parts)
				if preview != normal || !errors.Is(normalErr, partCase.want) {
					t.Fatalf("preview = %q, %v; normal = %q, %v", preview, previewErr, normal, normalErr)
				}
				if normalErr == nil {
					if len(callbacks) != 1 || callbacks[0] != normal {
						t.Fatalf("normal render lost its callback after preview: %v", callbacks)
					}
				} else if len(callbacks) != 0 {
					t.Fatalf("failed render published a callback: %v", callbacks)
				}
				if len(sink.events) == 0 || sink.events[0].Name != "renderer_render_started" || sink.events[len(sink.events)-1].Name != "renderer_render_finished" {
					t.Fatalf("normal render lost its observations after preview: %v", sink.events)
				}
			})
		}
	}
}

func TestRendererPreviewPreservesObservationAndCallbackConfiguration(t *testing.T) {
	for _, tc := range previewRendererCases() {
		t.Run(tc.name, func(t *testing.T) {
			sink := &previewObservationSink{}
			calls := 0
			renderer := tc.new(sink, func(_ []Part, _ string) { calls++ })
			ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{Prompt: gai.CaptureEnabled})
			parts := []Part{NewTextPart(strings.Repeat("long preview text 日本 ", 10))}
			preview, err := renderPreview(ctx, renderer, parts)
			if err != nil || calls != 0 || len(sink.events) != 0 {
				t.Fatalf("preview published effects: %v, calls %d, events %v", err, calls, sink.events)
			}
			normal, err := renderer.Render(ctx, parts)
			if err != nil || normal != preview || calls != 1 || len(sink.events) != 3 {
				t.Fatalf("normal render after preview = %q, %v; calls %d, events %v", normal, err, calls, sink.events)
			}
			requirePreviewTruncation(t, sink.events[1].Fields, "rendered")
			requirePreviewTruncation(t, sink.events[2].Fields, "prompt")
			// Repeating the preview must neither detach configuration nor publish.
			if _, err := renderPreview(ctx, renderer, parts); err != nil || calls != 1 || len(sink.events) != 3 {
				t.Fatalf("second preview changed publication state: %v, calls %d, events %v", err, calls, sink.events)
			}
			if _, err := renderer.Render(ctx, parts); err != nil || calls != 2 || len(sink.events) != 6 {
				t.Fatalf("second normal render lost configuration: %v, calls %d, events %v", err, calls, sink.events)
			}
		})
	}
}

type customPreviewRenderer struct {
	previews  int
	published []string
}

var _ Renderer = (*customPreviewRenderer)(nil)
var _ PreviewRenderer = (*customPreviewRenderer)(nil)

func (r *customPreviewRenderer) Render(ctx context.Context, parts []Part) (string, error) {
	text, err := (SimpleRenderer{}).Render(ctx, parts)
	if err != nil {
		return "", err
	}
	text = "custom-prefix:" + text
	r.published = append(r.published, text)
	return text, nil
}
func (r *customPreviewRenderer) RenderPreview(ctx context.Context, parts []Part) (string, error) {
	r.previews++
	text, err := renderPreview(ctx, SimpleRenderer{}, parts)
	if err != nil {
		return "", err
	}
	return "custom-prefix:" + text, nil
}

type previewByteCounter struct{}

func (previewByteCounter) ID() string                      { return "test/preview-bytes" }
func (previewByteCounter) Fidelity() ai.TokenCountFidelity { return ai.TokenCountEstimated }
func (previewByteCounter) CountTokens(ctx context.Context, text string) (int, error) {
	return len(text), ctx.Err()
}

type previewContextSource struct{ budgets []int }

func (*previewContextSource) Name() string { return "preview-context" }
func (s *previewContextSource) Function(_ context.Context, budget int) (Part, error) {
	s.budgets = append(s.budgets, budget)
	return NewTextPart("x"), nil
}

func TestBuilderUsesCustomPreviewForAllocationAndNormalRenderForFinalRequest(t *testing.T) {
	renderer := &customPreviewRenderer{}
	source := &previewContextSource{}
	builder := New(Definition{
		Renderer:           renderer,
		TokenCounter:       previewByteCounter{},
		SystemInstructions: []Part{NewTextPart("s")},
		ContextSources:     []ContextSource{source},
		PromptInput:        PromptInput{User: ai.TextParts("u"), Context: []Part{NewTextPart("c")}},
	})
	if err := builder.SetBudgetAllocation(1000, 0, 3); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.BuildContext(t.Context()); err != nil || renderer.previews != 3 || len(renderer.published) != 0 || len(source.budgets) != 1 || source.budgets[0] != 921 {
		t.Fatalf("allocation = %v, previews %d, published %v, allowances %v", err, renderer.previews, renderer.published, source.budgets)
	}
	request, err := builder.BuildRequest(t.Context(), nil)
	if err != nil || len(request.Messages) != 4 || renderer.previews != 3 || len(renderer.published) != 3 {
		t.Fatalf("final request = %+v, %v; previews %d, published %v", request, err, renderer.previews, renderer.published)
	}
	for i := 0; i < 3; i++ {
		if request.Messages[i].Text() != renderer.published[i] || !strings.HasPrefix(request.Messages[i].Text(), "custom-prefix:") {
			t.Fatalf("final message %d differs from normal publication: %+v / %q", i, request.Messages[i], renderer.published[i])
		}
	}
	if request.Messages[3].Role != ai.RoleUser || request.Messages[3].Text() != "u" {
		t.Fatalf("canonical user content changed: %+v", request.Messages[3])
	}
	count, err := ai.EstimateRequestTokens(t.Context(), request, previewByteCounter{})
	if err != nil || count.InputTokens != 98 {
		t.Fatalf("custom rendered request cost = %+v, %v", count, err)
	}
}

// Embedding a builtin and overriding Render does not opt into a separate
// preview contract. Its actual output and publication behavior must survive.
type embeddedPreviewWrapper struct {
	SimpleRenderer
	published []string
}

func (r *embeddedPreviewWrapper) Render(ctx context.Context, parts []Part) (string, error) {
	text, err := r.SimpleRenderer.Render(ctx, parts)
	if err != nil {
		return "", err
	}
	text = "wrapper-prefix:" + text
	r.published = append(r.published, text)
	return text, nil
}

func TestRenderPreviewPreservesEmbeddingWrapperRenderContract(t *testing.T) {
	sink := &previewObservationSink{}
	callbacks := 0
	wrapper := &embeddedPreviewWrapper{SimpleRenderer: SimpleRenderer{ObservationSink: sink}}
	if err := wrapper.SetRenderResultCallback(t.Context(), func(_ []Part, _ string) { callbacks++ }); err != nil {
		t.Fatal(err)
	}
	if _, inferred := any(wrapper).(PreviewRenderer); inferred {
		t.Fatal("embedding a builtin implicitly opted wrapper into a different preview contract")
	}
	parts := []Part{NewTextPart("x")}
	preview, err := renderPreview(t.Context(), wrapper, parts)
	if err != nil || preview != "wrapper-prefix:x" || len(wrapper.published) != 1 || callbacks != 1 || len(sink.events) != 3 {
		t.Fatalf("wrapper preview fallback = %q, %v; publications %v, callbacks %d, events %v", preview, err, wrapper.published, callbacks, sink.events)
	}
	normal, err := wrapper.Render(t.Context(), parts)
	if err != nil || normal != preview || len(wrapper.published) != 2 || callbacks != 2 || len(sink.events) != 6 {
		t.Fatalf("wrapper normal rendering = %q, %v; publications %v, callbacks %d, events %v", normal, err, wrapper.published, callbacks, sink.events)
	}
}
