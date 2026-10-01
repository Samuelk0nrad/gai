package context

import (
	"context"
	"fmt"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/internal/observe"
)

const defaultRendererObservationPreviewChars = 500

type renderObserver struct {
	renderer    string
	debug       gai.ObservationSink
	operation   *observe.Operation
	previewSize int
	parts       []map[string]any
}

func newRenderObserver(renderer string, debug gai.ObservationSink, previewSize int) *renderObserver {
	if previewSize <= 0 {
		previewSize = defaultRendererObservationPreviewChars
	}
	return &renderObserver{
		renderer:    renderer,
		debug:       debug,
		operation:   observe.New(debug, "context:"+rendererSourceName(renderer)),
		previewSize: previewSize,
	}
}

func (o *renderObserver) started(ctx context.Context, partCount int) {
	if !o.enabled(ctx) {
		return
	}
	o.emit(ctx, "renderer_render_started", map[string]any{
		"renderer":   o.renderer,
		"part_count": partCount,
	}, nil)
}

func (o *renderObserver) partRendered(ctx context.Context, index int, part Part, node *RenderNode, rendered string) {
	if !o.enabled(ctx) {
		return
	}
	fields := map[string]any{
		"renderer":       o.renderer,
		"part_index":     index,
		"part_name":      renderPartName(part),
		"rendered_chars": len(rendered),
	}
	var contentKinds map[gai.ContentKind]struct{}
	if node != nil {
		fields["node"], contentKinds = rendererNodeStructure(ctx, o.debug, *node, o.previewSize, gai.ContentKindPrompt)
	}
	if kind, ok := singleRendererContentKind(contentKinds); ok {
		addRendererCapturedContent(ctx, o.debug, fields, "rendered", rendered, o.previewSize, kind)
	} else if _, hasPolicy := gai.ContentCapturePolicyFromContext(ctx); !hasPolicy {
		addRendererCapturedContent(ctx, o.debug, fields, "rendered", rendered, o.previewSize, gai.ContentKindPrompt)
	}
	o.parts = append(o.parts, fields)
	o.emit(ctx, "renderer_part_rendered", fields, nil)
}

func (o *renderObserver) partFailed(ctx context.Context, index int, part Part, err error) {
	if !o.enabled(ctx) {
		return
	}
	fields := map[string]any{
		"renderer":   o.renderer,
		"part_index": index,
		"part_name":  renderPartName(part),
	}
	o.emit(ctx, "renderer_part_failed", fields, err)
}

func (o *renderObserver) finished(ctx context.Context, err error, prompt string) {
	if !o.enabled(ctx) {
		return
	}
	fields := map[string]any{
		"renderer":     o.renderer,
		"part_count":   len(o.parts),
		"prompt_chars": len(prompt),
		"structure":    o.parts,
	}
	addRendererCapturedContent(ctx, o.debug, fields, "prompt", prompt, o.previewSize, gai.ContentKindPrompt)
	o.emit(ctx, "renderer_render_finished", fields, err)
}

func (o *renderObserver) enabled(ctx context.Context) bool {
	return o != nil && o.operation != nil && gai.ObservationEnabled(ctx, o.debug)
}

func (o *renderObserver) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if o == nil || o.operation == nil {
		return
	}
	o.operation.Emit(ctx, name, fields, err)
}

func rendererSourceName(renderer string) string {
	if renderer == "xml" {
		return "XMLRenderer"
	}
	return "SimpleRenderer"
}

func renderPartName(part Part) string {
	if part == nil {
		return "<nil>"
	}
	return part.Name()
}

func rendererNodeStructure(ctx context.Context, debug gai.ObservationSink, node RenderNode, previewSize int, inheritedKind gai.ContentKind) (map[string]any, map[gai.ContentKind]struct{}) {
	structure := map[string]any{
		"type":        node.Type,
		"value_chars": len(node.Value),
		"child_count": len(node.Children),
	}
	kind := rendererNodeContentKind(node.Type, inheritedKind)
	contentKinds := map[gai.ContentKind]struct{}{}
	if len(node.Fields) > 0 {
		fields := make([]map[string]any, 0, len(node.Fields))
		for _, field := range node.Fields {
			entry := map[string]any{"key": field.Key, "value_chars": len(field.Value)}
			addRendererCapturedContent(ctx, debug, entry, "value", field.Value, previewSize, kind)
			contentKinds[kind] = struct{}{}
			fields = append(fields, entry)
		}
		structure["fields"] = fields
	}
	if node.Value != "" {
		addRendererCapturedContent(ctx, debug, structure, "value", node.Value, previewSize, kind)
		contentKinds[kind] = struct{}{}
	}
	if len(node.Children) > 0 {
		children := make([]map[string]any, 0, len(node.Children))
		for _, child := range node.Children {
			childStructure, childKinds := rendererNodeStructure(ctx, debug, child, previewSize, kind)
			children = append(children, childStructure)
			for childKind := range childKinds {
				contentKinds[childKind] = struct{}{}
			}
		}
		structure["children"] = children
	}
	return structure, contentKinds
}

func rendererNodeContentKind(nodeType string, inherited gai.ContentKind) gai.ContentKind {
	switch nodeType {
	case "history":
		return gai.ContentKindMemory
	case string(ai.ContentToolCall):
		return gai.ContentKindToolInput
	case string(ai.ContentToolResult), string(ai.RoleTool):
		return gai.ContentKindToolOutput
	default:
		return inherited
	}
}

func singleRendererContentKind(kinds map[gai.ContentKind]struct{}) (gai.ContentKind, bool) {
	if len(kinds) != 1 {
		return "", false
	}
	for kind := range kinds {
		return kind, true
	}
	return "", false
}

func addRendererCapturedContent(ctx context.Context, debug gai.ObservationSink, fields map[string]any, key, content string, previewSize int, kind gai.ContentKind) {
	gai.AddObservationContent(ctx, debug, fields, key, kind, content)
	captured, ok := fields[key].(string)
	if !ok {
		return
	}
	addRendererPreview(fields, key, captured, previewSize)
}

func addRendererPreview(fields map[string]any, key, content string, previewSize int) {
	if len([]rune(content)) <= previewSize*2 {
		return
	}

	runes := []rune(content)
	delete(fields, key)
	omitted := len(runes) - previewSize*2
	fields[key+"_mode"] = "truncated"
	fields[key+"_head"] = string(runes[:previewSize])
	fields[key+"_omitted"] = fmt.Sprintf("[%d chars omitted]", omitted)
	fields[key+"_tail"] = string(runes[len(runes)-previewSize:])
}
