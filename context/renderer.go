package context

import (
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

// Renderer converts ordered prompt parts into the model-facing prompt string.
// The built-in renderers also provide render-result callbacks and tool signature
// formatting, but custom renderers only need to implement Render.
type Renderer interface {
	Render(ctx context.Context, contextParts []Part) (string, error)
}

// PreviewRenderer optionally renders budget candidates without publishing render
// results. RenderPreview must match Render's output and errors for the same stable
// parts while omitting the renderer's callbacks and observations. It still calls
// Part.Render; custom parts are responsible for their own effects. Builder uses
// quiet previews internally for the built-in XML and Simple renderers.
type PreviewRenderer interface {
	Renderer
	RenderPreview(ctx context.Context, contextParts []Part) (string, error)
}

// ToolSignature is the minimal tool metadata required for prompt formatting.
type ToolSignature interface {
	Name() string
	Description() string
	Params() ai.ToolParameters
}

// RenderResultCallback receives the source parts and completed prompt immediately
// before a successful Render call returns.
type RenderResultCallback func(parts []Part, prompt string)

// RenderField is a named scalar attribute on a RenderNode.
type RenderField struct {
	Key   string
	Value string
}

// RenderNode is the renderer-neutral tree emitted by context parts.
type RenderNode struct {
	Type     string
	Fields   []RenderField
	Value    string
	Children []RenderNode
}

type (
	// XMLRenderer renders nodes as structured XML.
	XMLRenderer struct {
		// ObservationSink enables detailed renderer events. Prompt content is included
		// only when the context has an enabling ContentCapturePolicy.
		ObservationSink gai.ObservationSink
		// ObservationPreviewChars controls how much content is retained at each end of an observation preview.
		ObservationPreviewChars int
		renderResultCallback    RenderResultCallback
	}
	// SimpleRenderer renders nodes as compact role-labelled plain text.
	SimpleRenderer struct {
		// ObservationSink enables detailed renderer events. Prompt content is included
		// only when the context has an enabling ContentCapturePolicy.
		ObservationSink gai.ObservationSink
		// ObservationPreviewChars controls how much content is retained at each end of an observation preview.
		ObservationPreviewChars int
		renderResultCallback    RenderResultCallback
	}
)

var (
	_ Renderer = (*XMLRenderer)(nil)
	_ Renderer = (*SimpleRenderer)(nil)
)

// renderPreview recognizes exact built-in types, so a custom renderer that
// embeds one and overrides Render keeps its own formatting unless it explicitly
// implements PreviewRenderer. A promoted preview would bypass that override.
func renderPreview(ctx context.Context, renderer Renderer, parts []Part) (string, error) {
	switch r := renderer.(type) {
	case XMLRenderer:
		return r.render(ctx, parts, false)
	case *XMLRenderer:
		return r.render(ctx, parts, false)
	case SimpleRenderer:
		return r.render(ctx, parts, false)
	case *SimpleRenderer:
		return r.render(ctx, parts, false)
	case PreviewRenderer:
		return r.RenderPreview(ctx, parts)
	default:
		return renderer.Render(ctx, parts)
	}
}

func renderToolSignatures(tools []ToolSignature) (string, error) {
	if len(tools) == 0 {
		return "", nil
	}

	sorted := make([]ToolSignature, 0, len(tools))
	for _, tool := range tools {
		if tool != nil {
			sorted = append(sorted, tool)
		}
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Name() < sorted[j].Name()
	})

	var builder strings.Builder
	for _, t := range sorted {
		builder.WriteString("\n<tool name=\"")
		builder.WriteString(t.Name())
		builder.WriteString("\">")
		builder.WriteString("\n<description>")
		builder.WriteString(t.Description())
		builder.WriteString("</description>")
		builder.WriteString("\n<signature>")
		params, err := t.Params().JSONSchema()
		if err != nil {
			return "", fmt.Errorf("tool %q: %w", t.Name(), err)
		}
		builder.Write(params)
		builder.WriteString("</signature>")
		builder.WriteString("\n</tool>")
	}
	return builder.String(), nil
}

func (r XMLRenderer) Render(ctx context.Context, parts []Part) (string, error) {
	return r.render(ctx, parts, true)
}

func (r XMLRenderer) render(ctx context.Context, parts []Part, publish bool) (string, error) {
	var obs *renderObserver
	if publish {
		obs = newRenderObserver("xml", r.ObservationSink, r.ObservationPreviewChars)
	}
	obs.started(ctx, len(parts))
	if len(parts) == 0 {
		if publish {
			r.notifyRenderResult(parts, "")
		}
		obs.finished(ctx, nil, "")
		return "", nil
	}

	var builder strings.Builder
	for index, part := range parts {
		if part == nil {
			obs.partRendered(ctx, index, nil, nil, "")
			continue
		}
		node, err := part.Render(ctx)
		if err != nil {
			obs.partFailed(ctx, index, part, err)
			obs.finished(ctx, err, builder.String())
			return "", err
		}
		start := builder.Len()
		if err := writeXMLNode(&builder, node, 0); err != nil {
			obs.partFailed(ctx, index, part, err)
			obs.finished(ctx, err, builder.String())
			return "", err
		}
		obs.partRendered(ctx, index, part, &node, builder.String()[start:])
	}

	prompt := builder.String()
	if publish {
		r.notifyRenderResult(parts, prompt)
	}
	obs.finished(ctx, nil, prompt)
	return prompt, nil
}

func (r SimpleRenderer) Render(ctx context.Context, parts []Part) (string, error) {
	return r.render(ctx, parts, true)
}

func (r SimpleRenderer) render(ctx context.Context, parts []Part, publish bool) (string, error) {
	var obs *renderObserver
	if publish {
		obs = newRenderObserver("simple", r.ObservationSink, r.ObservationPreviewChars)
	}
	obs.started(ctx, len(parts))
	if len(parts) == 0 {
		if publish {
			r.notifyRenderResult(parts, "")
		}
		obs.finished(ctx, nil, "")
		return "", nil
	}

	blocks := make([]string, 0, len(parts))
	for index, part := range parts {
		block, node, err := renderSimplePart(ctx, part)
		if err != nil {
			obs.partFailed(ctx, index, part, err)
			obs.finished(ctx, err, strings.Join(blocks, "\n\n"))
			return "", err
		}
		obs.partRendered(ctx, index, part, node, block)
		if block == "" {
			continue
		}
		blocks = append(blocks, block)
	}

	prompt := strings.Join(blocks, "\n\n")
	if publish {
		r.notifyRenderResult(parts, prompt)
	}
	obs.finished(ctx, nil, prompt)
	return prompt, nil
}

func (r XMLRenderer) RenderToolSignatures(tools []ToolSignature) (string, error) {
	return renderToolSignatures(tools)
}

func (r SimpleRenderer) RenderToolSignatures(tools []ToolSignature) (string, error) {
	return renderToolSignatures(tools)
}

func renderSimplePart(ctx context.Context, part Part) (string, *RenderNode, error) {
	if part == nil {
		return "", nil, nil
	}

	switch p := part.(type) {
	case SystemPart:
		return renderSimpleInstructions(ctx, p)
	case MessagePart:
		node, err := p.Render(ctx)
		if err != nil {
			return "", nil, err
		}
		return renderSimpleMessageNode(node), &node, nil
	}

	node, err := part.Render(ctx)
	if err != nil {
		return "", nil, err
	}
	return renderSimpleNode(node), &node, nil
}

func renderSimpleInstructions(ctx context.Context, part SystemPart) (string, *RenderNode, error) {
	var builder strings.Builder
	builder.WriteString("<Instructions>")
	node := RenderNode{Type: "instructions"}

	wroteInstruction := false
	for _, instruction := range part.Instructions {
		if instruction == nil {
			continue
		}

		body, child, err := renderSimpleInstruction(ctx, instruction)
		if err != nil {
			return "", nil, err
		}
		node.Children = append(node.Children, child)
		if body == "" {
			continue
		}

		builder.WriteString("\n\n")
		if !isSimpleTextInstruction(instruction) {
			builder.WriteString(formatSimpleInstructionLabel(instruction.Name()))
			builder.WriteString(":\n")
		}
		builder.WriteString(body)
		wroteInstruction = true
	}

	if wroteInstruction {
		builder.WriteString("\n\n")
	} else {
		builder.WriteString("\n")
	}
	builder.WriteString("</Instructions>")
	return builder.String(), &node, nil
}

func isSimpleTextInstruction(part Part) bool {
	switch part.(type) {
	case TextPart, *TextPart:
		return true
	default:
		return false
	}
}

func renderSimpleInstruction(ctx context.Context, part Part) (string, RenderNode, error) {
	switch p := part.(type) {
	case TextPart:
		node, err := p.Render(ctx)
		return p.Content, node, err
	case MessagePart:
		node, err := p.Render(ctx)
		if err != nil {
			return "", RenderNode{}, err
		}
		return renderSimpleMessageNode(node), node, nil
	}

	node, err := part.Render(ctx)
	if err != nil {
		return "", RenderNode{}, err
	}
	return renderSimpleNodeBody(node), node, nil
}

func renderSimpleNode(node RenderNode) string {
	switch node.Type {
	case string(ai.ContentText):
		return node.Value
	case "history":
		return renderSimpleHistory(node)
	case "tools":
		return renderSimpleTools(node)
	case string(ai.RoleUser), string(ai.RoleAssistant), string(ai.RoleTool), string(ai.RoleSystem), "summary", "message":
		return renderSimpleMessageNode(node)
	default:
		return renderSimpleGenericNode(node)
	}
}

func renderSimpleTools(node RenderNode) string {
	var builder strings.Builder
	builder.WriteString("<tools>")
	for _, child := range node.Children {
		if child.Type == "tool_usage" {
			body := renderSimpleNodeBody(child)
			if body != "" {
				builder.WriteString("\n")
				builder.WriteString("usage:\n")
				builder.WriteString(body)
			}
			continue
		}
		if child.Type != "tool" {
			continue
		}

		builder.WriteString("\n")
		builder.WriteString("tool: ")
		name := simpleNodeFieldValue(child, "name")
		if name == "" {
			name = simpleNodeChildValue(child, "tool-name")
		}
		builder.WriteString(name)
		for _, toolChild := range child.Children {
			if toolChild.Type == "tool-name" {
				continue
			}
			body := renderSimpleNodeBody(toolChild)
			if body == "" {
				continue
			}
			builder.WriteString("\n")
			builder.WriteString(toolChild.Type)
			builder.WriteString(": ")
			builder.WriteString(body)
		}
	}
	if len(node.Children) > 0 {
		builder.WriteString("\n")
	}
	builder.WriteString("</tools>")
	return builder.String()
}

func renderSimpleGenericNode(node RenderNode) string {
	bodyParts := make([]string, 0, len(node.Children)+1)
	if node.Value != "" {
		bodyParts = append(bodyParts, node.Value)
	}
	for _, child := range node.Children {
		if rendered := renderSimpleNode(child); rendered != "" {
			bodyParts = append(bodyParts, rendered)
		}
	}
	body := strings.Join(bodyParts, "\n")
	if node.Type == "" {
		return body
	}

	var builder strings.Builder
	builder.WriteString("<")
	builder.WriteString(node.Type)
	for _, field := range node.Fields {
		if field.Key == "" {
			continue
		}
		builder.WriteString(" ")
		builder.WriteString(field.Key)
		builder.WriteString("=")
		encoded, _ := json.Marshal(field.Value)
		builder.Write(encoded)
	}
	builder.WriteString(">")
	if body != "" {
		builder.WriteString("\n")
		builder.WriteString(body)
	}
	builder.WriteString("\n</")
	builder.WriteString(node.Type)
	builder.WriteString(">")
	return builder.String()
}

func renderSimpleHistory(node RenderNode) string {
	var builder strings.Builder
	builder.WriteString("<history>")
	if len(node.Children) > 0 {
		builder.WriteString("\n")
		for i, child := range node.Children {
			if i > 0 {
				builder.WriteString("\n")
			}
			builder.WriteString(renderSimpleMessageNode(child))
		}
		builder.WriteString("\n")
	}
	builder.WriteString("</history>")
	return builder.String()
}

func renderSimpleMessageNode(node RenderNode) string {
	return formatSimpleLine(simpleMessageLabel(node.Type), renderSimpleNodeBody(node))
}

func simpleMessageLabel(nodeType string) string {
	switch nodeType {
	case string(ai.RoleTool):
		return "tool res"
	default:
		return nodeType
	}
}

func formatSimpleLine(label, body string) string {
	if body == "" {
		return label + ":"
	}
	if strings.Contains(body, "\n") {
		return label + ":\n" + body
	}
	return label + ": " + body
}

func renderSimpleNodeBody(node RenderNode) string {
	switch node.Type {
	case string(ai.ContentText):
		return node.Value
	case string(ai.ContentToolCall):
		return renderSimpleToolCall(node)
	case string(ai.ContentToolResult):
		results := make([]string, 0, len(node.Children))
		for _, child := range node.Children {
			if child.Type == "result" {
				results = append(results, child.Value)
			}
		}
		result := node.Value
		if len(results) > 0 {
			result = strings.Join(results, "\n")
		}
		if simpleNodeFieldValue(node, "is_error") == "true" {
			return "error: " + result
		}
		return result
	}

	parts := make([]string, 0, len(node.Children)+1)
	if node.Value != "" {
		parts = append(parts, node.Value)
	}
	for _, child := range node.Children {
		body := renderSimpleNodeBody(child)
		if body != "" {
			parts = append(parts, body)
		}
	}
	return strings.Join(parts, "\n")
}

func renderSimpleToolCall(node RenderNode) string {
	name := simpleNodeFieldValue(node, "name")
	arguments := simpleNodeChildValue(node, "arguments")
	if name == "" {
		return arguments
	}

	var argsValue any
	rawArgs := json.RawMessage(arguments)
	if len(arguments) > 0 && json.Valid(rawArgs) {
		argsValue = rawArgs
	} else {
		argsValue = arguments
	}

	payload := map[string]any{
		"type":      "function",
		"name":      name,
		"arguments": argsValue,
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return `{"type":"function","name":"","arguments":""}`
	}
	return string(encoded)
}

func simpleNodeChildValue(node RenderNode, childType string) string {
	for _, child := range node.Children {
		if child.Type == childType {
			return child.Value
		}
	}
	return node.Value
}

func simpleNodeFieldValue(node RenderNode, key string) string {
	for _, field := range node.Fields {
		if field.Key == key {
			return field.Value
		}
	}
	return ""
}

func formatSimpleInstructionLabel(name string) string {
	if name == "" {
		return "Instruction"
	}

	parts := strings.FieldsFunc(name, func(r rune) bool {
		return r == '_' || r == '-' || unicode.IsSpace(r)
	})
	if len(parts) == 0 {
		return "Instruction"
	}

	for i, part := range parts {
		r, size := utf8.DecodeRuneInString(part)
		if r == utf8.RuneError && size == 0 {
			continue
		}
		parts[i] = string(unicode.ToUpper(r)) + strings.ToLower(part[size:])
	}
	return strings.Join(parts, " ")
}

func writeXMLNode(builder *strings.Builder, node RenderNode, depth int) error {
	if !validXMLName(node.Type) {
		return fmt.Errorf("invalid xml tag name: %q", node.Type)
	}
	indent(builder, depth)
	builder.WriteString("<")
	builder.WriteString(node.Type)
	for _, field := range node.Fields {
		if !validXMLName(field.Key) {
			return fmt.Errorf("invalid xml attribute name: %q", field.Key)
		}
		builder.WriteString(" ")
		builder.WriteString(field.Key)
		builder.WriteString(`="`)
		writeEscaped(builder, field.Value)
		builder.WriteString(`"`)
	}
	builder.WriteString(">")
	if node.Value != "" {
		builder.WriteString("\n")
		indent(builder, depth+1)
		writeEscaped(builder, node.Value)
		builder.WriteString("\n")
	}
	if len(node.Children) > 0 {
		builder.WriteString("\n")
	}
	for _, child := range node.Children {
		if err := writeXMLNode(builder, child, depth+1); err != nil {
			return err
		}
	}
	if node.Value == "" && len(node.Children) == 0 {
		builder.WriteString("\n")
	}
	indent(builder, depth)
	builder.WriteString("</")
	builder.WriteString(node.Type)
	builder.WriteString(">\n")
	return nil
}

func indent(builder *strings.Builder, depth int) {
	for range depth {
		builder.WriteString("  ")
	}
}

func writeEscaped(builder *strings.Builder, text string) {
	if text == "" {
		return
	}
	_ = xml.EscapeText(builder, []byte(text))
}

func validXMLName(name string) bool {
	if name == "" {
		return false
	}
	for i, r := range name {
		if i == 0 {
			if r != '_' && !unicode.IsLetter(r) {
				return false
			}
			continue
		}
		if r != '_' && r != '-' && r != '.' && !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

func (r *XMLRenderer) SetRenderResultCallback(_ context.Context, callback RenderResultCallback) error {
	r.renderResultCallback = callback
	return nil
}

func (r XMLRenderer) notifyRenderResult(parts []Part, prompt string) {
	if r.renderResultCallback != nil {
		r.renderResultCallback(parts, prompt)
	}
}

func (r *SimpleRenderer) SetRenderResultCallback(_ context.Context, callback RenderResultCallback) error {
	r.renderResultCallback = callback
	return nil
}

func (r SimpleRenderer) notifyRenderResult(parts []Part, prompt string) {
	if r.renderResultCallback != nil {
		r.renderResultCallback(parts, prompt)
	}
}
