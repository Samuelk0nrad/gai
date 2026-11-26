package loop

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/Samuelk0nrad/gai/ai"
)

// ToolResult is the normalized result of an invocation or result processor.
// Err takes precedence over Text and Media. Errors are immutable caller-owned
// values.
//
// Media holds images, PDFs or other files that a ResultTool returns. They follow
// Text in the canonical tool result, as ai.ContentMedia parts, and reach the
// model only through providers that accept media in tool results. Processors
// receive and return Media: a processor that builds a new ToolResult must copy
// Media to keep it. Text processors such as RedactToolResult,
// LimitToolResultBytes and TruncateToolResultBytes see only Text; media bytes
// are not redacted, limited or truncated.
type ToolResult struct {
	Text  string
	Media []ai.MediaPart
	Err   error
}

// String returns the model-facing text, giving errors precedence over output.
// It does not include Media.
func (r ToolResult) String() string {
	if r.Err != nil {
		return r.Err.Error()
	}
	return r.Text
}

// normalizeToolResult discards text when a handler returns an error.
func normalizeToolResult(text string, err error) ToolResult {
	if err != nil {
		return ToolResult{Err: err}
	}
	return ToolResult{Text: text}
}

// normalized discards text and media from an error result, and copies media
// so later mutation by the producer cannot change the published result.
func (r ToolResult) normalized() ToolResult {
	if r.Err != nil {
		return ToolResult{Err: r.Err}
	}
	return ToolResult{Text: r.Text, Media: cloneMedia(r.Media)}
}

func cloneMedia(src []ai.MediaPart) []ai.MediaPart {
	if len(src) == 0 {
		return nil
	}
	out := make([]ai.MediaPart, len(src))
	for i, media := range src {
		media.Data = append([]byte(nil), media.Data...)
		out[i] = media
	}
	return out
}

func equalMedia(a, b []ai.MediaPart) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].MIMEType != b[i].MIMEType || a[i].URI != b[i].URI || a[i].Name != b[i].Name || !bytes.Equal(a[i].Data, b[i].Data) {
			return false
		}
	}
	return true
}

// validateMedia checks each media part as canonical content.
func validateMedia(media []ai.MediaPart) error {
	for i := range media {
		if err := (ai.ContentPart{Kind: ai.ContentMedia, Media: &media[i]}).Validate(); err != nil {
			return fmt.Errorf("tool result media %d: %w", i, err)
		}
	}
	return nil
}

// canonical converts a processed result to the canonical tool result for
// call. A result without media keeps exactly one text part, as before media
// existed. With media, non-empty text comes first, then each media part.
func (r ToolResult) canonical(call ai.ToolCall) ai.ToolResult {
	out := ai.ToolResult{ToolCallID: call.ID, Name: call.Name, IsError: r.Err != nil}
	if r.Err != nil || len(r.Media) == 0 {
		out.Parts = ai.TextParts(r.String())
		return out
	}
	if r.Text != "" {
		out.Parts = ai.TextParts(r.Text)
	}
	for _, media := range cloneMedia(r.Media) {
		out.Parts = append(out.Parts, ai.ContentPart{Kind: ai.ContentMedia, Media: &media})
	}
	return out
}

// ResultTool is an optional Tool extension for tools whose result carries
// media, such as an image or PDF, as well as text. The loop calls
// FunctionResult instead of Function. A returned error takes precedence over
// the result, as for Function. Build one with NewResultTool.
type ResultTool interface {
	Tool
	FunctionResult(context.Context, ai.ToolCall) (ToolResult, error)
}

// ToolResultFunc is the implementation of a ResultTool.
type ToolResultFunc func(context.Context, ai.ToolCall) (ToolResult, error)

// InvokeTool runs tool with call through FunctionResult when the tool is a
// ResultTool, otherwise through Function, and returns the normalized result.
// Tool wrappers that embed a Tool use it to forward FunctionResult. It does
// not validate the call or the media and bypasses Loop policy and processing.
func InvokeTool(ctx context.Context, tool Tool, call ai.ToolCall) ToolResult {
	if resultTool, ok := tool.(ResultTool); ok {
		result, err := resultTool.FunctionResult(ctx, call)
		if err != nil {
			return ToolResult{Err: err}
		}
		return result.normalized()
	}
	text, err := tool.Function(ctx, call)
	return normalizeToolResult(text, err)
}

// Tool defines a function that a model may request during a loop run.
// Implementations must be safe for concurrent use and honor context cancellation.
type Tool interface {
	// Name is the stable identifier used by model calls.
	Name() string
	// Description explains the tool to the model.
	Description() string
	// Params declares the model-facing JSON argument schema.
	Params() ai.ToolParameters
	// Function receives a call owned by this invocation. Err takes precedence over text.
	Function(context.Context, ai.ToolCall) (string, error)
}

// ToolFunc is the implementation of a text tool, without its declaration.
type ToolFunc func(context.Context, ai.ToolCall) (string, error)

type functionTool struct {
	name, description string
	paramsJSON        []byte
	rawSchema         json.RawMessage
	rawSchemaSet      bool
	function          ToolFunc
}

// NewTool combines declaration metadata and a function. It validates the
// declaration and snapshots parameters; subsequent caller mutations are isolated.
func NewTool(name, description string, params ai.ToolParameters, function ToolFunc) (Tool, error) {
	if function == nil {
		return nil, fmt.Errorf("%w: tool function is nil", ai.ErrInvalidToolDefinition)
	}
	raw, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("%w: parameters: %w", ai.ErrInvalidToolDefinition, err)
	}
	tool := &functionTool{name: name, description: description, paramsJSON: raw, function: function}
	if _, err := ToolDefinitions([]Tool{tool}); err != nil {
		return nil, err
	}
	return tool, nil
}

// NewRawSchemaTool is NewTool with an exact JSON Schema parameter definition.
// The schema is copied and passed to every provider without reducing it to the
// typed ToolParameters subset.
func NewRawSchemaTool(name, description string, schema json.RawMessage, function ToolFunc) (Tool, error) {
	if function == nil {
		return nil, fmt.Errorf("%w: tool function is nil", ai.ErrInvalidToolDefinition)
	}
	tool := &functionTool{
		name:         name,
		description:  description,
		rawSchema:    append(json.RawMessage(nil), schema...),
		rawSchemaSet: true,
		function:     function,
	}
	if _, err := ToolDefinitions([]Tool{tool}); err != nil {
		return nil, err
	}
	return tool, nil
}

// Name returns the validated tool identifier.
func (t *functionTool) Name() string { return t.name }

// Description returns the model-facing tool description.
func (t *functionTool) Description() string { return t.description }

// Params returns an independent parameter snapshot, preserving JSON numbers.
func (t *functionTool) Params() ai.ToolParameters {
	if t.rawSchemaSet {
		return ai.ToolParameters{RawSchema: append(json.RawMessage(nil), t.rawSchema...)}
	}
	var params ai.ToolParameters
	decoder := json.NewDecoder(bytes.NewReader(t.paramsJSON))
	decoder.UseNumber()
	if err := decoder.Decode(&params); err != nil {
		panic("loop: invalid private tool parameter snapshot")
	}
	return params
}

// Function invokes the registered handler with the supplied call.
func (t *functionTool) Function(ctx context.Context, call ai.ToolCall) (string, error) {
	return t.function(ctx, call)
}

type resultTool struct {
	functionTool
	result ToolResultFunc
}

// NewResultTool is NewTool for a handler that returns a ToolResult, whose
// Media is sent to the model after its Text. The returned tool is a
// ResultTool; its Function returns only the text.
func NewResultTool(name, description string, params ai.ToolParameters, function ToolResultFunc) (Tool, error) {
	if function == nil {
		return nil, fmt.Errorf("%w: tool function is nil", ai.ErrInvalidToolDefinition)
	}
	text := func(ctx context.Context, call ai.ToolCall) (string, error) {
		result, err := function(ctx, call)
		if err != nil {
			return "", err
		}
		return result.String(), result.Err
	}
	base, err := NewTool(name, description, params, text)
	if err != nil {
		return nil, err
	}
	return &resultTool{functionTool: *base.(*functionTool), result: function}, nil
}

// FunctionResult invokes the registered handler with the supplied call.
func (t *resultTool) FunctionResult(ctx context.Context, call ai.ToolCall) (ToolResult, error) {
	return t.result(ctx, call)
}

// CallTool validates a call and invokes the matching tool with an isolated copy.
// This low-level helper bypasses Loop execution policy and result processing.
// Invalid calls and missing tools return ordinary tool errors.
func CallTool(ctx context.Context, req ai.ToolCall, tools []Tool) ToolResult {
	result, _ := invokeTool(ctx, req, tools)
	return result
}

func invokeTool(ctx context.Context, req ai.ToolCall, tools []Tool) (ToolResult, bool) {
	if err := req.Validate(); err != nil {
		return ToolResult{Err: err}, false
	}
	if !json.Valid(req.Args) {
		return ToolResult{Err: ErrToolCallMalformed}, false
	}
	if err := ctx.Err(); err != nil {
		return ToolResult{Err: err}, false
	}
	for index, tool := range tools {
		if nilImplementation(tool) {
			return ToolResult{Err: fmt.Errorf("%w: tool at index %d is nil", ai.ErrInvalidToolDefinition, index)}, false
		}
		if tool.Name() == req.Name {
			result := InvokeTool(ctx, tool, req.Clone())
			if err := validateMedia(result.Media); err != nil {
				return ToolResult{Err: err}, true
			}
			return result, true
		}
	}
	return ToolResult{Err: fmt.Errorf("%w: %s", ErrToolNotFound, req.Name)}, false
}

// nilImplementation detects nil interfaces and typed-nil dependencies without invoking them.
func nilImplementation(tool any) bool {
	if tool == nil {
		return true
	}
	v := reflect.ValueOf(tool)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	}
	return false
}

// DecodeToolArgs validates the call and decodes its JSON arguments into target.
func DecodeToolArgs[T any](req ai.ToolCall, target *T) error {
	if err := req.Validate(); err != nil {
		return err
	}
	if target == nil {
		return ErrArgsDecodeTarget
	}
	if err := json.Unmarshal(req.Args, target); err != nil {
		return fmt.Errorf("%w: %w", ErrToolCallMalformed, err)
	}
	return nil
}

// ToolDefinitions converts runtime tools into provider-neutral model request
// definitions. Tool execution remains owned by loop.Tool.
func ToolDefinitions(tools []Tool) ([]ai.ToolDefinition, error) {
	definitions := make([]ai.ToolDefinition, 0, len(tools))
	names := make(map[string]struct{}, len(tools))
	for index, tool := range tools {
		if nilImplementation(tool) {
			return nil, fmt.Errorf("%w: tool at index %d is nil", ai.ErrInvalidToolDefinition, index)
		}
		if _, err := optionsForTool(tool); err != nil {
			return nil, err
		}
		params, err := tool.Params().JSONSchema()
		if err != nil {
			return nil, fmt.Errorf("tool %q: %w", tool.Name(), err)
		}
		name := tool.Name()
		definition, err := ai.NewToolDefinition(name, tool.Description(), params)
		if err != nil {
			return nil, fmt.Errorf("tool %q: %w", name, err)
		}
		if name != definition.Name {
			return nil, fmt.Errorf("%w: tool name %q must be canonical", ai.ErrInvalidToolDefinition, name)
		}
		if _, duplicate := names[definition.Name]; duplicate {
			return nil, fmt.Errorf("%w: duplicate tool name %q", ai.ErrInvalidToolDefinition, definition.Name)
		}
		names[definition.Name] = struct{}{}
		definitions = append(definitions, definition)
	}
	return definitions, nil
}

// ToolCallToString returns a diagnostic representation of tc.
func ToolCallToString(tc ai.ToolCall) string {
	var builder strings.Builder
	builder.WriteString("id: ")
	builder.WriteString(tc.ID)
	builder.WriteString(",type: ")
	builder.WriteString(tc.Type)
	builder.WriteString(",name: ")
	builder.WriteString(tc.Name)
	builder.WriteString(",arguments: ")
	builder.Write(tc.Args)
	return builder.String()
}
