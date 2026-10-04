package loop

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/lace-ai/gai/ai"
)

// ToolResult is the normalized result of an invocation or result processor.
// Err takes precedence over Text. Errors are immutable caller-owned values.
type ToolResult struct {
	Text string
	Err  error
}

// String returns the model-facing text, giving errors precedence over output.
func (r ToolResult) String() string {
	if r.Err != nil {
		return r.Err.Error()
	}
	return r.Text
}

func normalizeToolResult(text string, err error) ToolResult {
	if err != nil {
		return ToolResult{Err: err}
	}
	return ToolResult{Text: text}
}

// Tool defines a function that a model may request during a loop run.
// Implementations must be safe for concurrent use and honor context cancellation.
type Tool interface {
	Name() string
	Description() string
	Params() ai.ToolParameters
	// Function receives a call owned by this invocation. Err takes precedence over text.
	Function(context.Context, ai.ToolCall) (string, error)
}

// ToolFunc is the implementation of a text tool, without its declaration.
type ToolFunc func(context.Context, ai.ToolCall) (string, error)

type functionTool struct {
	name, description string
	paramsJSON        []byte
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
func (t *functionTool) Name() string        { return t.name }
func (t *functionTool) Description() string { return t.description }
func (t *functionTool) Params() ai.ToolParameters {
	var params ai.ToolParameters
	decoder := json.NewDecoder(bytes.NewReader(t.paramsJSON))
	decoder.UseNumber()
	if err := decoder.Decode(&params); err != nil {
		panic("loop: invalid private tool parameter snapshot")
	}
	return params
}
func (t *functionTool) Function(ctx context.Context, call ai.ToolCall) (string, error) {
	return t.function(ctx, call)
}

// CallTool validates a call and invokes the matching tool with an isolated copy.
// This low-level helper bypasses Loop execution policy and result processing.
// Invalid calls and missing tools return ordinary tool errors.
func CallTool(ctx context.Context, req ai.ToolCall, tools []Tool) ToolResult {
	if err := req.Validate(); err != nil {
		return ToolResult{Err: err}
	}
	if !json.Valid(req.Args) {
		return ToolResult{Err: ErrToolCallMalformed}
	}
	if err := ctx.Err(); err != nil {
		return ToolResult{Err: err}
	}
	for index, tool := range tools {
		if nilImplementation(tool) {
			return ToolResult{Err: fmt.Errorf("%w: tool at index %d is nil", ai.ErrInvalidToolDefinition, index)}
		}
		if tool.Name() == req.Name {
			text, err := tool.Function(ctx, req.Clone())
			return normalizeToolResult(text, err)
		}
	}
	return ToolResult{Err: fmt.Errorf("%w: %s", ErrToolNotFound, req.Name)}
}

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
