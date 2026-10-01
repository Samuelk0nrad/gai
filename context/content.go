package context

import (
	"context"
	"encoding/json"
	"fmt"
)

// Content decodes the legacy message payload format.
//
// Deprecated: use ai.ContentPart for all new conversation content.
type Content interface {
	String() string
	Type() string
	Marshal() ([]byte, error)
	Render(ctx context.Context) (RenderNode, error)
}

const (
	ContentTypeText          = "text"
	ContentTypeToolCall      = "tool_call"
	ContentTypeToolResult    = "tool_result"
	ContentTypeToolResultErr = "tool_result_err"
)

// TextContent contains plain message text.
type TextContent struct {
	Text string
}

func (c TextContent) String() string {
	return c.Text
}

func (c TextContent) Type() string {
	return ContentTypeText
}

func (c TextContent) Marshal() ([]byte, error) {
	return json.Marshal(c)
}

func (c TextContent) Render(ctx context.Context) (RenderNode, error) {
	return RenderNode{Type: ContentTypeText, Value: c.Text}, nil
}

// NewTextContent creates plain message content.
func NewTextContent(text string) TextContent {
	return TextContent{Text: text}
}

// ToolCallContent records a tool name and its serialized arguments.
type ToolCallContent struct {
	ToolName string
	Args     string
}

func (c ToolCallContent) String() string {
	return c.ToolName + "(" + c.Args + ")"
}

func (c ToolCallContent) Type() string {
	return ContentTypeToolCall
}

func (c ToolCallContent) Marshal() ([]byte, error) {
	return json.Marshal(c)
}

func (c ToolCallContent) Render(ctx context.Context) (RenderNode, error) {
	return RenderNode{
		Type:   ContentTypeToolCall,
		Fields: []RenderField{{Key: "name", Value: c.ToolName}},
		Children: []RenderNode{
			{Type: "arguments", Value: c.Args},
		},
	}, nil
}

// NewToolCallContent creates tool-call message content.
func NewToolCallContent(toolName, args string) ToolCallContent {
	return ToolCallContent{
		ToolName: toolName,
		Args:     args,
	}
}

// ToolResultContent records a successful tool result.
type ToolResultContent struct {
	ToolName          string
	Result            string
	Precomputed       bool
	PrecomputedResult string
}

func (c ToolResultContent) String() string {
	return c.ToolName + " result: " + c.Result
}

func (c ToolResultContent) Type() string {
	return ContentTypeToolResult
}

func (c ToolResultContent) Marshal() ([]byte, error) {
	return json.Marshal(c)
}

func (c ToolResultContent) Render(ctx context.Context) (RenderNode, error) {
	return RenderNode{
		Type:   ContentTypeToolResult,
		Fields: []RenderField{{Key: "name", Value: c.ToolName}},
		Children: []RenderNode{
			{Type: "result", Value: c.Result},
		},
	}, nil
}

// NewToolResultContent creates successful tool-result content.
func NewToolResultContent(toolName, result string, precomputed bool, precomputedResult string) ToolResultContent {
	return ToolResultContent{
		ToolName:          toolName,
		Result:            result,
		Precomputed:       precomputed,
		PrecomputedResult: precomputedResult,
	}
}

// ToolResultErrContent records a failed tool execution.
type ToolResultErrContent struct {
	ToolName string
	Err      string
}

func (c ToolResultErrContent) String() string {
	return c.ToolName + " error: " + c.Err
}

func (c ToolResultErrContent) Type() string {
	return ContentTypeToolResultErr
}

func (c ToolResultErrContent) Marshal() ([]byte, error) {
	return json.Marshal(c)
}

func (c ToolResultErrContent) Render(ctx context.Context) (RenderNode, error) {
	return RenderNode{
		Type:   ContentTypeToolResultErr,
		Fields: []RenderField{{Key: "name", Value: c.ToolName}},
		Children: []RenderNode{
			{Type: "error", Value: c.Err},
		},
	}, nil
}

// NewToolResultErrContent creates failed tool-result content.
func NewToolResultErrContent(toolName, err string) ToolResultErrContent {
	return ToolResultErrContent{
		ToolName: toolName,
		Err:      err,
	}
}

// NewContentFromType decodes a marshalled Content value using its type name.
func NewContentFromType(contentType string, data []byte) (Content, error) {
	switch contentType {
	case ContentTypeText:
		var textContent TextContent
		err := json.Unmarshal(data, &textContent)
		if err != nil {
			return nil, fmt.Errorf("%w %q: %w", ErrContentUnmarshal, contentType, err)
		}
		return textContent, nil
	case ContentTypeToolCall:
		var toolCallContent ToolCallContent
		err := json.Unmarshal(data, &toolCallContent)
		if err != nil {
			return nil, fmt.Errorf("%w %q: %w", ErrContentUnmarshal, contentType, err)
		}
		return toolCallContent, nil
	case ContentTypeToolResult:
		var toolResultContent ToolResultContent
		err := json.Unmarshal(data, &toolResultContent)
		if err != nil {
			return nil, fmt.Errorf("%w %q: %w", ErrContentUnmarshal, contentType, err)
		}
		return toolResultContent, nil
	case ContentTypeToolResultErr:
		var toolResultErrContent ToolResultErrContent
		err := json.Unmarshal(data, &toolResultErrContent)
		if err != nil {
			return nil, fmt.Errorf("%w %q: %w", ErrContentUnmarshal, contentType, err)
		}
		return toolResultErrContent, nil
	default:
		return nil, fmt.Errorf("%w: %s", ErrUnknownContentType, contentType)
	}
}
