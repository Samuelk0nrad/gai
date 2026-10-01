package context

import (
	"context"
	"strconv"

	"github.com/lace-ai/gai/ai"
)

// Part is a token-countable unit that can produce a renderer-neutral node.
type Part interface {
	Name() string
	Tokens(ctx context.Context, counter ai.TokenCounter) (int, error)
	Render(ctx context.Context) (RenderNode, error)
}

// TextPart is a plain-text prompt part.
type TextPart struct {
	Content string
}

// NewTextPart creates a plain-text part. Counting does not mutate the part.
func NewTextPart(content string) TextPart {
	return TextPart{
		Content: content,
	}
}

func (t TextPart) Name() string {
	return "text"
}

func (t TextPart) Tokens(ctx context.Context, counter ai.TokenCounter) (int, error) {
	if counter == nil {
		return 0, ErrTokenCounterNotFound
	}
	return counter.CountTokens(ctx, t.Content)
}

func (t TextPart) Render(ctx context.Context) (RenderNode, error) {
	return RenderNode{Type: "text", Value: t.Content}, nil
}

// MessagePart wraps canonical conversation content for context sources and
// standalone renderers. Builders consume ConversationMessages directly.
type MessagePart struct{ Message ai.Message }

func NewMessagePart(message ai.Message) MessagePart { return MessagePart{Message: message.Clone()} }

func (m MessagePart) Name() string { return "message" }

func (m MessagePart) ConversationMessages() []ai.Message { return []ai.Message{m.Message.Clone()} }

func (m MessagePart) Tokens(ctx context.Context, counter ai.TokenCounter) (int, error) {
	if counter == nil {
		return 0, ErrTokenCounterNotFound
	}
	text, err := messageTokenText(m.Message)
	if err != nil {
		return 0, err
	}
	return counter.CountTokens(ctx, text)
}

func (m MessagePart) Render(ctx context.Context) (RenderNode, error) {
	// Enforce the canonical fallback's capability policy before formatting.
	if _, err := ai.RenderMessages(ctx, []ai.Message{m.Message}); err != nil {
		return RenderNode{}, err
	}
	node := RenderNode{Type: roleRenderType(m.Message.Role)}
	for _, part := range m.Message.Parts {
		child := RenderNode{Type: string(part.Kind)}
		switch part.Kind {
		case ai.ContentText, ai.ContentReasoning:
			child.Value = part.Text
		case ai.ContentJSON:
			child.Value = string(part.JSON)
		case ai.ContentToolCall:
			child.Fields = []RenderField{{Key: "id", Value: part.ToolCall.ID}, {Key: "name", Value: part.ToolCall.Name}}
			child.Children = []RenderNode{{Type: "arguments", Value: string(part.ToolCall.Args)}}
		case ai.ContentToolResult:
			result := part.ToolResult
			child.Fields = []RenderField{{Key: "id", Value: result.ToolCallID}, {Key: "name", Value: result.Name}, {Key: "is_error", Value: strconv.FormatBool(result.IsError)}}
			for _, resultPart := range result.Parts {
				value := resultPart.Text
				if resultPart.Kind == ai.ContentJSON {
					value = string(resultPart.JSON)
				}
				child.Children = append(child.Children, RenderNode{Type: "result", Value: value})
			}
		}
		node.Children = append(node.Children, child)
	}
	if len(node.Children) == 1 && node.Children[0].Type == string(ai.ContentText) {
		node.Value = node.Children[0].Value
		node.Children = nil
	}
	return node, nil
}

func roleRenderType(role ai.Role) string {
	if IsValidRole(role) {
		return string(role)
	}
	return "message"
}

// SystemPart groups multiple parts under one system-instruction node.
type SystemPart struct {
	Instructions []Part
}

// NewSystemPart creates a grouped system-instruction part.
func NewSystemPart(instructions []Part) SystemPart {
	return SystemPart{
		Instructions: instructions,
	}
}

func (i SystemPart) Name() string {
	return "system"
}

func (i SystemPart) Tokens(ctx context.Context, counter ai.TokenCounter) (int, error) {
	count := 0
	for _, part := range i.Instructions {
		tokens, err := part.Tokens(ctx, counter)
		if err != nil {
			return 0, err
		}
		count += tokens
	}
	return count, nil
}

func (i SystemPart) Render(ctx context.Context) (RenderNode, error) {
	node := RenderNode{Type: "instructions"}
	for _, part := range i.Instructions {
		child, err := part.Render(ctx)
		if err != nil {
			return RenderNode{}, err
		}
		node.Children = append(node.Children, child)
	}
	return node, nil
}
