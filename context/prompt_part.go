package context

import (
	"context"

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

// MessagePart associates message content with a conversation role.
type MessagePart struct {
	Role    Role
	Content Content
}

// NewMessagePart creates a role-aware message part.
func NewMessagePart(role Role, content Content) MessagePart {
	return MessagePart{
		Role:    role,
		Content: content,
	}
}

func (m MessagePart) Name() string {
	return "message"
}

func (m MessagePart) Tokens(ctx context.Context, counter ai.TokenCounter) (int, error) {
	if counter == nil {
		return 0, ErrTokenCounterNotFound
	}
	content := ""
	if m.Content != nil {
		content = m.Content.String()
	}
	return counter.CountTokens(ctx, content)
}

func (m MessagePart) Render(ctx context.Context) (RenderNode, error) {
	node := RenderNode{
		Type: roleRenderType(m.Role),
	}
	if m.Content == nil {
		return node, nil
	}
	child, err := m.Content.Render(ctx)
	if err != nil {
		return RenderNode{}, err
	}
	if child.Type == ContentTypeText && len(child.Fields) == 0 && len(child.Children) == 0 {
		node.Value = child.Value
		return node, nil
	}
	node.Children = []RenderNode{child}
	return node, nil
}

func roleRenderType(role Role) string {
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
