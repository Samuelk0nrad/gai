package history

import (
	"context"
	"encoding/json"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

const historyToolResultPreviewRunes = 500

// Content is a minimal persisted history message representation.
type Content struct {
	Text  string
	Role  gaictx.Role
	Value gaictx.Content `json:"-"`
}

func (c Content) String() string {
	if c.Value != nil {
		return c.Value.String()
	}
	return c.Text
}

func (c Content) Type() string {
	if c.Value != nil {
		return c.Value.Type()
	}
	return gaictx.ContentTypeText
}

func (c Content) Marshal() ([]byte, error) {
	return json.Marshal(c)
}

func (c Content) Render(ctx context.Context) (gaictx.RenderNode, error) {
	if c.Value != nil {
		return c.Value.Render(ctx)
	}
	return gaictx.RenderNode{Type: gaictx.ContentTypeText, Value: c.Text}, nil
}

func MapMessageToContent(msg gaictx.Message) Content {
	if msg.Content == nil {
		return Content{Role: msg.Role}
	}
	return Content{
		Text:  msg.Content.String(),
		Role:  msg.Role,
		Value: msg.Content,
	}
}

// Part is the rendered prompt part emitted by HistorySource.
type Part struct {
	Contents   []Content
	TokenCount map[string]int
}

func (p *Part) Name() string {
	return "history"
}

func (p *Part) Render(ctx context.Context) (gaictx.RenderNode, error) {
	node := gaictx.RenderNode{Type: "history"}
	if p == nil || len(p.Contents) == 0 {
		return node, nil
	}

	for _, content := range p.Contents {
		child, err := content.Render(ctx)
		if err != nil {
			return gaictx.RenderNode{}, err
		}
		if content.Type() == gaictx.ContentTypeToolResult {
			truncateToolResult(&child)
		}
		if content.Role == "summary" {
			node.Children = append(node.Children, wrapContentNode("summary", child))
			continue
		}
		node.Children = append(node.Children, wrapContentNode(roleRenderType(content.Role), child))
	}
	return node, nil
}

func truncateToolResult(node *gaictx.RenderNode) {
	for index := range node.Children {
		if node.Children[index].Type != "result" {
			continue
		}
		runes := []rune(node.Children[index].Value)
		if len(runes) <= historyToolResultPreviewRunes {
			return
		}
		node.Children[index].Value = string(runes[:historyToolResultPreviewRunes]) + "\n[tool result truncated]"
		return
	}
}

func wrapContentNode(nodeType string, child gaictx.RenderNode) gaictx.RenderNode {
	node := gaictx.RenderNode{Type: nodeType}
	if child.Type == gaictx.ContentTypeText && len(child.Fields) == 0 && len(child.Children) == 0 {
		node.Value = child.Value
		return node
	}
	node.Children = []gaictx.RenderNode{child}
	return node
}

func roleRenderType(role gaictx.Role) string {
	if gaictx.IsValidRole(role) {
		return string(role)
	}
	return "message"
}

func (p *Part) Tokens(ctx context.Context, counter ai.TokenCounter) (int, error) {
	if counter == nil {
		return 0, gaictx.ErrTokenCounterNotFound
	}
	counterID := counter.ID()
	if count, ok := p.TokenCount[counterID]; ok && count >= 0 {
		return count, nil
	} else if ok {
		delete(p.TokenCount, counterID)
	}

	count := 0
	for _, content := range p.Contents {
		tokens, err := counter.CountTokens(ctx, content.String())
		if err != nil {
			return 0, err
		}
		count += tokens
	}
	p.saveTokens(counterID, count)
	return count, nil
}

func (p *Part) saveTokens(counterID string, tokens int) {
	if p.TokenCount == nil {
		p.TokenCount = make(map[string]int)
	}
	p.TokenCount[counterID] = tokens
}
