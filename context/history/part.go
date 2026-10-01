package history

import (
	"context"
	"encoding/json"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

const historyToolResultPreviewRunes = 500

// Part is the selected canonical history emitted by HistorySource. The shared
// projection applies the tool-result preview before either native mapping or
// fallback rendering, so both transports consume exactly the same content.
type Part struct {
	Messages   []ai.Message
	TokenCount map[string]int
}

func (p *Part) Name() string { return "history" }

func (p *Part) ConversationMessages() []ai.Message {
	if p == nil {
		return nil
	}
	messages := ai.CloneMessages(p.Messages)
	for mi := range messages {
		if len(messages[mi].Extensions) > 0 {
			continue
		}
		for pi := range messages[mi].Parts {
			part := &messages[mi].Parts[pi]
			if part.ToolResult == nil || part.ToolResult.IsError || len(part.Extensions) > 0 {
				continue
			}
			remaining := historyToolResultPreviewRunes
			markerAdded := false
			for ri := range part.ToolResult.Parts {
				result := &part.ToolResult.Parts[ri]
				// Only plain text can safely be truncated. JSON and provider
				// data must remain complete and valid.
				if result.Kind != ai.ContentText || len(result.Extensions) > 0 {
					continue
				}
				if markerAdded {
					result.Text = ""
					continue
				}
				runes := []rune(result.Text)
				if len(runes) > remaining {
					result.Text = string(runes[:remaining]) + "\n[tool result truncated]"
					remaining = 0
					markerAdded = true
				} else {
					remaining -= len(runes)
				}
			}
		}
	}
	return messages
}

func (p *Part) Render(ctx context.Context) (gaictx.RenderNode, error) {
	node := gaictx.RenderNode{Type: "history"}
	for _, message := range p.ConversationMessages() {
		child, err := gaictx.NewMessagePart(message).Render(ctx)
		if err != nil {
			return gaictx.RenderNode{}, err
		}
		node.Children = append(node.Children, child)
	}
	return node, nil
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
	for _, message := range p.ConversationMessages() {
		text := message.Text()
		if len(message.Parts) != 1 || message.Parts[0].Kind != ai.ContentText || len(message.Extensions) > 0 || len(message.Parts[0].Extensions) > 0 {
			encoded, err := json.Marshal(message)
			if err != nil {
				return 0, err
			}
			text = string(encoded)
		}
		tokens, err := counter.CountTokens(ctx, text)
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
