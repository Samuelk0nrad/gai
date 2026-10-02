package history

import (
	"context"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

const historyToolResultPreviewRunes = 500

// Part is the selected canonical history emitted by HistorySource. The shared
// projection applies the tool-result preview before either native mapping or
// fallback rendering, so both transports consume exactly the same content.
type Part struct {
	Messages []ai.Message
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
				// Inspect only the retained prefix and one extra rune. Repeated
				// local counts must not allocate a rune slice for the full result.
				end, used := len(result.Text), 0
				for index := range result.Text {
					if used == remaining {
						end = index
						break
					}
					used++
				}
				if end < len(result.Text) {
					// Preserve the existing normalization of invalid UTF-8 in
					// truncated text while converting only the retained prefix.
					result.Text = string([]rune(result.Text[:end])) + "\n[tool result truncated]"
					remaining = 0
					markerAdded = true
				} else {
					remaining -= used
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

// Tokens counts the selected, previewed messages on demand. It does not cache
// counts or mutate the part, so read-only parts can be counted concurrently with
// a concurrency-safe counter.
func (p *Part) Tokens(ctx context.Context, counter ai.TokenCounter) (int, error) {
	if counter == nil {
		return 0, gaictx.ErrTokenCounterNotFound
	}
	count := 0
	for _, message := range p.ConversationMessages() {
		tokens, err := (gaictx.StoredMessage{Message: message}).Tokens(ctx, counter)
		if err != nil {
			return 0, err
		}
		count += tokens
	}
	return count, nil
}
