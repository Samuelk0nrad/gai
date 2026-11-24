package history

import (
	"context"
	"strings"
	"unicode/utf8"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput"
)

// DefaultToolResultPreviewRunes is how many runes of each plain-text tool
// result history keeps when no WithToolResultPreview option is given.
const DefaultToolResultPreviewRunes = 500

// toolResultTruncatedMarker follows a tool result preview that omitted text.
const toolResultTruncatedMarker = "\n[tool result truncated]"

// Option configures a HistorySource or a Compactor.
type Option func(*options)

type options struct {
	// previewRunes uses Part's encoding: zero selects the default, negative
	// keeps tool results whole.
	previewRunes int
}

func applyOptions(opts []Option) options {
	var o options
	for _, opt := range opts {
		if opt != nil {
			opt(&o)
		}
	}
	return o
}

// WithToolResultPreview sets how many runes of each plain-text tool result
// history keeps, followed by a truncation marker when text is omitted. Zero or
// a negative value keeps tool results whole. Without this option history keeps
// DefaultToolResultPreviewRunes. Selection, token counting and rendering all use
// the same projection.
//
// Give a HistorySource and a Compactor of the same session the same value:
// the compactor selects and counts the history the prompt carries.
// HistorySource.CompactHistory uses the source's value.
func WithToolResultPreview(runes int) Option {
	return func(o *options) {
		if runes <= 0 {
			o.previewRunes = -1
			return
		}
		o.previewRunes = runes
	}
}

// Part is the selected canonical history emitted by HistorySource. The shared
// projection applies the tool-result preview before either native mapping or
// fallback rendering, so both transports consume exactly the same content.
type Part struct {
	Messages []ai.Message

	// previewRunes is the tool-result preview length. Zero selects
	// DefaultToolResultPreviewRunes, so a Part built by hand keeps the default;
	// negative keeps tool results whole.
	previewRunes int
}

func (p *Part) Name() string { return "history" }

// previewLimit returns the rune limit, or -1 when results stay whole.
func previewLimit(previewRunes int) int {
	switch {
	case previewRunes == 0:
		return DefaultToolResultPreviewRunes
	case previewRunes < 0:
		return -1
	default:
		return previewRunes
	}
}

func (p *Part) ConversationMessages() []ai.Message {
	if p == nil {
		return nil
	}
	return previewMessages(p.Messages, p.previewRunes)
}

// previewMessages returns detached messages with the tool-result preview of
// previewRunes, encoded as in Part, applied.
func previewMessages(source []ai.Message, previewRunes int) []ai.Message {
	messages := ai.CloneMessages(source)
	limit := previewLimit(previewRunes)
	if limit < 0 {
		return messages
	}
	for mi := range messages {
		if len(messages[mi].Extensions) > 0 {
			continue
		}
		for pi := range messages[mi].Parts {
			part := &messages[mi].Parts[pi]
			if previewable(part) {
				previewToolResult(part.ToolResult, limit)
			}
		}
	}
	return messages
}

// previewable reports whether a part of a message without extensions takes
// the tool-result preview.
func previewable(part *ai.ContentPart) bool {
	return part.ToolResult != nil && !part.ToolResult.IsError && len(part.Extensions) == 0
}

// previewableText reports whether a tool result part is plain text that the
// preview may shorten.
func previewableText(part *ai.ContentPart) bool {
	return part.Kind == ai.ContentText && len(part.Extensions) == 0
}

// previewToolResult keeps limit runes of the result's plain text in place.
func previewToolResult(toolResult *ai.ToolResult, limit int) {
	remaining := limit
	markerAdded := false
	for ri := range toolResult.Parts {
		result := &toolResult.Parts[ri]
		// Only plain text can safely be truncated. JSON and provider
		// data must remain complete and valid.
		if !previewableText(result) {
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
			result.Text = string([]rune(result.Text[:end])) + toolResultTruncatedMarker
			remaining = 0
			markerAdded = true
		} else {
			remaining -= used
		}
	}
}

// previewedLength returns the preview length a projected tool result was cut
// at, or false when it carries no truncation marker. The preview writes the
// marker whole into one part and empties the plain text after it, so the last
// nonempty plain-text part must end with it. Runes are counted per part, as
// previewToolResult counts them, without joining the text.
func previewedLength(toolResult *ai.ToolResult) (int, bool) {
	runes, last := 0, ""
	for i := range toolResult.Parts {
		part := &toolResult.Parts[i]
		if previewableText(part) && part.Text != "" {
			runes += utf8.RuneCountInString(part.Text)
			last = part.Text
		}
	}
	if !strings.HasSuffix(last, toolResultTruncatedMarker) {
		return 0, false
	}
	return runes - utf8.RuneCountInString(toolResultTruncatedMarker), true
}

func (p *Part) Render(ctx context.Context) (modelinput.RenderNode, error) {
	node := modelinput.RenderNode{Type: "history"}
	for _, message := range p.ConversationMessages() {
		child, err := modelinput.NewMessagePart(message).Render(ctx)
		if err != nil {
			return modelinput.RenderNode{}, err
		}
		node.Children = append(node.Children, child)
	}
	return node, nil
}

// Tokens estimates the selected, previewed messages including message framing,
// using the same projection as the finalized request guard. It does not cache
// counts or mutate the part, so read-only parts can be counted concurrently with
// a concurrency-safe counter.
func (p *Part) Tokens(ctx context.Context, counter ai.TokenCounter) (int, error) {
	if counter == nil {
		return 0, modelinput.ErrTokenCounterNotFound
	}
	return ai.EstimateMessageTokens(ctx, p.ConversationMessages(), counter)
}
