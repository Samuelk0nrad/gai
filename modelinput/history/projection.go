package history

import (
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput"
)

const summaryMessagePrefix = "Conversation summary:\n"

// summaryMessage is the user message that carries a stored summary in prompts.
func summaryMessage(summary *Summary) ai.Message {
	parts := ai.TextParts(summaryMessagePrefix + summary.Content.Text)
	if len(summary.Content.Extensions) > 0 {
		// Opaque state belongs to the original part, whose text must stay
		// unchanged. Only plain summaries can coalesce the prefix.
		parts = ai.TextParts(summaryMessagePrefix)
		parts = append(parts, ai.CloneParts([]ai.ContentPart{summary.Content})...)
	}
	return ai.Message{Role: ai.RoleUser, Parts: parts}
}

// fullProjection renders the whole working history as the previous run sent
// it: the summary, then every turn. Older turns carry the history preview
// projection, as HistorySource selected them. The newest turn is raw, because
// that run sent it as live conversation rather than as selected history.
// previewRunes is the tool-result preview, encoded as in Part.
func fullProjection(state *HistoryState, previewRunes int) []ai.Message {
	if state == nil {
		return nil
	}
	var messages []ai.Message
	if state.Summary != nil {
		messages = append(messages, summaryMessage(state.Summary))
	}
	turns := sortTurnsByCount(append([]modelinput.Turn(nil), state.Turns...))
	for i := range turns {
		block := turnMessages(&turns[i])
		if i < len(turns)-1 {
			block = previewMessages(block, previewRunes)
		} else {
			block = ai.CloneMessages(block)
		}
		messages = append(messages, block...)
	}
	return messages
}
