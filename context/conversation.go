package context

import "github.com/lace-ai/gai/ai"

// Conversation exposes the canonical ordered messages for a model request.
type Conversation interface {
	Messages() []ai.Message
}

// ConversationPart exposes selected canonical messages from a context source.
// Selection, summarization, and truncation must be applied before returning them.
// Builders preserve these messages without rendering away their structured parts.
type ConversationPart interface {
	Part
	ConversationMessages() []ai.Message
}
