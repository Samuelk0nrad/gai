package context

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/lace-ai/gai/ai"
)

// MessageSchemaVersion is the current serialized storage envelope version.
const MessageSchemaVersion = 1

// Turn groups one user message with the assistant and tool messages it caused.
type Turn struct {
	ID          string
	Count       int
	UserMessage *StoredMessage
	Messages    []StoredMessage
}

// StoredMessage wraps canonical content with persistence-only metadata. Message
// order is the enclosing Turn's slice order; Count orders the turns themselves.
type StoredMessage struct {
	SchemaVersion int        `json:"schema_version"`
	ID            string     `json:"id,omitempty"`
	SessionID     string     `json:"session_id,omitempty"`
	TurnID        string     `json:"turn_id,omitempty"`
	Message       ai.Message `json:"message"`
}

// MarshalJSON always emits the current schema version.
func (m StoredMessage) MarshalJSON() ([]byte, error) {
	type stored StoredMessage
	if m.SchemaVersion != 0 && m.SchemaVersion != MessageSchemaVersion {
		return nil, fmt.Errorf("unsupported message schema version: %d", m.SchemaVersion)
	}
	m.SchemaVersion = MessageSchemaVersion
	if err := m.Message.Validate(); err != nil {
		return nil, fmt.Errorf("stored message: %w", err)
	}
	return json.Marshal(stored(m))
}

// UnmarshalJSON requires the current schema and canonical message content.
func (m *StoredMessage) UnmarshalJSON(data []byte) error {
	type stored StoredMessage
	var decoded stored
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	if decoded.SchemaVersion != MessageSchemaVersion {
		return fmt.Errorf("unsupported message schema version: %d", decoded.SchemaVersion)
	}
	if err := decoded.Message.Validate(); err != nil {
		return fmt.Errorf("stored message: %w", err)
	}
	*m = StoredMessage(decoded)
	return nil
}

// Tokens counts the turn's combined canonical content on demand. Counting does
// not mutate the turn or persist counts. Callers may count shared turns
// concurrently when their content is read-only and the counter is concurrency-safe.
func (t *Turn) Tokens(ctx context.Context, counter ai.TokenCounter) (int, error) {
	if t == nil {
		return 0, ErrMessageNotFound
	}
	if counter == nil {
		return 0, ErrTokenCounterNotFound
	}
	content, err := combinedMessageContent(t.messages())
	if err != nil {
		return 0, err
	}
	return counter.CountTokens(ctx, content)
}

func (t *Turn) messages() []StoredMessage {
	if t == nil {
		return nil
	}
	messages := make([]StoredMessage, 0, len(t.Messages)+1)
	if t.UserMessage != nil {
		messages = append(messages, *t.UserMessage)
	}
	messages = append(messages, t.Messages...)
	return messages
}

func combinedMessageContent(messages []StoredMessage) (string, error) {
	var builder strings.Builder
	for i, message := range messages {
		if i > 0 {
			builder.WriteString("\n")
		}
		text, err := messageTokenText(message.Message)
		if err != nil {
			return "", err
		}
		builder.WriteString(text)
	}
	return builder.String(), nil
}

// IsValidRole reports whether role is one of the built-in roles.
func IsValidRole(role ai.Role) bool {
	switch role {
	case ai.RoleSystem, ai.RoleUser, ai.RoleAssistant, ai.RoleTool:
		return true
	default:
		return false
	}
}

// Tokens counts canonical message content on demand without mutating the value.
func (m StoredMessage) Tokens(ctx context.Context, counter ai.TokenCounter) (int, error) {
	if counter == nil {
		return 0, ErrTokenCounterNotFound
	}
	content, err := messageTokenText(m.Message)
	if err != nil {
		return 0, err
	}
	return counter.CountTokens(ctx, content)
}

// messageTokenText is a local estimate input, never a transport projection.
// It includes opaque bytes so token budgeting does not require a lossy renderer.
func messageTokenText(message ai.Message) (string, error) {
	if len(message.Parts) == 0 && len(message.Extensions) == 0 {
		return "", nil
	}
	if len(message.Extensions) == 0 && len(message.Parts) == 1 && message.Parts[0].Kind == ai.ContentText && len(message.Parts[0].Extensions) == 0 {
		return message.Parts[0].Text, nil
	}
	encoded, err := json.Marshal(message)
	return string(encoded), err
}
