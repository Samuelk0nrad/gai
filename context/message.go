package context

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/lace-ai/gai"
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
	TokenCount  map[string]int
	debugSink   gai.ObservationSink
}

// StoredMessage wraps canonical content with persistence-only metadata. Message
// order is the enclosing Turn's slice order; Count orders the turns themselves.
type StoredMessage struct {
	SchemaVersion int        `json:"schema_version"`
	ID            string     `json:"id,omitempty"`
	SessionID     string     `json:"session_id,omitempty"`
	TurnID        string     `json:"turn_id,omitempty"`
	Message       ai.Message `json:"message"`
	// TokenCount keys identify the counter and its version.
	TokenCount map[string]int `json:"token_count,omitempty"`
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

// TurnTokenStore persists calculated token counts for a turn.
type TurnTokenStore interface {
	UpdateTurnTokens(ctx context.Context, turnID string, counter string, tokens int) error
}

// Tokenize returns the turn token count, using cached message or turn counts
// when available and optionally persisting a newly calculated value.
func (t *Turn) Tokenize(ctx context.Context, counter ai.TokenCounter, store TurnTokenStore) (int, error) {
	if t == nil {
		return 0, ErrMessageNotFound
	}
	if counter == nil {
		return 0, ErrTokenCounterNotFound
	}
	counterID := counter.ID()
	if count, ok := t.TokenCount[counterID]; ok && count >= 0 {
		return count, nil
	} else if ok {
		delete(t.TokenCount, counterID)
	}

	messages := t.messages()
	if count, ok := messagesTokenCount(messages, counterID); ok {
		return t.saveTokens(ctx, store, counterID, count)
	}

	content, err := combinedMessageContent(messages)
	if err != nil {
		return 0, err
	}
	count, err := counter.CountTokens(ctx, content)
	if err != nil {
		return 0, err
	}
	_, err = t.saveTokens(ctx, store, counterID, count)
	if err != nil {
		if gai.ObservationEnabled(ctx, t.debugSink) {
			gai.EmitObservation(ctx, t.debugSink, gai.Observation{
				Name:   "turn_token_save_failed",
				Source: "context:Turn.Tokenize",
				Fields: map[string]any{
					"turn_id":     t.ID,
					"turn_count":  t.Count,
					"counter_id":  counterID,
					"token_count": count,
				},
				Err: err,
			})
		}
	}
	return count, nil
}

// SetObservationSink configures diagnostics for non-fatal turn operations.
func (t *Turn) SetObservationSink(sink gai.ObservationSink) {
	t.debugSink = sink
}

func (t *Turn) saveTokens(ctx context.Context, store TurnTokenStore, counterID string, count int) (int, error) {
	if t.TokenCount == nil {
		t.TokenCount = make(map[string]int)
	}
	t.TokenCount[counterID] = count
	if store == nil || t.ID == "" {
		return count, nil
	}
	if err := store.UpdateTurnTokens(ctx, t.ID, counterID, count); err != nil {
		return 0, err
	}
	return count, nil
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

func messagesTokenCount(messages []StoredMessage, counterID string) (int, bool) {
	total := 0
	for _, message := range messages {
		count, ok := message.TokenCount[counterID]
		if !ok || count < 0 {
			return 0, false
		}
		total += count
	}
	return total, true
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

// Tokens returns the message token count for counter, caching the result.
func (m StoredMessage) Tokens(ctx context.Context, counter ai.TokenCounter) (int, error) {
	if counter == nil {
		return 0, ErrTokenCounterNotFound
	}
	counterID := counter.ID()
	if count, ok := m.TokenCount[counterID]; ok && count >= 0 {
		return count, nil
	} else if ok {
		delete(m.TokenCount, counterID)
	}
	content, err := messageTokenText(m.Message)
	if err != nil {
		return 0, err
	}
	count, err := counter.CountTokens(ctx, content)
	if err != nil {
		return 0, err
	}
	if m.TokenCount == nil {
		m.TokenCount = make(map[string]int)
	}
	m.TokenCount[counterID] = count
	return count, nil
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
