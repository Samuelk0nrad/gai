package context

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

// Role is retained as a source compatibility alias. New code should use ai.Role.
type Role = ai.Role

const (
	RoleSystem    = ai.RoleSystem
	RoleUser      = ai.RoleUser
	RoleAssistant = ai.RoleAssistant
	RoleTool      = ai.RoleTool
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

// UnmarshalJSON accepts canonical envelopes and legacy plain-text messages.
// Legacy tool payloads have no reliable call ID, so they must be migrated by the
// application using its authoritative call records rather than guessed by name.
func (m *StoredMessage) UnmarshalJSON(data []byte) error {
	type stored StoredMessage
	var canonical stored
	if err := json.Unmarshal(data, &canonical); err != nil {
		return err
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(data, &keys); err != nil {
		return err
	}
	hasMessage := false
	for key := range keys {
		if strings.EqualFold(key, "message") {
			hasMessage = true
		}
	}
	if hasMessage {
		if canonical.SchemaVersion != MessageSchemaVersion {
			return fmt.Errorf("unsupported message schema version: %d", canonical.SchemaVersion)
		}
		if err := canonical.Message.Validate(); err != nil {
			return fmt.Errorf("stored message: %w", err)
		}
		*m = StoredMessage(canonical)
		return nil
	}
	if canonical.SchemaVersion != 0 {
		return fmt.Errorf("versioned message envelope requires message content")
	}
	var legacy struct {
		ID         string
		SessionID  string
		TurnID     string
		Role       ai.Role
		Content    json.RawMessage
		TokenCount map[string]int
	}
	if err := json.Unmarshal(data, &legacy); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if len(legacy.Content) > 0 && string(legacy.Content) != "null" {
		if err := json.Unmarshal(legacy.Content, &fields); err != nil {
			return fmt.Errorf("decode legacy message content: %w", err)
		}
		for key := range fields {
			if !strings.EqualFold(key, "text") {
				return fmt.Errorf("legacy structured message cannot be migrated without authoritative tool call IDs: %s", key)
			}
		}
	}
	var text struct{ Text string }
	if len(fields) > 0 {
		if err := json.Unmarshal(legacy.Content, &text); err != nil {
			return err
		}
	}
	if legacy.Role == ai.RoleTool {
		return fmt.Errorf("legacy tool message requires authoritative tool call IDs")
	}
	if !IsValidRole(legacy.Role) {
		return fmt.Errorf("invalid legacy message role: %q", legacy.Role)
	}
	*m = StoredMessage{SchemaVersion: MessageSchemaVersion, ID: legacy.ID, SessionID: legacy.SessionID, TurnID: legacy.TurnID, Message: ai.TextMessage(legacy.Role, text.Text), TokenCount: legacy.TokenCount}
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
func IsValidRole(role Role) bool {
	switch role {
	case RoleSystem, RoleUser, RoleAssistant, RoleTool:
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
