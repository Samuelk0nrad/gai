package history

import (
	"encoding/json"
	"fmt"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

// HistorySchemaVersion describes the canonical serialized history format.
// It is independent of a store revision.
const HistorySchemaVersion = 1

// HistoryState is working conversation history consumed by HistorySource.
// Turns contains the completed unsummarized tail; Summary represents older turns.
type HistoryState struct {
	SchemaVersion int `json:"schema_version"`
	Turns         []gaictx.Turn
	Summary       *Summary
}

// MarshalJSON versions newly persisted canonical state.
func (s HistoryState) MarshalJSON() ([]byte, error) {
	type state HistoryState
	if s.SchemaVersion != 0 && s.SchemaVersion != HistorySchemaVersion {
		return nil, fmt.Errorf("unsupported history schema version: %d", s.SchemaVersion)
	}
	s.SchemaVersion = HistorySchemaVersion
	return json.Marshal(state(s))
}

// UnmarshalJSON requires the current versioned history schema.
func (s *HistoryState) UnmarshalJSON(data []byte) error {
	type state HistoryState
	var decoded state
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	if decoded.SchemaVersion != HistorySchemaVersion {
		return fmt.Errorf("unsupported history schema version: %d", decoded.SchemaVersion)
	}
	*s = HistoryState(decoded)
	return nil
}

// Clone returns a detached copy, including nested canonical payloads. The input
// must remain stable while cloning; stores must synchronize concurrent writers.
// A nil state clones to nil.
func (s *HistoryState) Clone() *HistoryState {
	if s == nil {
		return nil
	}
	out := *s
	if s.Summary != nil {
		summary := *s.Summary
		summary.Content = ai.CloneParts([]ai.ContentPart{s.Summary.Content})[0]
		out.Summary = &summary
	}
	if s.Turns != nil {
		out.Turns = make([]gaictx.Turn, len(s.Turns))
		for i, turn := range s.Turns {
			out.Turns[i] = turn
			if turn.UserMessage != nil {
				user := *turn.UserMessage
				user.Message = user.Message.Clone()
				out.Turns[i].UserMessage = &user
			}
			if turn.Messages != nil {
				out.Turns[i].Messages = make([]gaictx.StoredMessage, len(turn.Messages))
				for j, message := range turn.Messages {
					out.Turns[i].Messages[j] = message
					out.Turns[i].Messages[j].Message = message.Message.Clone()
				}
			}
		}
	}
	return &out
}
