package history

import (
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

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
