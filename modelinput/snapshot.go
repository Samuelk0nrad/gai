package modelinput

import (
	"context"
	"encoding/json"

	"github.com/lace-ai/gai/ai"
)

// Snapshot contains the materialized request prefix. It contains no context
// sources, renderers, counters, or local token estimates. Restoring it never
// evaluates the original prompt factory or refreshes selected history.
type Snapshot struct {
	Context  []PartSnapshot         `json:"context,omitempty"`
	Messages []ai.Message           `json:"messages"`
	User     []ai.ContentPart       `json:"user,omitempty"`
	Budget   ai.RequestBudgetConfig `json:"budget"`
}

// Clone detaches canonical messages, user content and materialized context.
func (s Snapshot) Clone() Snapshot {
	s.Messages = ai.CloneMessages(s.Messages)
	s.User = ai.CloneParts(s.User)
	data, _ := json.Marshal(s.Context)
	s.Context = nil
	_ = json.Unmarshal(data, &s.Context)
	return s
}

// Snapshot captures a Builder after BuildContext has materialized its sources.
func (b *Builder) Snapshot(ctx context.Context) (Snapshot, error) {
	request, err := b.BuildRequest(ctx, nil)
	if err != nil {
		return Snapshot{}, err
	}
	parts, err := snapshotParts(ctx, b.input.Context)
	if err != nil {
		return Snapshot{}, err
	}
	return Snapshot{Context: parts, Messages: ai.CloneMessages(request.Messages), User: ai.CloneParts(b.input.User), Budget: b.RequestBudget()}, nil
}

// Restore creates a run-owned builder using exactly the saved canonical prefix.
func Restore(snapshot Snapshot) (*Builder, error) {
	if err := (ai.AIRequest{Messages: snapshot.Messages}).Validate(); err != nil {
		return nil, err
	}
	if err := snapshot.Budget.Validate(); err != nil {
		return nil, err
	}
	snapshot = snapshot.Clone()
	b := New(Definition{PromptInput: snapshot.Input(), TokenBudget: snapshot.Budget.Limit, OutputTokenReserve: snapshot.Budget.OutputReserve})
	b.frozen = &snapshot
	return b, nil
}

// MaterializeInput preserves the separation of machine context and user content
// while lowering interface-valued context parts to canonical messages once.
func MaterializeInput(ctx context.Context, input PromptInput) ([]ai.Message, error) {
	if len(input.User) == 0 && len(input.Context) == 0 {
		return nil, nil
	}
	b := New(Definition{PromptInput: input})
	if _, err := b.BuildContext(ctx); err != nil {
		return nil, err
	}
	request, err := b.BuildRequest(ctx, nil)
	return request.Messages, err
}

// PartSnapshot preserves a materialized input part for later workflow mapping.
// No original Part implementation is retained or invoked after restoration.
type PartSnapshot struct {
	PartName string       `json:"name"`
	Node     RenderNode   `json:"node"`
	Messages []ai.Message `json:"messages,omitempty"`
}
type frozenPart struct{ snapshot PartSnapshot }

// Name preserves the semantic name of the original context part.
func (p frozenPart) Name() string { return p.snapshot.PartName }

// Render returns frozen content without invoking the original renderer.
func (p frozenPart) Render(ctx context.Context) (RenderNode, error) {
	data, _ := json.Marshal(p.snapshot.Node)
	var node RenderNode
	_ = json.Unmarshal(data, &node)
	return node, ctx.Err()
}

// Tokens counts frozen content locally without refreshing its source.
func (p frozenPart) Tokens(ctx context.Context, counter ai.TokenCounter) (int, error) {
	data, err := json.Marshal(p.snapshot.Node)
	if err != nil {
		return 0, err
	}
	return counter.CountTokens(ctx, string(data))
}

type frozenConversationPart struct{ frozenPart }

// ConversationMessages returns a detached canonical context transcript.
func (p frozenConversationPart) ConversationMessages() []ai.Message {
	return ai.CloneMessages(p.snapshot.Messages)
}

// Input restores semantic user/context values, not original application types.
func (s Snapshot) Input() PromptInput {
	s = s.Clone()
	input := PromptInput{User: ai.CloneParts(s.User)}
	for _, part := range s.Context {
		p := frozenPart{snapshot: part}
		if part.Messages != nil {
			input.Context = append(input.Context, frozenConversationPart{p})
		} else {
			input.Context = append(input.Context, p)
		}
	}
	return input
}

// snapshotParts materializes interface-valued context for later workflow input mapping.
func snapshotParts(ctx context.Context, parts []Part) ([]PartSnapshot, error) {
	var result []PartSnapshot
	for _, part := range parts {
		if part == nil {
			continue
		}
		p := PartSnapshot{PartName: part.Name()}
		if canonical, ok := part.(ConversationPart); ok {
			p.Messages = ai.CloneMessages(canonical.ConversationMessages())
		} else {
			var err error
			p.Node, err = part.Render(ctx)
			if err != nil {
				return nil, err
			}
		}
		result = append(result, p)
	}
	return result, nil
}
