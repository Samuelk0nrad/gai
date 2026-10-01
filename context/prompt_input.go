package context

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/lace-ai/gai/ai"
)

// PromptInput contains the run-specific user content and structured context.
// User may be nil for context-only agents. Context parts are rendered after
// configured context sources and before the user message.
type PromptInput struct {
	User    []ai.ContentPart
	Context []Part
}

// Clone deeply copies canonical user content and copies the context slice.
// Part values must be treated as immutable after they are supplied to a run.
func (i PromptInput) Clone() PromptInput {
	cloned := i
	cloned.User = ai.CloneParts(i.User)
	cloned.Context = append([]Part(nil), i.Context...)
	return cloned
}

// NamedPart is a token-countable scalar context value with a semantic name.
type NamedPart struct {
	name string
	text TextPart
}

// NewNamedPart creates a named scalar prompt part.
func NewNamedPart(name, value string) (*NamedPart, error) {
	name = strings.TrimSpace(name)
	if !validXMLName(name) {
		return nil, fmt.Errorf("%w: %q", ErrPromptPartName, name)
	}
	return &NamedPart{name: name, text: NewTextPart(value)}, nil
}

// NewJSONPart marshals value once and creates a named prompt part containing
// that exact JSON representation.
func NewJSONPart(name string, value any) (*NamedPart, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("marshal %s prompt part: %w", name, err)
	}
	return NewNamedPart(name, string(encoded))
}

func (p *NamedPart) Name() string {
	if p == nil {
		return ""
	}
	return p.name
}

func (p *NamedPart) Tokens(ctx context.Context, counter ai.TokenCounter) (int, error) {
	if p == nil {
		return 0, nil
	}
	return p.text.Tokens(ctx, counter)
}

func (p *NamedPart) Render(ctx context.Context) (RenderNode, error) {
	if err := ctx.Err(); err != nil {
		return RenderNode{}, err
	}
	if p == nil {
		return RenderNode{}, nil
	}
	return RenderNode{Type: p.name, Value: p.text.Content}, nil
}
