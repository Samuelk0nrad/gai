package ai

import "context"

// Tokenizer exposes the legacy concrete-provider tokenization API. Depending
// on the provider it can perform network I/O. Runtime budgeting uses the
// separate local-only TokenCounter contract and never selects Tokenizer
// automatically. Concrete provider APIs retain it for explicit callers.
type Tokenizer interface {
	// Tokenize splits text into the tokenizer's token representation.
	Tokenize(ctx context.Context, text string) ([]string, error)
	// CountTokens returns the number of tokens required for text.
	CountTokens(ctx context.Context, text string) (int, error)
	// ID returns a stable identifier for the tokenizer implementation.
	ID() string
}
