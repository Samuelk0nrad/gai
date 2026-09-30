package openai

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/lace-ai/gai/ai"
	tiktoken "github.com/tiktoken-go/tokenizer"
)

const tokenizerIDPrefix = "openai.tiktoken-go/v0.8.1:"

// TokenizerUnavailableError reports that no local tokenizer mapping is known
// for a model. It intentionally does not guess an encoding for unknown models.
type TokenizerUnavailableError struct {
	Model string
}

func (e *TokenizerUnavailableError) Error() string {
	return fmt.Sprintf("openai tokenizer unavailable for model %q", e.Model)
}

func (e *TokenizerUnavailableError) Unwrap() error { return ai.ErrTokenizerUnsupported }

// NewTokenizer returns a local tokenizer for a supported OpenAI model.
//
// Counts are exact for the selected text encoding, but do not include OpenAI
// request framing, messages, tools, or billing overhead. It performs no
// provider network I/O. Unknown models return a TokenizerUnavailableError so a
// caller can choose its own fallback; they are never assigned a guessed codec.
func NewTokenizer(model string) (ai.Tokenizer, error) {
	model = strings.TrimSpace(model)
	encoding, ok := openAIEncodingForModel(model)
	if !ok {
		return nil, &TokenizerUnavailableError{Model: model}
	}
	codec, err := tiktoken.Get(encoding)
	if err != nil {
		return nil, fmt.Errorf("load OpenAI tokenizer %q: %w", encoding, err)
	}
	return &Tokenizer{codec: codec, encoding: encoding}, nil
}

// Tokenizer is a local text-encoding tokenizer backed by tiktoken-go. It also
// implements the count-only runtime capability, without network I/O.
type Tokenizer struct {
	codec    tiktoken.Codec
	encoding tiktoken.Encoding
}

var _ ai.Tokenizer = (*Tokenizer)(nil)
var _ ai.TokenCounter = (*Tokenizer)(nil)

func (*Tokenizer) Fidelity() ai.TokenCountFidelity { return ai.TokenCountExact }

// TokenCounter returns a local counter for a known encoding. Unknown models
// return nil so runtime budgeting selects the generic estimator. Encoding
// initialization errors are returned by CountTokens rather than hidden behind
// an estimate.
func (m *Model) TokenCounter() ai.TokenCounter {
	encoding, ok := openAIEncodingForModel(strings.TrimSpace(m.name))
	if !ok {
		return nil
	}
	return &modelTokenCounter{encoding: encoding}
}

type modelTokenCounter struct {
	encoding tiktoken.Encoding
}

func (c *modelTokenCounter) ID() string                    { return tokenizerIDPrefix + string(c.encoding) }
func (*modelTokenCounter) Fidelity() ai.TokenCountFidelity { return ai.TokenCountExact }
func (c *modelTokenCounter) CountTokens(ctx context.Context, text string) (int, error) {
	return countLocalTokens(ctx, c.encoding, text, nil)
}

func (t *Tokenizer) ID() string { return tokenizerIDPrefix + string(t.encoding) }

// CountTokens preserves the pinned encoding's split boundaries and merge order.
// Cancellation is checked during counting work, including long pair merges.
func (t *Tokenizer) CountTokens(ctx context.Context, text string) (int, error) {
	return countLocalTokens(ctx, t.encoding, text, t.codec)
}

func countLocalTokens(ctx context.Context, encoding tiktoken.Encoding, text string, source tiktoken.Codec) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if text == "" {
		return 0, nil
	}
	codec, err := localCountCodec(ctx, encoding, source)
	if err != nil {
		return 0, fmt.Errorf("load OpenAI counter %q: %w", encoding, err)
	}
	count, err := codec.count(ctx, text)
	if err != nil {
		return 0, fmt.Errorf("count OpenAI tokens: %w", err)
	}
	return count, nil
}

func (t *Tokenizer) Tokenize(ctx context.Context, text string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	_, tokens, err := t.codec.Encode(text)
	if err != nil {
		return nil, fmt.Errorf("tokenize OpenAI text: %w", err)
	}
	return tokens, nil
}

func openAIEncodingForModel(model string) (tiktoken.Encoding, bool) {
	model = strings.ToLower(strings.TrimSpace(model))
	switch model {
	case "gpt-5", "gpt-5.1", GPT56, GPT56Terra, GPT56Sol, GPT56Luna, GPT41, GPT41Mini, GPT41Nano, GPT4o, GPT4oMini, "o1", O3, O3Mini, O4Mini:
		return tiktoken.O200kBase, true
	case "gpt-4":
		return tiktoken.Cl100kBase, true
	}
	for _, mapping := range []struct {
		prefix   string
		encoding tiktoken.Encoding
	}{
		{"gpt-5.1-", tiktoken.O200kBase},
		{"gpt-5-", tiktoken.O200kBase},
		{"gpt-4.1-", tiktoken.O200kBase},
		{"gpt-4o-", tiktoken.O200kBase},
		{"o1-", tiktoken.O200kBase},
		{"o3-", tiktoken.O200kBase},
		{"o4-", tiktoken.O200kBase},
		{"gpt-4-", tiktoken.Cl100kBase},
	} {
		if strings.HasPrefix(model, mapping.prefix) {
			return mapping.encoding, true
		}
	}
	return "", false
}

func isTokenizerAvailableForModel(model string) bool {
	_, ok := openAIEncodingForModel(model)
	return ok
}

// IsTokenizerUnavailable reports whether err identifies an unsupported local
// OpenAI model-to-encoding mapping.
func IsTokenizerUnavailable(err error) bool {
	var unavailable *TokenizerUnavailableError
	return errors.As(err, &unavailable)
}
