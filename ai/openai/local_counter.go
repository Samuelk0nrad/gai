package openai

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/lace-ai/gai/ai"
	tiktoken "github.com/tiktoken-go/tokenizer"
)

const counterIDPrefix = "openai.tiktoken-go/v0.8.1:blocks-10000-runes-v1:"

// TokenCounterUnavailableError reports that no local counter mapping is known
// for a model. It intentionally does not guess an encoding for unknown models.
type TokenCounterUnavailableError struct {
	Model string
}

func (e *TokenCounterUnavailableError) Error() string {
	return fmt.Sprintf("openai token counter unavailable for model %q", e.Model)
}

func (e *TokenCounterUnavailableError) Unwrap() error { return ai.ErrTokenCounterUnsupported }

// NewTokenCounter returns a local count-only counter for a known OpenAI model.
// It uses bounded Unicode blocks, has estimated fidelity, and performs no
// provider I/O. Unknown models return TokenCounterUnavailableError. Codec
// initialization failures are returned by CountTokens instead of being hidden.
func NewTokenCounter(model string) (ai.TokenCounter, error) {
	model = strings.TrimSpace(model)
	encoding, ok := openAIEncodingForModel(model)
	if !ok {
		return nil, &TokenCounterUnavailableError{Model: model}
	}
	return &modelTokenCounter{encoding: encoding}, nil
}

// TokenCounter returns a local block counter for a known encoding. Unknown models
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

func (c *modelTokenCounter) ID() string                    { return counterIDPrefix + string(c.encoding) }
func (*modelTokenCounter) Fidelity() ai.TokenCountFidelity { return ai.TokenCountEstimated }
func (c *modelTokenCounter) CountTokens(ctx context.Context, text string) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if text == "" {
		return 0, nil
	}
	codec, err := localTokenCodec(ctx, c.encoding)
	if err != nil {
		return 0, fmt.Errorf("load OpenAI counter %q: %w", c.encoding, err)
	}
	return countTokenBlocks(ctx, codec, text)
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

// IsTokenCounterUnavailable reports whether err identifies an unsupported local
// OpenAI model-to-encoding mapping.
func IsTokenCounterUnavailable(err error) bool {
	var unavailable *TokenCounterUnavailableError
	return errors.As(err, &unavailable)
}
