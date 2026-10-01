package ai

import (
	"context"
	"unicode/utf8"
)

// TokenCountFidelity describes the counting algorithm, not the accuracy of a
// full provider request or the number of tokens billed by a provider.
type TokenCountFidelity uint8

const (
	TokenCountUnknown TokenCountFidelity = iota
	// TokenCountExact counts the selected local text encoding exactly.
	TokenCountExact
	// TokenCountEstimated approximates text tokens.
	TokenCountEstimated
)

// TokenCounter counts text locally. Implementations must not perform network
// I/O, must honor cancellation, and must return counting failures as errors.
// ID identifies both the algorithm and its data version, so counts from
// different implementations cannot be confused. Counts exclude provider
// request framing unless a separate request-level API explicitly includes it.
type TokenCounter interface {
	CountTokens(context.Context, string) (int, error)
	ID() string
	Fidelity() TokenCountFidelity
}

// TokenCounterProvider is an optional model capability for automatic budgeting.
// Return a cheap local counter, a model-specific local estimator, or nil when
// neither is available. Returning nil selects GAI's generic estimator. Legacy
// Tokenizer methods are never consulted automatically.
type TokenCounterProvider interface {
	TokenCounter() TokenCounter
}

// TextTokenEstimator is the generic, allocation-free local fallback. It counts
// each contiguous ASCII run at one token per four bytes, rounded up, and each
// non-ASCII character at one token per UTF-8 byte. Invalid UTF-8 bytes count as
// one each. The Unicode allowance deliberately favors overestimation for CJK
// and emoji rather than assuming English-like token density. This heuristic
// can still undercount, especially for unusual ASCII text; it is not a
// guaranteed upper bound, an exact encoding count, or a full request count.
// Applications can supply a better local counter through their definition or
// per-run override.
type TextTokenEstimator struct{}

func (TextTokenEstimator) ID() string { return "gai.estimate/utf8-v2" }

func (TextTokenEstimator) Fidelity() TokenCountFidelity { return TokenCountEstimated }

func (TextTokenEstimator) CountTokens(ctx context.Context, text string) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	characters := 0
	count, asciiRemainder := 0, 0
	for len(text) > 0 {
		// Keep cancellation responsive even for large inputs.
		if characters%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		if text[0] < utf8.RuneSelf {
			if asciiRemainder == 0 {
				count++
			}
			asciiRemainder = (asciiRemainder + 1) % 4
			text = text[1:]
		} else {
			_, size := utf8.DecodeRuneInString(text)
			count += size
			asciiRemainder = 0
			text = text[size:]
		}
		characters++
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	return count, nil
}
