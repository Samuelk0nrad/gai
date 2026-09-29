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
// one token per four Unicode code points, rounded up. This deliberately simple
// estimate is not a guaranteed upper bound, an exact encoding count, or an
// accurate provider request count. Applications can supply a better local
// counter through their definition or per-run override.
type TextTokenEstimator struct{}

func (TextTokenEstimator) ID() string { return "gai.estimate/chars-v1" }

func (TextTokenEstimator) Fidelity() TokenCountFidelity { return TokenCountEstimated }

func (TextTokenEstimator) CountTokens(ctx context.Context, text string) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	characters := 0
	for len(text) > 0 {
		// Keep cancellation responsive even for large inputs.
		if characters%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		_, size := utf8.DecodeRuneInString(text)
		text = text[size:]
		characters++
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	count := characters / 4
	if characters%4 != 0 {
		count++
	}
	return count, nil
}
