package history

import (
	"context"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput"
)

// Summarizer produces the text of a compacted history summary. Compact calls it
// at most once per operation, before its single CAS, with detached inputs.
//
// NewContinuationSummarizer is the default when a CompactorDefinition supplies
// only a Model. agent/summary.Summarizer implements the separate-prompt text
// summarizer used before continuation summaries existed.
type Summarizer interface {
	SummarizeHistory(ctx context.Context, input SummaryInput) (SummaryOutput, error)
}

// SummaryInput describes one compaction. Turns and Kept partition the stored
// turns, oldest first. Summarizers must not retain or modify the inputs.
type SummaryInput struct {
	// Prefix is the request to continue, when the caller supplied one: an
	// unchanged earlier request that contains PreviousSummary and Turns in
	// their projected form. Nil for summarizers that build their own prompt.
	Prefix *ai.AIRequest
	// PreviousSummary is the stored summary that the new one must replace.
	PreviousSummary *Summary
	// Turns are the oldest turns to fold into the new summary.
	Turns []modelinput.Turn
	// Kept are the newer turns that stay unsummarized.
	Kept []modelinput.Turn
	// MaxTokens limits the summary output when positive. Zero selects the
	// summarizer's default, DefaultSummaryMaxTokens for a continuation
	// summary, which adds the continued request's reasoning BudgetTokens.
	MaxTokens int
	// ResponseFormat selects how the prefix's structured-output setting is used.
	ResponseFormat SummaryResponseFormat
	// RequestBudget limits the summary request. A zero budget skips the check.
	RequestBudget ai.RequestBudgetConfig
	// Counter estimates the summary request for RequestBudget.
	Counter ai.TokenCounter
}

// SummaryOutput is a generated summary.
type SummaryOutput struct {
	// Text replaces PreviousSummary.
	Text string
	// FoldedTurns is how many of the oldest input Turns the summary covers.
	// Zero means all of them. A summarizer may fold fewer turns, but at least
	// one, when the request would not fit otherwise.
	FoldedTurns int
	// Usage is the provider usage of the summary request, when reported.
	Usage         ai.Usage
	UsageReported bool
}

// SummaryResponseFormat selects how a continuation summary handles the
// ResponseFormat of the request it continues.
type SummaryResponseFormat uint8

const (
	// SummaryResponseFormatOmit removes the response format, so the summary is
	// plain text. This is the default. When the continued request has a
	// structured format, the change costs the prompt cache on providers that
	// key it on the format; Anthropic invalidates the conversation's cache.
	SummaryResponseFormatOmit SummaryResponseFormat = iota
	// SummaryResponseFormatKeep sends the response format unchanged. The
	// summary is then whatever text the model returns in that format.
	SummaryResponseFormatKeep
	// SummaryResponseFormatReject fails before the model call when the
	// continued request has a response format.
	SummaryResponseFormatReject
)

// Validate rejects unknown values.
func (f SummaryResponseFormat) Validate() error {
	if f > SummaryResponseFormatReject {
		return ErrInvalidSummaryResponseFormat
	}
	return nil
}
