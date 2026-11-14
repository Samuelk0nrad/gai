package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/Samuelk0nrad/gai/ai"
	antropic "github.com/anthropics/anthropic-sdk-go"
)

// Error is an error returned by the Anthropic API.
type Error struct {
	StatusCode int
	Type       string
	Message    string
}

func (e *Error) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.Type != "" {
		return fmt.Sprintf("anthropic API error (status %d, %s): %s", e.StatusCode, e.Type, e.Message)
	}
	return fmt.Sprintf("anthropic API error (status %d): %s", e.StatusCode, e.Message)
}

// RefusalError reports a response that ended with stop_reason "refusal". The
// model or a safety classifier declined the request; with server-side
// fallbacks on, every model in the chain declined it.
//
// Category, Explanation and RecommendedModel come from stop_details and may
// be empty: the API sends null when a refusal has no named category, no
// explanation or no recommended model. Category is an open set, such as
// "cyber", "bio", "frontier_llm", "reasoning_extraction" or "general_harms".
// RecommendedModel names a model to retry on directly when a fallback could
// not run.
//
// RefusalError unwraps to an *ai.TerminalError with Reason "refusal", so
// errors.Is(err, ai.ErrUnsuccessfulGeneration) and errors.As with
// *ai.TerminalError keep working.
type RefusalError struct {
	Category         string
	Explanation      string
	RecommendedModel string
}

func (e *RefusalError) Error() string {
	if e == nil {
		return "<nil>"
	}
	var b strings.Builder
	b.WriteString("anthropic generation ended unsuccessfully: refusal")
	if e.Category != "" {
		fmt.Fprintf(&b, " (category %s)", e.Category)
	}
	if e.Explanation != "" {
		fmt.Fprintf(&b, ": %s", e.Explanation)
	}
	return b.String()
}

func (e *RefusalError) Unwrap() error {
	return &ai.TerminalError{Provider: "anthropic", Reason: string(antropic.StopReasonRefusal)}
}

// newRefusalError reads stop_details. The SDK type has no recommended_model
// field yet, so it is read from the raw JSON.
func newRefusalError(details antropic.RefusalStopDetails) *RefusalError {
	refusal := &RefusalError{Category: string(details.Category), Explanation: details.Explanation}
	var extra struct {
		RecommendedModel *string `json:"recommended_model"`
	}
	if raw := details.RawJSON(); raw != "" && json.Unmarshal([]byte(raw), &extra) == nil && extra.RecommendedModel != nil {
		refusal.RecommendedModel = *extra.RecommendedModel
	}
	return refusal
}

func anthropicTerminalError(reason string, details antropic.RefusalStopDetails) error {
	switch reason {
	case "", "end_turn", "stop_sequence", "tool_use":
		return nil
	case string(antropic.StopReasonRefusal):
		return newRefusalError(details)
	default:
		return &ai.TerminalError{Provider: "anthropic", Reason: reason}
	}
}
