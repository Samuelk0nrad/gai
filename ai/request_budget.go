package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
)

// RequestCountingMode selects local estimation or explicit provider preflight.
type RequestCountingMode uint8

const (
	// RequestCountEstimate performs local work only and is the default.
	RequestCountEstimate RequestCountingMode = iota
	// RequestCountAccurate requires the model's complete-request capability.
	// Provider preflight can still differ from reported generation usage.
	RequestCountAccurate
)

// RequestBudgetConfig defines independent input and total-window limits.
// A zero limit disables that limit. Limit includes input, OutputReserve and
// SafetyMargin; InputLimit applies only to input plus SafetyMargin. OutputReserve
// is increased to the finalized request's MaxTokens when that is larger.
// SafetyMargin is an absolute token allowance, applied in either mode; zero
// adds no allowance. No model output default is guessed when MaxTokens is zero.
type RequestBudgetConfig struct {
	Limit         int
	InputLimit    int
	OutputReserve int
	SafetyMargin  int
	Mode          RequestCountingMode
}

var (
	ErrInvalidRequestBudget       = errors.New("invalid request budget")
	ErrRequestBudgetExceeded      = errors.New("request token budget exceeded")
	ErrRequestCountFailed         = errors.New("request token count failed")
	ErrInputTokenCountUnsupported = errors.New("accurate input token counting unsupported")
)

// Validate checks configuration without performing counting or provider I/O.
func (c RequestBudgetConfig) Validate() error {
	if c.Limit < 0 || c.InputLimit < 0 || c.OutputReserve < 0 || c.SafetyMargin < 0 {
		return fmt.Errorf("%w: limits, reserve and margin must be non-negative", ErrInvalidRequestBudget)
	}
	if c.Mode != RequestCountEstimate && c.Mode != RequestCountAccurate {
		return fmt.Errorf("%w: unknown counting mode %d", ErrInvalidRequestBudget, c.Mode)
	}
	if _, err := AddTokenCounts(c.OutputReserve, c.SafetyMargin, c.InputLimit); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidRequestBudget, err)
	}
	return nil
}

// InputTokenCounter is an optional complete-request capability. It may perform
// network I/O and is called only in explicitly selected accurate mode. The
// implementation must count equivalent generation inputs or return unsupported;
// local text tokenization alone does not satisfy this contract.
type InputTokenCounter interface {
	CountInputTokens(context.Context, AIRequest) (int, error)
}

// InputTokenCountUnsupportedError identifies an unavailable accurate mechanism.
type InputTokenCountUnsupportedError struct {
	Model  string
	Reason string
}

func (e *InputTokenCountUnsupportedError) Error() string {
	return fmt.Sprintf("%s: model=%q: %s", ErrInputTokenCountUnsupported, e.Model, e.Reason)
}
func (*InputTokenCountUnsupportedError) Unwrap() error { return ErrInputTokenCountUnsupported }

// RequestCountError preserves local/provider failures, including cancellation.
type RequestCountError struct{ Cause error }

func (e *RequestCountError) Error() string {
	return fmt.Sprintf("%s: %v", ErrRequestCountFailed, e.Cause)
}
func (e *RequestCountError) Unwrap() []error { return []error{ErrRequestCountFailed, e.Cause} }

// RequestBudgetResult records the decision for one finalized generation input.
// Fidelity describes the whole-request method and remains estimated even when
// CounterFidelity is exact for text. Breakdowns are diagnostic estimates, not
// reusable per-message counts. Provider preflight reports only InputTokens.
type RequestBudgetResult struct {
	InputTokens        int
	MessageTokens      int
	ToolTokens         int
	OptionTokens       int
	FramingTokens      int
	CheckpointTokens   int
	Method             string
	Fidelity           TokenCountFidelity
	CounterID          string
	CounterFidelity    TokenCountFidelity
	Limit              int
	InputLimit         int
	OutputReserve      int
	SafetyMargin       int
	TotalTokens        int
	OutputLimitUnknown bool
}

// RequestBudgetExceededError carries the rejected request's budget diagnostics.
type RequestBudgetExceededError struct{ Budget RequestBudgetResult }

func (e *RequestBudgetExceededError) Error() string {
	return fmt.Sprintf("%s: input=%d total=%d window=%d input_limit=%d reserve=%d margin=%d method=%s",
		ErrRequestBudgetExceeded, e.Budget.InputTokens, e.Budget.TotalTokens, e.Budget.Limit,
		e.Budget.InputLimit, e.Budget.OutputReserve, e.Budget.SafetyMargin, e.Budget.Method)
}
func (*RequestBudgetExceededError) Unwrap() error { return ErrRequestBudgetExceeded }

// AddTokenCounts rejects negative values and integer overflow in accounting.
func AddTokenCounts(counts ...int) (int, error) {
	maxInt := int(^uint(0) >> 1)
	total := 0
	for _, count := range counts {
		if count < 0 || count > maxInt-total {
			return 0, fmt.Errorf("invalid or overflowing token count: %d", count)
		}
		total += count
	}
	return total, nil
}

// RequestEstimateID versions the portable input projection. Plain text is
// concatenated literally with four estimated framing tokens per message.
// Structured/opaque messages use canonical JSON, including their framing;
// native tools and non-default request options use separate JSON projections.
// Three estimated tokens cover the response prefix. This is not native framing
// or an exact provider encoding/billing count.
const RequestEstimateID = "gai.request/canonical-messages-v1"

// EstimateRequestTokens estimates a finalized request using local work only.
// Rendered tool protocols already live in Messages; only native Tools add a
// separate schema cost. MaxTokens is output capacity, not input content.
func EstimateRequestTokens(ctx context.Context, req AIRequest, counter TokenCounter) (RequestBudgetResult, error) {
	result := RequestBudgetResult{Method: "local_estimate", Fidelity: TokenCountEstimated, FramingTokens: 3}
	if unavailableCounter(counter) {
		return result, &RequestCountError{Cause: errors.New("local counter unavailable")}
	}
	result.CounterID = RequestEstimateID + ":" + counter.ID()
	result.CounterFidelity = counter.Fidelity()
	var err error
	var messageFraming int
	result.MessageTokens, messageFraming, err = estimateMessages(ctx, req.Messages, counter)
	if err != nil {
		return result, err
	}
	result.FramingTokens, err = AddTokenCounts(result.FramingTokens, messageFraming)
	if err != nil {
		return result, &RequestCountError{Cause: err}
	}
	if len(req.Tools) > 0 {
		result.ToolTokens, err = countRequestJSON(ctx, req.Tools, counter)
		if err != nil {
			return result, err
		}
	}
	if req.ToolChoice.Mode != "" || len(req.ToolChoice.Names) > 0 || req.ResponseFormat.Type != "" || req.Reasoning != (ReasoningConfig{}) {
		options := struct {
			ToolChoice     ToolChoice
			ResponseFormat ResponseFormat
			Reasoning      ReasoningConfig
		}{req.ToolChoice, req.ResponseFormat, req.Reasoning}
		result.OptionTokens, err = countRequestJSON(ctx, options, counter)
		if err != nil {
			return result, err
		}
	}
	result.InputTokens, err = AddTokenCounts(result.MessageTokens, result.ToolTokens, result.OptionTokens, result.FramingTokens)
	if err != nil {
		return result, &RequestCountError{Cause: err}
	}
	return result, nil
}

// EstimateMessageTokens estimates newly appended message envelopes for a valid
// reported-usage checkpoint. It adds no request-wide tools/options/framing cost.
func EstimateMessageTokens(ctx context.Context, messages []Message, counter TokenCounter) (int, error) {
	content, framing, err := estimateMessages(ctx, messages, counter)
	if err != nil {
		return 0, err
	}
	total, err := AddTokenCounts(content, framing)
	if err != nil {
		return 0, &RequestCountError{Cause: err}
	}
	return total, nil
}

func estimateMessages(ctx context.Context, messages []Message, counter TokenCounter) (int, int, error) {
	if err := ctx.Err(); err != nil {
		return 0, 0, &RequestCountError{Cause: err}
	}
	if unavailableCounter(counter) {
		return 0, 0, &RequestCountError{Cause: errors.New("local counter unavailable")}
	}
	count, framing := 0, 0
	for _, message := range messages {
		plain := len(message.Extensions) == 0 && len(message.Parts) > 0
		for _, part := range message.Parts {
			plain = plain && part.Kind == ContentText && len(part.Extensions) == 0
		}
		var tokens int
		var err error
		if plain {
			text := message.Parts[0].Text
			if len(message.Parts) > 1 {
				var combined strings.Builder
				for _, part := range message.Parts {
					combined.WriteString(part.Text)
				}
				text = combined.String()
			}
			tokens, err = countLocalText(ctx, text, counter)
			if err == nil {
				framing, err = AddTokenCounts(framing, 4)
			}
		} else {
			tokens, err = countRequestJSON(ctx, message, counter)
		}
		if err != nil {
			return 0, 0, &RequestCountError{Cause: err}
		}
		count, err = AddTokenCounts(count, tokens)
		if err != nil {
			return 0, 0, &RequestCountError{Cause: err}
		}
	}
	return count, framing, nil
}

func countRequestJSON(ctx context.Context, value any, counter TokenCounter) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, &RequestCountError{Cause: err}
	}
	payload, err := json.Marshal(value)
	if err != nil {
		return 0, &RequestCountError{Cause: err}
	}
	return countLocalText(ctx, string(payload), counter)
}

func countLocalText(ctx context.Context, text string, counter TokenCounter) (int, error) {
	if unavailableCounter(counter) {
		return 0, &RequestCountError{Cause: errors.New("local counter unavailable")}
	}
	if err := ctx.Err(); err != nil {
		return 0, &RequestCountError{Cause: err}
	}
	tokens, err := counter.CountTokens(ctx, text)
	if err == nil {
		_, err = AddTokenCounts(tokens)
	}
	if err == nil {
		err = ctx.Err()
	}
	if err != nil {
		return 0, &RequestCountError{Cause: err}
	}
	return tokens, nil
}

func unavailableCounter(counter TokenCounter) bool {
	if counter == nil {
		return true
	}
	v := reflect.ValueOf(counter)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	}
	return false
}
