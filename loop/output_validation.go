package loop

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const (
	retryReasonStructuredDecode     = "structured_decode"
	retryReasonStructuredValidation = "structured_validation"
)

// OutputValidation validates a final no-tool model response before the loop
// commits it. MaxRetries is opt-in and counts additional model attempts.
type OutputValidation struct {
	Validate func(json.RawMessage) error
	// MaxRetries zero validates once without retrying.
	MaxRetries int
	// RetryPrompt customizes the temporary user correction for the next attempt.
	// Nil uses a concise path-aware default.
	RetryPrompt func(error) string
}

func (v OutputValidation) validate() error {
	switch {
	case v.Validate == nil:
		return fmt.Errorf("%w: validator is nil", ErrOutputValidationConfig)
	case v.MaxRetries < 0:
		return fmt.Errorf("%w: MaxRetries must be non-negative", ErrOutputValidationConfig)
	default:
		return nil
	}
}

func (v OutputValidation) prompt(err error) (string, error) {
	prompt := "The previous response was invalid. Return only JSON that matches the requested schema."
	if v.RetryPrompt != nil {
		prompt = v.RetryPrompt(err)
	}
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return "", fmt.Errorf("%w: retry prompt is empty", ErrOutputValidationConfig)
	}
	return prompt, nil
}

type structuredOutputError interface {
	StructuredOutputErrorKind() string
}

func outputValidationRetryReason(err error) string {
	var typed structuredOutputError
	if errors.As(err, &typed) && typed.StructuredOutputErrorKind() == retryReasonStructuredDecode {
		return retryReasonStructuredDecode
	}
	return retryReasonStructuredValidation
}
