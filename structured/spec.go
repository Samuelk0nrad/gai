package structured

import (
	"encoding/json"
	"errors"
	"fmt"
)

// ValidationRetryPolicy opts into bounded retries for invalid model output.
// MaxRetries counts additional generations after the first invalid response.
type ValidationRetryPolicy struct {
	MaxRetries int
	// RetryPrompt can customize the corrective user message. Nil uses a concise
	// path-aware prompt. Returned text must not be empty.
	RetryPrompt func(error) string
}

// Validate checks the retry bound.
func (p ValidationRetryPolicy) Validate() error {
	if p.MaxRetries < 0 {
		return errors.New("structured validation MaxRetries must be non-negative")
	}
	return nil
}

// Prompt returns the correction sent with the next validation attempt.
func (p ValidationRetryPolicy) Prompt(err error) string {
	if p.RetryPrompt != nil {
		return p.RetryPrompt(err)
	}
	return fmt.Sprintf("The previous response was invalid (%v). Return only JSON that matches the requested schema.", err)
}

// Spec binds a schema to T and optional semantic validation and retry policy.
type Spec[T any] struct {
	Schema Schema
	// Validate runs after schema validation and typed decoding.
	Validate func(T) error
	Retry    ValidationRetryPolicy
}

// Result contains the typed value and an isolated copy of the exact accepted JSON.
type Result[T any] struct {
	Value T
	JSON  json.RawMessage
}

// Validate checks the schema and retry configuration without decoding output.
func (s Spec[T]) ValidateSpec() error {
	if err := s.Schema.validateConfiguration(); err != nil {
		return err
	}
	return s.Retry.Validate()
}

// Decode validates one complete JSON document and decodes it into T.
func (s Spec[T]) Decode(raw json.RawMessage) (Result[T], error) {
	var zero Result[T]
	if err := s.ValidateSpec(); err != nil {
		return zero, err
	}
	instance, err := decodeDocument(raw)
	if err != nil {
		return zero, &DecodeError{Path: errorPath(err), JSON: cloneJSON(raw), Err: err}
	}
	if err := s.Schema.compiled.Validate(instance); err != nil {
		return zero, validationError(err, raw)
	}
	var value T
	if err := json.Unmarshal(raw, &value); err != nil {
		return zero, &DecodeError{Path: decodePath(raw, err), JSON: cloneJSON(raw), Err: err}
	}
	if s.Validate != nil {
		if err := s.Validate(value); err != nil {
			var validation *ValidationError
			if errors.As(err, &validation) {
				result := *validation
				result.JSON = cloneJSON(raw)
				return zero, &result
			}
			return zero, &ValidationError{Path: "", Keyword: "semantic", JSON: cloneJSON(raw), Err: err}
		}
	}
	return Result[T]{Value: value, JSON: cloneJSON(raw)}, nil
}
