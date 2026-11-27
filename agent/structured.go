package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/loop"
	"github.com/Samuelk0nrad/gai/structured"
)

// RunStructured runs one workflow with a provider-neutral JSON Schema response
// format, validates the primary agent's final response before it is committed,
// and returns the accepted response as both T and exact JSON. Validation retries
// are disabled unless Spec.Retry.MaxRetries is positive.
//
// Durable workflows are not supported because rejected validation attempts are
// not yet represented in durable continuation checkpoints.
func RunStructured[T any](ctx context.Context, a *Agent, input RunInput, spec structured.Spec[T]) (structured.Result[T], error) {
	var zero structured.Result[T]
	if err := spec.ValidateSpec(); err != nil {
		return zero, err
	}
	if input.Durability != nil {
		return zero, fmt.Errorf("%w: durable workflows are not supported", loop.ErrOutputValidationConfig)
	}

	input = cloneRunInput(input)
	if input.Execution == nil {
		input.Execution = &ExecutionOverrides{}
	}
	input.Execution.ResponseFormat = &ai.ResponseFormat{
		Type:   ai.ResponseFormatJSONSchema,
		Name:   spec.Schema.Name,
		Schema: append(json.RawMessage(nil), spec.Schema.JSON...),
	}

	workflow, err := a.NewRun(ctx, input)
	if err != nil {
		return zero, err
	}
	var accepted structured.Result[T]
	workflow.loop.OutputValidation = &loop.OutputValidation{
		MaxRetries: spec.Retry.MaxRetries,
		RetryPrompt: func(err error) string {
			return spec.Retry.Prompt(err)
		},
		Validate: func(raw json.RawMessage) error {
			result, err := spec.Decode(raw)
			if err == nil {
				accepted = result
			}
			return err
		},
	}
	if _, err := workflow.Run(ctx); err != nil {
		return zero, err
	}
	return accepted, nil
}
