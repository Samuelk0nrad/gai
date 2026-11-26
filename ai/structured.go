package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/Samuelk0nrad/gai/structured"
)

// GenerateStructured performs a provider-neutral model generation, validates
// the complete response, and returns a typed result. Invalid JSON is retried
// only when Spec.Retry explicitly allows it.
func GenerateStructured[T any](ctx context.Context, model Model, req AIRequest, spec structured.Spec[T]) (structured.Result[T], error) {
	var zero structured.Result[T]
	if model == nil {
		return zero, errors.New("structured generation model is nil")
	}
	if err := spec.ValidateSpec(); err != nil {
		return zero, err
	}
	base := req.Copy()
	base.ResponseFormat = ResponseFormat{
		Type:   ResponseFormatJSONSchema,
		Name:   spec.Schema.Name,
		Schema: append(json.RawMessage(nil), spec.Schema.JSON...),
	}
	if err := base.Validate(); err != nil {
		return zero, err
	}

	var correction string
	for retries := 0; ; retries++ {
		attempt := base.Copy()
		if correction != "" {
			attempt.Messages = append(attempt.Messages, TextMessage(RoleUser, correction))
		}
		response, err := generateResponse(ctx, model, attempt)
		if err != nil {
			return zero, err
		}
		result, err := spec.Decode(json.RawMessage(response.Text()))
		if err == nil {
			return result, nil
		}
		if !structuredOutputError(err) || retries >= spec.Retry.MaxRetries {
			return zero, err
		}
		correction = strings.TrimSpace(spec.Retry.Prompt(err))
		if correction == "" {
			return zero, fmt.Errorf("structured validation retry prompt is empty: %w", err)
		}
	}
}

func generateResponse(ctx context.Context, model Model, req AIRequest) (AIResponse, error) {
	var response AIResponse
	for token := range model.GenerateStream(ctx, req) {
		token = token.Clone()
		if err := token.Validate(); err != nil {
			return response, err
		}
		if token.Err != nil {
			return response, token.Err
		}
		response.AppendToken(token)
	}
	if err := ctx.Err(); err != nil {
		return response, err
	}
	return response, nil
}

func structuredOutputError(err error) bool {
	var validation *structured.ValidationError
	if errors.As(err, &validation) {
		return true
	}
	var decode *structured.DecodeError
	return errors.As(err, &decode)
}
