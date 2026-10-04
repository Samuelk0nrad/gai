package loop

import (
	"context"
	"errors"
	"fmt"
	"slices"
)

// ChainToolResultProcessors transforms output in declaration order. Each callback
// receives an independent call snapshot. Put redaction before the final byte limit.
func ChainToolResultProcessors(processors ...ToolResultProcessor) (ToolResultProcessor, error) {
	for _, p := range processors {
		if nilImplementation(p) {
			return nil, fmt.Errorf("%w: nil processor", ErrToolResultProcess)
		}
	}
	processors = slices.Clone(processors)
	return ToolResultProcessorFunc(func(ctx context.Context, input ToolPolicyInput, result ToolResult) (ToolResult, error) {
		for _, processor := range processors {
			if err := ctx.Err(); err != nil {
				return ToolResult{}, err
			}
			copied := input
			copied.Call = input.Call.Clone()
			next, err := processor.Process(ctx, copied, result)
			if err != nil {
				return ToolResult{}, err
			}
			result = normalizeToolResult(next.Text, next.Err)
		}
		return result, nil
	}), nil
}

// RejectToolResult withholds output using safe model-facing text. It does not
// undo any side effects or make an invocation safe to retry.
func RejectToolResult(reason string) ToolResult {
	if reason == "" {
		reason = ErrToolResultRejected.Error()
	}
	return ToolResult{Err: safeToolError{text: reason, kind: ErrToolResultRejected}}
}

type safeToolError struct {
	text string
	kind error
}

func (e safeToolError) Error() string { return e.text }
func (e safeToolError) Unwrap() error { return e.kind }

// RedactToolResult sanitizes successful output and error text. Error replacements
// do not wrap the original error, preventing raw error causes from escaping.
// A failing redactor terminates processing without publishing the original result.
func RedactToolResult(redact func(context.Context, string) (string, error)) (ToolResultProcessor, error) {
	if redact == nil {
		return nil, fmt.Errorf("%w: nil redactor", ErrToolResultProcess)
	}
	return ToolResultProcessorFunc(func(ctx context.Context, _ ToolPolicyInput, result ToolResult) (ToolResult, error) {
		text, err := redact(ctx, result.String())
		if err != nil {
			return ToolResult{}, err
		}
		if result.Err == nil {
			return ToolResult{Text: text}, nil
		}
		var kind error
		// Preserve only known safe classifications, never arbitrary original causes.
		for _, candidate := range []error{ErrToolOutputLimit, ErrToolResultRejected, ErrToolDenied, ErrToolApprovalRequired, context.DeadlineExceeded, context.Canceled} {
			if errors.Is(result.Err, candidate) {
				kind = candidate
				break
			}
		}
		return ToolResult{Err: safeToolError{text: text, kind: kind}}, nil
	}), nil
}

// LimitToolResultBytes rejects oversized model-facing text, including error text.
// It never truncates JSON or UTF-8. Zero disables the limit. This cannot bound
// allocations already made by a handler; only retained/published output is bounded.
func LimitToolResultBytes(maxBytes int) (ToolResultProcessor, error) {
	if maxBytes < 0 {
		return nil, fmt.Errorf("%w: negative result byte limit", ErrToolResultProcess)
	}
	return ToolResultProcessorFunc(func(_ context.Context, _ ToolPolicyInput, result ToolResult) (ToolResult, error) {
		if maxBytes > 0 && len(result.String()) > maxBytes {
			message := "tool output too large"
			if len(message) > maxBytes {
				message = message[:maxBytes]
			} // Fixed ASCII diagnostic, never truncated handler output.
			return ToolResult{Err: safeToolError{text: message, kind: ErrToolOutputLimit}}, nil
		}
		return result, nil
	}), nil
}
