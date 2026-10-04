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
		return ToolResult{Err: safeToolError{text: text, kind: errors.Join(safeToolErrorKinds(result.Err)...)}}, nil
	}), nil
}

// MinToolResultBytes is the smallest supported nonzero result budget. It fits
// every complete framework diagnostic, including an approval-required refusal.
const MinToolResultBytes = len("tool approval required")

// LimitToolResultBytes rejects oversized model-facing text, including error text.
// It never truncates JSON or diagnostics. Zero disables the limit; smaller
// positive budgets than MinToolResultBytes are invalid. This cannot bound
// allocations already made by a handler; only retained/published output is bounded.
func LimitToolResultBytes(maxBytes int) (ToolResultProcessor, error) {
	if maxBytes < 0 || (maxBytes > 0 && maxBytes < MinToolResultBytes) {
		return nil, fmt.Errorf("%w: result byte limit must be zero or at least %d", ErrToolResultProcess, MinToolResultBytes)
	}
	return ToolResultProcessorFunc(func(_ context.Context, _ ToolPolicyInput, result ToolResult) (ToolResult, error) {
		if maxBytes > 0 && len(result.String()) > maxBytes {
			message := "tool output too large"
			kinds := []error{ErrToolOutputLimit}
			for _, refusal := range []error{ErrToolDenied, ErrToolApprovalRequired} {
				if errors.Is(result.Err, refusal) {
					kinds = append(kinds, refusal)
					message = refusal.Error()
				}
			}
			return ToolResult{Err: safeToolError{text: message, kind: errors.Join(kinds...)}}, nil
		}
		return result, nil
	}), nil
}

// safeToolErrorKinds retains framework classifications without arbitrary error causes.
func safeToolErrorKinds(err error) []error {
	var kinds []error
	for _, candidate := range []error{ErrToolOutputLimit, ErrToolResultRejected, ErrToolDenied, ErrToolApprovalRequired, context.DeadlineExceeded, context.Canceled} {
		if errors.Is(err, candidate) {
			kinds = append(kinds, candidate)
		}
	}
	return kinds
}

// preserveToolRefusal keeps authorization failure independent of output filtering.
func preserveToolRefusal(original, processed ToolResult) ToolResult {
	for _, refusal := range []error{ErrToolDenied, ErrToolApprovalRequired} {
		if errors.Is(original.Err, refusal) && !errors.Is(processed.Err, refusal) {
			kinds := append(safeToolErrorKinds(processed.Err), refusal)
			processed = ToolResult{Err: safeToolError{text: processed.String(), kind: errors.Join(kinds...)}}
		}
	}
	return processed
}
