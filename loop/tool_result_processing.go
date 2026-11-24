package loop

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"unicode/utf8"
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
			result = next.normalized()
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
			return ToolResult{Text: text, Media: result.Media}, nil
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
			return rejectOversizedToolResult(result), nil
		}
		return result, nil
	}), nil
}

// MinTruncateToolResultBytes is the smallest supported nonzero budget for
// TruncateToolResultBytes. It fits the longest truncation note with room left
// for the start of the result.
const MinTruncateToolResultBytes = 128

// TruncateToolResultBytes cuts oversized successful output instead of
// rejecting it. It keeps the start of the text, cut on a UTF-8 boundary, and
// appends a note with the kept and original sizes:
//
//	\n\n[tool result truncated: showing first K of N bytes]
//
// The note counts toward maxBytes, so the result never exceeds it. The cut is
// plain text: a cut JSON result is no longer valid JSON. Put a structure-aware
// processor before this one to shorten JSON by fields instead. The same input
// always produces the same output, and output that already fits is unchanged,
// so applying the processor again to stored results with the same or a larger
// limit is a no-op. A result that already ends with a truncation note is cut
// again from its kept text and keeps the original size in the new note, so a
// later, smaller limit gives the same text as cutting the original once.
//
// Oversized error text, including ErrToolDenied and ErrToolApprovalRequired
// refusals, is handled exactly like LimitToolResultBytes: it is replaced with a
// safe ErrToolOutputLimit or refusal message rather than cut. Zero disables the
// limit; positive budgets below MinTruncateToolResultBytes are invalid. Put it
// last in ChainToolResultProcessors, after redaction. Like LimitToolResultBytes,
// it cannot bound allocations already made by a handler.
func TruncateToolResultBytes(maxBytes int) (ToolResultProcessor, error) {
	if maxBytes < 0 || (maxBytes > 0 && maxBytes < MinTruncateToolResultBytes) {
		return nil, fmt.Errorf("%w: result truncation limit must be zero or at least %d", ErrToolResultProcess, MinTruncateToolResultBytes)
	}
	return ToolResultProcessorFunc(func(_ context.Context, _ ToolPolicyInput, result ToolResult) (ToolResult, error) {
		if maxBytes == 0 || len(result.String()) <= maxBytes {
			return result, nil
		}
		if result.Err != nil {
			return rejectOversizedToolResult(result), nil
		}
		return ToolResult{Text: truncateToolText(result.Text, maxBytes), Media: result.Media}, nil
	}), nil
}

// truncateToolText keeps the longest UTF-8-aligned prefix that fits maxBytes
// together with its note. len(text) must exceed maxBytes.
func truncateToolText(text string, maxBytes int) string {
	original := len(text)
	if kept, size, ok := parseTruncatedToolText(text); ok {
		text, original = kept, size
	}
	// The kept size never exceeds maxBytes, so a note sized with maxBytes is at
	// least as long as the final note; cutting to fit it always leaves room.
	keep := alignToolTextCut(text, maxBytes-len(truncationNote(maxBytes, original)))
	// The final note prints keep, which can have fewer digits than maxBytes;
	// grow the prefix into the bytes that frees while it still fits.
	for {
		next := alignToolTextCut(text, maxBytes-len(truncationNote(keep, original)))
		if next <= keep || next+len(truncationNote(next, original)) > maxBytes {
			break
		}
		keep = next
	}
	return text[:keep] + truncationNote(keep, original)
}

// alignToolTextCut moves a cut at or before len(text) back to a UTF-8 boundary.
// Invalid UTF-8 backs off at most one partial rune.
func alignToolTextCut(text string, keep int) int {
	keep = min(max(keep, 0), len(text))
	for i := 0; i < utf8.UTFMax-1 && keep > 0 && keep < len(text) && !utf8.RuneStart(text[keep]); i++ {
		keep--
	}
	return keep
}

func truncationNote(kept, original int) string {
	return fmt.Sprintf("\n\n[tool result truncated: showing first %d of %d bytes]", kept, original)
}

// truncationNotePattern matches a note produced by truncationNote at the end of text.
var truncationNotePattern = regexp.MustCompile(`\n\n\[tool result truncated: showing first ([0-9]+) of ([0-9]+) bytes\]\z`)

// parseTruncatedToolText recognizes output of an earlier truncation: a note
// whose kept size matches the text before it and whose original size is larger.
func parseTruncatedToolText(text string) (kept string, original int, ok bool) {
	match := truncationNotePattern.FindStringSubmatchIndex(text)
	if match == nil {
		return "", 0, false
	}
	keep, err1 := strconv.Atoi(text[match[2]:match[3]])
	size, err2 := strconv.Atoi(text[match[4]:match[5]])
	if err1 != nil || err2 != nil || keep != match[0] || size <= keep {
		return "", 0, false
	}
	return text[:keep], size, true
}

// rejectOversizedToolResult replaces output with a complete safe diagnostic,
// preserving framework refusal classifications.
func rejectOversizedToolResult(result ToolResult) ToolResult {
	message := "tool output too large"
	kinds := []error{ErrToolOutputLimit}
	for _, refusal := range []error{ErrToolDenied, ErrToolApprovalRequired} {
		if errors.Is(result.Err, refusal) {
			kinds = append(kinds, refusal)
			message = refusal.Error()
		}
	}
	return ToolResult{Err: safeToolError{text: message, kind: errors.Join(kinds...)}}
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
