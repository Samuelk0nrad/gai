package gai

import (
	"bytes"
	"context"
	"encoding/json"
	"unicode/utf8"

	"github.com/Samuelk0nrad/gai/internal/appclock"
)

const (
	// DefaultContentCaptureMaxBytes is the finite limit used when a policy
	// enables content capture without specifying MaxBytes.
	DefaultContentCaptureMaxBytes = 16 * 1024
	// MaxContentCaptureBytes is the hard upper bound for captured content.
	MaxContentCaptureBytes = 1024 * 1024
)

// ContentKind identifies a category of model or agent content. Credentials,
// authorization values, request headers, and opaque provider state are
// deliberately not content categories and must never be captured.
type ContentKind string

const (
	ContentKindPrompt     ContentKind = "prompt"
	ContentKindCompletion ContentKind = "completion"
	ContentKindReasoning  ContentKind = "reasoning"
	ContentKindToolInput  ContentKind = "tool_input"
	ContentKindToolOutput ContentKind = "tool_output"
	ContentKindMemory     ContentKind = "memory"
)

// CaptureMode controls whether a content category may be observed.
type CaptureMode uint8

const (
	CaptureNone CaptureMode = iota
	CaptureEnabled
)

// ContentRedactor transforms content before it is bounded or observed.
// Implementations may be called concurrently and must be concurrency-safe.
type ContentRedactor func(ctx context.Context, kind ContentKind, value []byte) ([]byte, error)

// ContentCapturePolicy controls application-owned capture of sensitive model,
// agent, memory, and tool content. Its zero value captures nothing.
type ContentCapturePolicy struct {
	Prompt     CaptureMode
	Completion CaptureMode
	Reasoning  CaptureMode
	ToolInput  CaptureMode
	ToolOutput CaptureMode
	Memory     CaptureMode

	// MaxBytes bounds each captured value. Values <= 0 use
	// DefaultContentCaptureMaxBytes; values above MaxContentCaptureBytes are
	// clamped to that hard limit.
	MaxBytes int
	// Redact runs before truncation. An error or panic fails closed and omits
	// the content without changing application execution.
	Redact ContentRedactor
}

// CapturedContent is a redacted, bounded value safe to pass to an observability
// sink under the installed policy.
type CapturedContent struct {
	Value            []byte
	OriginalBytes    int
	CapturedBytes    int
	Truncated        bool
	RedactionApplied bool
}

type contentCapturePolicyContextKey struct{}

// WithContentCapturePolicy installs a request-scoped, vendor-neutral content
// capture policy. The policy remains local to ctx and is not propagated as
// OpenTelemetry baggage or outbound request metadata.
func WithContentCapturePolicy(ctx context.Context, policy ContentCapturePolicy) context.Context {
	return context.WithValue(ctx, contentCapturePolicyContextKey{}, policy)
}

// ContentCapturePolicyFromContext returns the explicitly installed policy.
// When no policy is installed, content capture defaults to disabled.
func ContentCapturePolicyFromContext(ctx context.Context) (ContentCapturePolicy, bool) {
	if ctx == nil {
		return ContentCapturePolicy{}, false
	}
	policy, ok := ctx.Value(contentCapturePolicyContextKey{}).(ContentCapturePolicy)
	return policy, ok
}

// CaptureContent applies the policy for kind. Redaction is always performed
// before deterministic truncation. Redactor failures and panics fail closed.
func CaptureContent(ctx context.Context, kind ContentKind, value []byte) (CapturedContent, bool) {
	policy, ok := ContentCapturePolicyFromContext(ctx)
	if !ok || policy.captureMode(kind) != CaptureEnabled {
		return CapturedContent{}, false
	}

	original := append([]byte(nil), value...)
	processed := append([]byte(nil), original...)
	redactionApplied := policy.Redact != nil
	if policy.Redact != nil {
		redacted, redactOK := safelyRedact(ctx, policy.Redact, kind, processed)
		if !redactOK {
			return CapturedContent{}, false
		}
		processed = append([]byte(nil), redacted...)
	}

	processed = bytes.ToValidUTF8(processed, []byte("\uFFFD"))
	limit := normalizedCaptureLimit(policy.MaxBytes)
	truncated := len(processed) > limit
	if truncated {
		processed = truncateCapturedContent(processed, limit)
	}
	processed = append([]byte(nil), processed...)

	return CapturedContent{
		Value:            processed,
		OriginalBytes:    len(original),
		CapturedBytes:    len(processed),
		Truncated:        truncated,
		RedactionApplied: redactionApplied,
	}, true
}

func (p ContentCapturePolicy) captureMode(kind ContentKind) CaptureMode {
	switch kind {
	case ContentKindPrompt:
		return p.Prompt
	case ContentKindCompletion:
		return p.Completion
	case ContentKindReasoning:
		return p.Reasoning
	case ContentKindToolInput:
		return p.ToolInput
	case ContentKindToolOutput:
		return p.ToolOutput
	case ContentKindMemory:
		return p.Memory
	default:
		return CaptureNone
	}
}

func normalizedCaptureLimit(limit int) int {
	if limit <= 0 {
		return DefaultContentCaptureMaxBytes
	}
	if limit > MaxContentCaptureBytes {
		return MaxContentCaptureBytes
	}
	return limit
}

func safelyRedact(ctx context.Context, redact ContentRedactor, kind ContentKind, value []byte) (redacted []byte, ok bool) {
	ok = false
	// Redactor time is application time, not provider latency.
	resume := appclock.Pause(ctx)
	defer resume()
	defer func() {
		if recover() != nil {
			redacted = nil
			ok = false
		}
	}()
	redacted, err := redact(ctx, kind, append([]byte(nil), value...))
	if err != nil {
		return nil, false
	}
	return redacted, true
}

func truncateCapturedContent(value []byte, limit int) []byte {
	if limit <= 0 {
		return nil
	}
	if json.Valid(value) && limit >= 2 {
		// A JSON string preview remains valid JSON even when the original value
		// was an object or array. Size metadata records that it is incomplete.
		preview := validUTF8Prefix(value, limit-2)
		for len(preview) > 0 {
			encoded, err := json.Marshal(string(preview))
			if err == nil && len(encoded) <= limit {
				return encoded
			}
			remove := len(encoded) - limit
			if remove < 1 {
				remove = 1
			}
			if remove >= len(preview) {
				preview = nil
				continue
			}
			preview = preview[:len(preview)-remove]
			for len(preview) > 0 && !utf8.Valid(preview) {
				preview = preview[:len(preview)-1]
			}
		}
		return []byte(`""`)
	}
	return validUTF8Prefix(value, limit)
}

func validUTF8Prefix(value []byte, limit int) []byte {
	if len(value) > limit {
		value = value[:limit]
	}
	for len(value) > 0 && !utf8.Valid(value) {
		value = value[:len(value)-1]
	}
	return append([]byte(nil), value...)
}
