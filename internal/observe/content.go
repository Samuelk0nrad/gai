package observe

import (
	"github.com/Samuelk0nrad/gai"
	"go.opentelemetry.io/otel/attribute"
)

// CapturedContentAttributes returns span attributes for content that already
// passed the request's ContentCapturePolicy. The captured value is written to
// every key in valueKeys, and its size, truncation, and redaction metadata is
// written under metaPrefix. Content is captured once and reused for every key.
func CapturedContentAttributes(metaPrefix string, captured gai.CapturedContent, valueKeys ...string) []attribute.KeyValue {
	value := string(captured.Value)
	attrs := make([]attribute.KeyValue, 0, len(valueKeys)+4)
	for _, key := range valueKeys {
		attrs = append(attrs, attribute.String(key, value))
	}
	return append(attrs,
		attribute.Int(metaPrefix+".original_bytes", captured.OriginalBytes),
		attribute.Int(metaPrefix+".captured_bytes", captured.CapturedBytes),
		attribute.Bool(metaPrefix+".truncated", captured.Truncated),
		attribute.Bool(metaPrefix+".redaction_applied", captured.RedactionApplied),
	)
}
