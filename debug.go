package gai

import (
	"context"
	"encoding/json"
	"reflect"
	"time"

	"github.com/Samuelk0nrad/gai/internal/appclock"
	"go.opentelemetry.io/otel/trace"
)

// Observation is a finalized, best-effort observability projection. It is not
// an ordered workflow execution event; applications needing replay or transport
// semantics should consume agent.Workflow.RunEvents instead.
type Observation struct {
	Name       string
	Source     string
	OccurredAt time.Time
	RunID      string
	TraceID    string
	SpanID     string
	Fields     map[string]any
	Err        error
}

// ObservationSink receives finalized observations synchronously. Emit may be
// invoked concurrently; implementations must be concurrency-safe, return
// promptly, and should enqueue internally when persistence is slow. GAI does
// not retry or wait for external persistence.
type ObservationSink interface {
	Emit(context.Context, Observation)
}

// ObservationSinkFunc adapts a function into an ObservationSink.
type ObservationSinkFunc func(context.Context, Observation)

func (f ObservationSinkFunc) Emit(ctx context.Context, observation Observation) {
	if f != nil {
		f(ctx, observation)
	}
}

// ObservationEnabled reports whether ctx has an observation destination.
func ObservationEnabled(ctx context.Context, sink ObservationSink) bool {
	return sink != nil || trace.SpanFromContext(ctx).IsRecording()
}

// ObservationContentEnabled reports whether a library-managed content field of
// kind would be captured under ctx's ContentCapturePolicy.
func ObservationContentEnabled(ctx context.Context, sink ObservationSink, kind ContentKind) bool {
	if !ObservationEnabled(ctx, sink) {
		return false
	}
	policy, ok := ContentCapturePolicyFromContext(ctx)
	return ok && policy.captureMode(kind) == CaptureEnabled
}

// AddObservationContent adds one library-managed content field after applying
// the request-scoped ContentCapturePolicy. With no installed policy, content
// capture is disabled.
func AddObservationContent(ctx context.Context, sink ObservationSink, fields map[string]any, field string, kind ContentKind, value any) {
	if fields == nil || field == "" {
		return
	}
	if !ObservationContentEnabled(ctx, sink, kind) {
		return
	}
	raw, ok := debugContentBytes(value)
	if !ok {
		return
	}
	captured, ok := CaptureContent(ctx, kind, raw)
	if !ok {
		return
	}
	fields[field] = string(captured.Value)
	fields[field+"_original_bytes"] = captured.OriginalBytes
	fields[field+"_captured_bytes"] = captured.CapturedBytes
	fields[field+"_truncated"] = captured.Truncated
	fields[field+"_redaction_applied"] = captured.RedactionApplied
	fields[field+"_content_kind"] = string(kind)
}

func debugContentBytes(value any) (raw []byte, ok bool) {
	ok = false
	defer func() {
		if recover() != nil {
			raw = nil
			ok = false
		}
	}()
	switch value := value.(type) {
	case string:
		return []byte(value), true
	case []byte:
		return append([]byte(nil), value...), true
	case json.RawMessage:
		return append([]byte(nil), value...), true
	default:
		encoded, err := json.Marshal(value)
		if err != nil {
			return nil, false
		}
		return encoded, true
	}
}

// EnrichObservation stamps time, run and trace correlation, and snapshots Fields
// so callers cannot mutate an emitted observation after the fact.
func EnrichObservation(ctx context.Context, observation Observation) Observation {
	if observation.OccurredAt.IsZero() {
		observation.OccurredAt = time.Now().UTC()
	}
	if observation.RunID == "" {
		observation.RunID, _ = ObservationRunIDFromContext(ctx)
	}
	if traceID, spanID, err := SpanContextIDs(ctx); err == nil {
		observation.TraceID = traceID
		observation.SpanID = spanID
	}
	fields := make(map[string]any, len(observation.Fields))
	for key, value := range observation.Fields {
		fields[key] = snapshotObservationField(value)
	}
	if observation.Err != nil {
		delete(fields, "error")
		fields["outcome"] = "error"
		fields["error_type"] = observationErrorType(observation.Err)
		// Errors may include provider responses, prompts, or tool output. Raw
		// errors remain available to control flow, but never leave it through a
		// finalized observation unless explicitly captured as managed content.
		observation.Err = nil
	}
	observation.Fields = fields
	return observation
}

// snapshotObservationField copies maps and slices recursively. These are the
// supported mutable structured field values; scalar values are immutable.
func snapshotObservationField(value any) any {
	if value == nil {
		return nil
	}
	return snapshotObservationReflectValue(reflect.ValueOf(value)).Interface()
}

func snapshotObservationReflectValue(value reflect.Value) reflect.Value {
	if !value.IsValid() {
		return value
	}

	switch value.Kind() {
	case reflect.Interface:
		if value.IsNil() {
			return value
		}
		copy := reflect.New(value.Type()).Elem()
		copy.Set(snapshotObservationReflectValue(value.Elem()))
		return copy
	case reflect.Map:
		if value.IsNil() {
			return value
		}
		copy := reflect.MakeMapWithSize(value.Type(), value.Len())
		iter := value.MapRange()
		for iter.Next() {
			copy.SetMapIndex(iter.Key(), snapshotObservationReflectValue(iter.Value()))
		}
		return copy
	case reflect.Slice:
		if value.IsNil() {
			return value
		}
		copy := reflect.MakeSlice(value.Type(), value.Len(), value.Len())
		for i := 0; i < value.Len(); i++ {
			copy.Index(i).Set(snapshotObservationReflectValue(value.Index(i)))
		}
		return copy
	default:
		return value
	}
}

func observationErrorType(err error) string {
	const maxErrorTypeBytes = 256
	typeName := reflect.TypeOf(err).String()
	if len(typeName) == 0 || len(typeName) > maxErrorTypeBytes {
		return "error"
	}
	return typeName
}

// EmitObservation finalizes and projects one semantic occurrence to OpenTelemetry
// and the optional application sink. Domain observers should use this helper
// rather than calling a sink directly.
func EmitObservation(ctx context.Context, sink ObservationSink, observation Observation) {
	observation = EnrichObservation(ctx, observation)
	RecordObservation(ctx, observation)
	if sink != nil {
		// Sink time is application time, not provider latency.
		resume := appclock.Pause(ctx)
		defer resume()
		sink.Emit(ctx, observation)
	}
}

type observationRunIDContextKey struct{}

// WithObservationRunID attaches workflow correlation to derived observations.
func WithObservationRunID(ctx context.Context, runID string) context.Context {
	return context.WithValue(ctx, observationRunIDContextKey{}, runID)
}

// ObservationRunIDFromContext returns the workflow correlation ID when present.
func ObservationRunIDFromContext(ctx context.Context) (string, bool) {
	if ctx == nil {
		return "", false
	}
	runID, ok := ctx.Value(observationRunIDContextKey{}).(string)
	return runID, ok && runID != ""
}
