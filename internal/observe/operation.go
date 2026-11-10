// Package observe provides repository-private primitives for shared observer plumbing.
package observe

import (
	"context"

	"github.com/Samuelk0nrad/gai"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// Operation owns the OpenTelemetry span and observation plumbing for one
// domain-specific observation. Domain observers retain event names, fields,
// attributes, errors, and content-capture decisions.
type Operation struct {
	sink   gai.ObservationSink
	span   trace.Span
	source string
}

// New returns an operation that emits domain-specific observations without
// owning a span.
func New(sink gai.ObservationSink, source string) *Operation {
	return &Operation{sink: sink, source: source}
}

// Start starts an operation span and returns an operation that can emit domain
// events and complete the span. This helper is internal so callers do not
// depend on a generic public observability abstraction.
func Start(ctx context.Context, sink gai.ObservationSink, tracerName, spanPrefix, operationAttr, operation, source string, attrs ...attribute.KeyValue) (context.Context, *Operation) {
	ctx, span := gai.StartOperationSpan(ctx, tracerName, spanPrefix, operationAttr, operation, attrs...)
	return ctx, &Operation{sink: sink, span: span, source: source}
}

// Set adds attributes to the operation span.
func (o *Operation) Set(attrs ...attribute.KeyValue) {
	if o == nil || o.span == nil {
		return
	}
	o.span.SetAttributes(attrs...)
}

// Emit finalizes and projects one domain-specific observation.
func (o *Operation) Emit(ctx context.Context, name string, fields map[string]any, err error) {
	if o == nil {
		return
	}
	gai.EmitObservation(ctx, o.sink, gai.Observation{Name: name, Source: o.source, Fields: fields, Err: err})
}

// Finish records err on the span, when present, and ends it.
func (o *Operation) Finish(err error) {
	if o == nil || o.span == nil {
		return
	}
	gai.EndSpan(o.span, err)
}
