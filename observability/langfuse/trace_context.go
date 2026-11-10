package langfuse

import (
	"context"
	"sort"
	"sync/atomic"

	"github.com/Samuelk0nrad/gai"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// NewTraceContextSpanProcessor returns a processor that maps gai.TraceContext
// fields to Langfuse v4 trace attributes before forwarding span lifecycle
// calls to next. When next is nil, it only applies the attribute mapping.
func NewTraceContextSpanProcessor(next sdktrace.SpanProcessor) sdktrace.SpanProcessor {
	return &traceContextSpanProcessor{next: next}
}

type traceContextSpanProcessor struct {
	next    sdktrace.SpanProcessor
	stopped atomic.Bool
}

func (p *traceContextSpanProcessor) OnStart(parent context.Context, span sdktrace.ReadWriteSpan) {
	if p == nil || p.stopped.Load() {
		return
	}
	if traceContext, ok := gai.TraceContextFromContext(parent); ok {
		span.SetAttributes(langfuseTraceContextAttributes(traceContext)...)
	}
	if p.next != nil {
		p.next.OnStart(parent, span)
	}
}

func (p *traceContextSpanProcessor) OnEnd(span sdktrace.ReadOnlySpan) {
	if p == nil || p.stopped.Load() || p.next == nil {
		return
	}
	p.next.OnEnd(span)
}

func (p *traceContextSpanProcessor) Shutdown(ctx context.Context) error {
	if p == nil || !p.stopped.CompareAndSwap(false, true) || p.next == nil {
		return nil
	}
	return p.next.Shutdown(ctx)
}

func (p *traceContextSpanProcessor) ForceFlush(ctx context.Context) error {
	if p == nil || p.stopped.Load() || p.next == nil {
		return nil
	}
	return p.next.ForceFlush(ctx)
}

func langfuseTraceContextAttributes(traceContext gai.TraceContext) []attribute.KeyValue {
	attributes := make([]attribute.KeyValue, 0, 6+len(traceContext.Metadata))
	attributes = appendLangfuseStringAttribute(attributes, "langfuse.trace.name", traceContext.Name)
	attributes = appendLangfuseStringAttribute(attributes, "langfuse.user.id", traceContext.UserID)
	attributes = appendLangfuseStringAttribute(attributes, "langfuse.session.id", traceContext.SessionID)
	if len(traceContext.Tags) > 0 {
		attributes = append(attributes, attribute.StringSlice("langfuse.trace.tags", traceContext.Tags))
	}
	attributes = appendLangfuseStringAttribute(attributes, "langfuse.release", traceContext.Release)
	attributes = appendLangfuseStringAttribute(attributes, "langfuse.environment", traceContext.Environment)
	keys := make([]string, 0, len(traceContext.Metadata))
	for key := range traceContext.Metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		attributes = append(attributes, attribute.String("langfuse.trace.metadata."+key, traceContext.Metadata[key]))
	}
	return attributes
}

func appendLangfuseStringAttribute(attributes []attribute.KeyValue, key, value string) []attribute.KeyValue {
	if value == "" {
		return attributes
	}
	return append(attributes, attribute.String(key, value))
}
