package tooldefinitions

import (
	"context"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/internal/observe"
	"go.opentelemetry.io/otel/attribute"
)

const tracerName = "github.com/lace-ai/gai/context/tooldefinitions"

type observer struct {
	operation   *observe.Operation
	toolCount   int
	tokenBudget int
}

func newObserver(ctx context.Context, debug gai.ObservationSink, toolCount int, tokenBudget int) (context.Context, *observer) {
	ctx, operation := observe.Start(ctx, debug, tracerName, "context.tool_definitions", "context.operation", "build", "context/tooldefinitions:Source.Function",
		attribute.String("context.source", "tool_definitions"),
		attribute.Int("context.tool_count", toolCount),
		attribute.Int("context.token_budget", tokenBudget),
	)
	return ctx, &observer{operation: operation, toolCount: toolCount, tokenBudget: tokenBudget}
}

func (o *observer) Finish(err error) {
	if o == nil {
		return
	}
	o.operation.Finish(err)
}

func (o *observer) Started(ctx context.Context) {
	o.emit(ctx, "tool_definitions_build_started", o.fields(), nil)
}

func (o *observer) Succeeded(ctx context.Context, toolNames []string) {
	fields := o.fields()
	fields["tool_names"] = toolNames
	if o != nil {
		o.operation.Set(attribute.StringSlice("context.tool_names", toolNames))
	}
	o.emit(ctx, "tool_definitions_build_finished", fields, nil)
}

func (o *observer) Failed(ctx context.Context, stage string, err error) {
	fields := o.fields()
	fields["stage"] = stage
	o.emit(ctx, "tool_definitions_build_failed", fields, err)
}

func (o *observer) fields() map[string]any {
	fields := map[string]any{}
	if o == nil {
		return fields
	}
	fields["tool_count"] = o.toolCount
	fields["token_budget"] = o.tokenBudget
	return fields
}

func (o *observer) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if o == nil {
		return
	}
	o.operation.Emit(ctx, name, fields, err)
}
