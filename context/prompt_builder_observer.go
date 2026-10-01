package context

import (
	"context"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/internal/observe"
	"go.opentelemetry.io/otel/attribute"
)

type promptContextBuildStats struct {
	SourceCount            int
	SystemInstructionCount int
	SystemTokens           int
	TokenBudget            int
	OutputTokenReserve     int
	RemainingTokens        int
	ContextPartCount       int
	IncludedSourceCount    int
	TokenCounterPresent    bool
}

type promptPartTokenStats struct {
	Tokens        int
	TokensCounted bool
}

type promptBuilderObserver struct {
	debug     gai.ObservationSink
	operation *observe.Operation
}

func newPromptBuilderDebugObserver(b *Builder) *promptBuilderObserver {
	if b == nil {
		return &promptBuilderObserver{}
	}
	return &promptBuilderObserver{debug: b.debugSink, operation: observe.New(b.debugSink, "context:Builder")}
}

func newPromptBuilderContextObserver(ctx context.Context, b *Builder) (context.Context, *promptBuilderObserver) {
	ctx, operation := observe.Start(ctx, b.debugSink, contextTracerName, "context.prompt_builder", "context.operation", "build_context", "context:Builder",
		attribute.Int("context.source_count", len(b.ContextSources)),
		attribute.Int("context.system_instruction_count", len(b.SystemInstructions)),
		attribute.Int("context.token_budget", b.TokenBudget),
		attribute.Int("context.output_token_reserve", b.OutputTokenReserve),
		attribute.Bool("context.counter_present", b.counter != nil),
	)
	return ctx, &promptBuilderObserver{debug: b.debugSink, operation: operation}
}

func (o *promptBuilderObserver) FinishContext(err error, stats promptContextBuildStats) {
	if o == nil {
		return
	}
	o.operation.Set(
		attribute.Int("context.system_tokens", stats.SystemTokens),
		attribute.Int("context.remaining_tokens", stats.RemainingTokens),
		attribute.Int("context.context_parts", stats.ContextPartCount),
		attribute.Int("context.included_source_count", stats.IncludedSourceCount),
	)
	o.operation.Finish(err)
}

func (o *promptBuilderObserver) TokenBudgetSkipped(ctx context.Context) {
	o.emit(ctx, "prompt_builder_token_budget_skipped", map[string]any{
		"reason": "token_budget_not_set",
	}, nil)
}

func (o *promptBuilderObserver) BuildStarted(ctx context.Context, stats promptContextBuildStats) {
	if o == nil {
		return
	}
	if o.operation != nil {
		o.operation.Set(
			attribute.Int("context.system_tokens", stats.SystemTokens),
			attribute.Int("context.remaining_tokens", stats.RemainingTokens),
		)
	}
	o.emit(ctx, "prompt_builder_context_build_started", map[string]any{
		"source_count":             stats.SourceCount,
		"system_instruction_count": stats.SystemInstructionCount,
		"system_tokens":            stats.SystemTokens,
		"token_budget":             stats.TokenBudget,
		"output_token_reserve":     stats.OutputTokenReserve,
		"remaining_tokens":         stats.RemainingTokens,
		"counter_present":          stats.TokenCounterPresent,
	}, nil)
}

func (o *promptBuilderObserver) SourceFailed(ctx context.Context, source string, remainingTokens int, err error) {
	if o == nil {
		return
	}
	if o.operation != nil {
		o.operation.Set(attribute.String("context.failed_source", source))
	}
	o.emit(ctx, "prompt_builder_source_failed", map[string]any{
		"source":           source,
		"remaining_tokens": remainingTokens,
	}, err)
}

func (o *promptBuilderObserver) SourceSkipped(ctx context.Context, source string) {
	o.emit(ctx, "prompt_builder_source_skipped", map[string]any{
		"source": source,
	}, nil)
}

func (o *promptBuilderObserver) SourceIncluded(ctx context.Context, source string, part string, stats promptPartTokenStats, remainingTokens int) {
	o.emit(ctx, "prompt_builder_source_included", map[string]any{
		"source":           source,
		"part":             part,
		"tokens":           stats.Tokens,
		"tokens_counted":   stats.TokensCounted,
		"remaining_tokens": remainingTokens,
	}, nil)
}

func (o *promptBuilderObserver) BuildFinished(ctx context.Context, stats promptContextBuildStats) {
	o.emit(ctx, "prompt_builder_context_build_finished", map[string]any{
		"source_count":     stats.SourceCount,
		"context_parts":    stats.ContextPartCount,
		"remaining_tokens": stats.RemainingTokens,
	}, nil)
}

func (o *promptBuilderObserver) TokenCountSkipped(ctx context.Context, fields map[string]any) {
	o.emit(ctx, "prompt_builder_token_count_skipped", fields, nil)
}

func (o *promptBuilderObserver) TokenCountFailed(ctx context.Context, fields map[string]any, err error) {
	o.emit(ctx, "prompt_builder_token_count_failed", fields, err)
}

func (o *promptBuilderObserver) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if o == nil {
		return
	}
	o.operation.Emit(ctx, name, fields, err)
}
