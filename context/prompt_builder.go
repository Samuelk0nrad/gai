package context

import (
	"context"
	"fmt"
	"strings"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

const contextTracerName = "github.com/lace-ai/gai/context"

// ContextSource produces one prompt part using the remaining context budget.
// Sources are evaluated in declaration order by Builder.BuildContext.
type ContextSource interface {
	Name() string
	Function(ctx context.Context, TokenBudget int) (Part, error)
}

// TokenCounterSetter is implemented by components that accept counter injection.
type TokenCounterSetter interface {
	SetTokenCounter(counter ai.TokenCounter)
}

// ContextSourceWithTokenCount optionally returns the token total calculated
// while selecting a part, avoiding a second count during the same build.
// Builder injects its counter before calling FunctionWithTokens. When a part is
// returned, its total must be non-negative and equal to counting that snapshot
// with the injected counter. Counts accompanying a nil part are ignored.
// It belongs only to this invocation, not to the part or persisted state.
// Builders use Function when budgeting is disabled or no counter is available.
type ContextSourceWithTokenCount interface {
	ContextSource
	TokenCounterSetter
	FunctionWithTokens(ctx context.Context, tokenBudget int) (Part, int, error)
}

// PromptBuilder is the prompt-construction contract consumed by agent loops.
// Input exposes the user content recorded in the loop's conversation. Builder
// configuration methods remain available on the concrete Builder; consumers
// that configure a custom builder may require additional capabilities.
type PromptBuilder interface {
	BuildContext(ctx context.Context) ([]Part, error)
	BuildRequest(ctx context.Context, conv Conversation) (ai.AIRequest, error)
	Input() PromptInput
}

// TokenBudget exposes prompt-window configuration and remaining capacity.
type TokenBudget interface {
	SetTokenLimit(limit int) error
	SetOutputTokenReserve(reserve int) error
	GetRemainingTokens() (int, error)
	TokenCounter() ai.TokenCounter
}

// Definition configures a Builder.
type Definition struct {
	// Renderer lowers arbitrary system/context parts to canonical text.
	// Structured conversation parts bypass it. Nil selects XMLRenderer.
	Renderer Renderer
	// SystemInstructions are placed before context, user, and conversation parts.
	SystemInstructions []Part
	// ContextSources dynamically produce context during BuildContext.
	ContextSources []ContextSource
	// PromptInput contains run-specific user content and structured context.
	PromptInput PromptInput
	// TokenBudget is the total prompt window. Non-positive values disable budgeting.
	TokenBudget int
	// OutputTokenReserve is withheld from the prompt budget for model output.
	OutputTokenReserve int
	// TokenCounter counts parts locally and is propagated to compatible context
	// sources. Nil selects ai.TextTokenEstimator.
	TokenCounter ai.TokenCounter
	// ObservationSink receives prompt-building diagnostics.
	ObservationSink gai.ObservationSink
}

// Builder assembles system instructions, dynamic context, user input, and loop
// messages into one canonical model request.
type Builder struct {
	SystemInstructions []Part
	ContextSources     []ContextSource
	ContextParts       []Part
	Iteration          []Part
	TokenBudget        int
	Renderer           Renderer
	defaultRenderer    *XMLRenderer
	debugSink          gai.ObservationSink
	input              PromptInput
	counter            ai.TokenCounter
	OutputTokenReserve int
}

// New creates a prompt builder from def. A nil TokenCounter selects the
// generic local estimator. Agent may replace it with an effective model counter.
func New(def Definition) *Builder {
	counter := def.TokenCounter
	if counter == nil {
		counter = ai.TextTokenEstimator{}
	}
	renderer := def.Renderer
	var defaultRenderer *XMLRenderer
	if renderer == nil {
		defaultRenderer = &XMLRenderer{
			ObservationSink:         def.ObservationSink,
			ObservationPreviewChars: 100,
		}
		renderer = defaultRenderer
	}
	return &Builder{
		SystemInstructions: append([]Part{}, def.SystemInstructions...),
		ContextSources:     append([]ContextSource{}, def.ContextSources...),
		ContextParts:       []Part{},
		Iteration:          []Part{},
		TokenBudget:        def.TokenBudget,
		OutputTokenReserve: def.OutputTokenReserve,
		Renderer:           renderer,
		defaultRenderer:    defaultRenderer,
		debugSink:          def.ObservationSink,
		input:              def.PromptInput.Clone(),
		counter:            counter,
	}
}

// NewBuilder creates a builder with a renderer and total token budget.
func NewBuilder(renderer Renderer, tokenBudget int) *Builder {
	return New(Definition{
		Renderer:    renderer,
		TokenBudget: tokenBudget,
	})
}

func (b *Builder) SetObservationSink(debugSink gai.ObservationSink) {
	b.debugSink = debugSink
	if b.defaultRenderer != nil {
		b.defaultRenderer.ObservationSink = debugSink
	}
}

func (b *Builder) AppendContextSource(ctx context.Context, source ContextSource) error {
	if setter, ok := source.(TokenCounterSetter); ok && b.counter != nil {
		setter.SetTokenCounter(b.counter)
	}
	b.ContextSources = append(b.ContextSources, source)
	return nil
}

// PrependContextSource adds source before all existing context sources.
func (b *Builder) PrependContextSource(ctx context.Context, source ContextSource) error {
	if setter, ok := source.(TokenCounterSetter); ok && b.counter != nil {
		setter.SetTokenCounter(b.counter)
	}
	b.ContextSources = append([]ContextSource{source}, b.ContextSources...)
	return nil
}

// ReplaceContextSource replaces all sources named name with source, preserving
// the position of the first matching source.
func (b *Builder) ReplaceContextSource(ctx context.Context, name string, source ContextSource) error {
	name = strings.TrimSpace(name)
	if b == nil {
		return ErrPromptBuilderNil
	}
	if name == "" || source == nil {
		return ErrPromptSource
	}
	if setter, ok := source.(TokenCounterSetter); ok && b.counter != nil {
		setter.SetTokenCounter(b.counter)
	}

	sources := make([]ContextSource, 0, len(b.ContextSources))
	replaced := false
	for _, existing := range b.ContextSources {
		if existing != nil && existing.Name() == name {
			if !replaced {
				sources = append(sources, source)
				replaced = true
			}
			continue
		}
		sources = append(sources, existing)
	}
	if !replaced {
		return ErrPromptSource
	}
	b.ContextSources = sources
	return nil
}

// RemoveContextSource removes all sources named name.
func (b *Builder) RemoveContextSource(ctx context.Context, name string) error {
	name = strings.TrimSpace(name)
	if b == nil {
		return ErrPromptBuilderNil
	}
	if name == "" {
		return ErrPromptSource
	}

	sources := make([]ContextSource, 0, len(b.ContextSources))
	removed := false
	for _, existing := range b.ContextSources {
		if existing != nil && existing.Name() == name {
			removed = true
			continue
		}
		sources = append(sources, existing)
	}
	if !removed {
		return ErrPromptSource
	}
	b.ContextSources = sources
	return nil
}

func (b *Builder) AppendContextSources(ctx context.Context, sources ...ContextSource) error {
	for _, source := range sources {
		if err := b.AppendContextSource(ctx, source); err != nil {
			return err
		}
	}
	return nil
}

// HasContextSource reports whether a source with name is already configured.
// It allows higher-level components to provide default sources without
// duplicating application-provided prompt context.
func (b *Builder) HasContextSource(name string) bool {
	name = strings.TrimSpace(name)
	if b == nil || name == "" {
		return false
	}
	for _, source := range b.ContextSources {
		if source != nil && source.Name() == name {
			return true
		}
	}
	return false
}

func (b *Builder) AppendSystemInstructions(ctx context.Context, instructions ...Part) error {
	b.SystemInstructions = append(b.SystemInstructions, instructions...)
	return nil
}

func (b *Builder) BuildContext(ctx context.Context) (contextParts []Part, err error) {
	ctx, obs := newPromptBuilderContextObserver(ctx, b)
	stats := promptContextBuildStats{
		SourceCount:            len(b.ContextSources),
		SystemInstructionCount: len(b.SystemInstructions),
		TokenBudget:            b.TokenBudget,
		OutputTokenReserve:     b.OutputTokenReserve,
		TokenCounterPresent:    b.counter != nil,
	}
	defer func() {
		stats.ContextPartCount = len(contextParts)
		obs.FinishContext(err, stats)
	}()
	if err = ctx.Err(); err != nil {
		return nil, err
	}

	if b.TokenBudget > 0 {
		stats.SystemTokens, err = b.SystemInstructionsTokens(ctx)
		if err != nil {
			return nil, err
		}
		stats.RemainingTokens = b.TokenBudget - b.OutputTokenReserve - stats.SystemTokens
	} else {
		obs.TokenBudgetSkipped(ctx)
		stats.RemainingTokens = 1000000 // effectively unlimited
	}
	obs.BuildStarted(ctx, stats)

	for _, source := range b.ContextSources {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		if source == nil {
			obs.SourceSkipped(ctx, "<nil>")
			continue
		}
		if setter, ok := source.(TokenCounterSetter); ok && b.counter != nil {
			setter.SetTokenCounter(b.counter)
		}
		var part Part
		var sourceTokens int
		countedSource, counted := source.(ContextSourceWithTokenCount)
		counted = counted && b.TokenBudget > 0 && b.counter != nil
		if counted {
			part, sourceTokens, err = countedSource.FunctionWithTokens(ctx, stats.RemainingTokens)
		} else {
			part, err = source.Function(ctx, stats.RemainingTokens)
		}
		if err != nil {
			obs.SourceFailed(ctx, source.Name(), stats.RemainingTokens, err)
			return nil, err
		}
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		if part != nil {
			contextParts = append(contextParts, part)
			stats.IncludedSourceCount++
			var tokenStats promptPartTokenStats
			if b.TokenBudget > 0 {
				fields := map[string]any{
					"source": source.Name(),
					"part":   part.Name(),
				}
				var tokens int
				if counted {
					tokens, err = b.validateTokenCount(ctx, sourceTokens, fields)
				} else {
					tokens, err = b.partTokens(ctx, part, fields)
				}
				if err != nil {
					return nil, err
				}
				tokenStats = promptPartTokenStats{Tokens: tokens, TokensCounted: b.counter != nil}
				if tokenStats.TokensCounted {
					stats.RemainingTokens -= tokens
				}
			}
			obs.SourceIncluded(ctx, source.Name(), part.Name(), tokenStats, stats.RemainingTokens)
		}
	}
	for _, part := range b.input.Context {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		if part == nil {
			continue
		}
		contextParts = append(contextParts, part)
		if b.TokenBudget <= 0 {
			continue
		}
		tokens, err := b.partTokens(ctx, part, map[string]any{
			"source": "prompt_input",
			"part":   part.Name(),
		})
		if err != nil {
			return nil, err
		}
		if b.counter != nil {
			stats.RemainingTokens -= tokens
		}
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	b.ContextParts = contextParts
	obs.BuildFinished(ctx, stats)
	return contextParts, nil
}

// BuildRequest assembles a single canonical conversation. Renderers lower only
// arbitrary context parts; structured history and user content retain their
// roles, ordered parts, call identifiers, and opaque provider metadata.
func (b *Builder) BuildRequest(ctx context.Context, conv Conversation) (ai.AIRequest, error) {
	if err := ctx.Err(); err != nil {
		return ai.AIRequest{}, err
	}
	var messages []ai.Message
	if len(b.SystemInstructions) > 0 {
		text, err := b.Renderer.Render(ctx, []Part{NewSystemPart(b.SystemInstructions)})
		if err != nil {
			return ai.AIRequest{}, err
		}
		messages = append(messages, ai.TextMessage(ai.RoleSystem, text))
	}
	for _, part := range b.ContextParts {
		if err := ctx.Err(); err != nil {
			return ai.AIRequest{}, err
		}
		if part == nil {
			continue
		}
		if canonical, ok := part.(ConversationPart); ok {
			messages = append(messages, ai.CloneMessages(canonical.ConversationMessages())...)
			continue
		}
		text, err := b.Renderer.Render(ctx, []Part{part})
		if err != nil {
			return ai.AIRequest{}, err
		}
		messages = append(messages, ai.TextMessage(ai.RoleUser, text))
	}
	if len(b.input.User) > 0 {
		messages = append(messages, ai.Message{Role: ai.RoleUser, Parts: ai.CloneParts(b.input.User)})
	}
	if conv != nil {
		messages = append(messages, ai.CloneMessages(conv.Messages())...)
	}
	if err := ctx.Err(); err != nil {
		return ai.AIRequest{}, err
	}
	request := ai.AIRequest{Messages: messages}
	if err := request.Validate(); err != nil {
		return ai.AIRequest{}, err
	}
	return request, nil
}

func (b *Builder) Input() PromptInput {
	return b.input.Clone()
}

func (b *Builder) SetTokenLimit(limit int) error {
	if limit < 0 {
		return fmt.Errorf("%w: %d", ErrInvalideTokenLimit, limit)
	}
	b.TokenBudget = limit
	return nil
}

func (b *Builder) SetInput(input PromptInput) {
	b.input = input.Clone()
}

func (b *Builder) TokenCounter() ai.TokenCounter {
	return b.counter
}

// SetTokenCounter replaces the local counter; nil selects the generic estimator.
func (b *Builder) SetTokenCounter(counter ai.TokenCounter) {
	if counter == nil {
		counter = ai.TextTokenEstimator{}
	}
	b.counter = counter
}

func (b *Builder) SetOutputTokenReserve(reserve int) error {
	if reserve < 0 {
		return fmt.Errorf("%w: %d", ErrInvalidOutputReserve, reserve)
	}
	b.OutputTokenReserve = reserve
	return nil
}

// SystemInstructionsTokens counts system parts without hiding counter errors.
func (b *Builder) SystemInstructionsTokens(ctx context.Context) (int, error) {
	count := 0
	if b.counter == nil {
		if len(b.SystemInstructions) > 0 {
			newPromptBuilderDebugObserver(b).TokenCountSkipped(ctx, map[string]any{
				"reason": "counter_missing",
				"scope":  "system_instructions",
				"parts":  len(b.SystemInstructions),
			})
		}
		return count, nil
	}
	for _, part := range b.SystemInstructions {
		tokens, err := b.partTokens(ctx, part, map[string]any{
			"scope": "system_instructions",
			"part":  part.Name(),
		})
		if err != nil {
			return 0, err
		}
		count += tokens
	}
	return count, nil
}

func (b *Builder) partTokens(ctx context.Context, part Part, fields map[string]any) (int, error) {
	obs := newPromptBuilderDebugObserver(b)
	if b.counter == nil {
		obs.TokenCountSkipped(ctx, mergeDebugFields(fields, map[string]any{
			"reason": "counter_missing",
		}))
		return 0, nil
	}
	tokens, err := part.Tokens(ctx, b.counter)
	if err != nil {
		obs.TokenCountFailed(ctx, fields, err)
		return 0, err
	}
	return b.validateTokenCount(ctx, tokens, fields)
}

// validateTokenCount applies the same validation and diagnostics to counts
// calculated by a part and build-local totals supplied by a context source.
func (b *Builder) validateTokenCount(ctx context.Context, tokens int, fields map[string]any) (int, error) {
	if tokens < 0 {
		err := fmt.Errorf("%w: %d", ErrInvalidTokenCount, tokens)
		newPromptBuilderDebugObserver(b).TokenCountFailed(ctx, fields, err)
		return 0, err
	}
	return tokens, nil
}

func mergeDebugFields(base map[string]any, extra map[string]any) map[string]any {
	merged := make(map[string]any, len(base)+len(extra))
	for key, value := range base {
		merged[key] = value
	}
	for key, value := range extra {
		merged[key] = value
	}
	return merged
}
