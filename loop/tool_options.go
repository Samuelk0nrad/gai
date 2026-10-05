package loop

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/lace-ai/gai/ai"
)

// ToolExecutionConfig controls admission within one loop run. Zero preserves
// unlimited concurrency and inherits the caller's deadline without adding one.
type ToolExecutionConfig struct {
	// MaxConcurrent bounds active handler/processor pipelines; zero is unlimited.
	MaxConcurrent int
	// DefaultTimeout requests handler cancellation, excluding approval, queueing,
	// and result processing. Returned handler results are preserved; started work is joined.
	DefaultTimeout time.Duration
}

// Validate rejects negative concurrency and deadline settings.
func (c ToolExecutionConfig) Validate() error {
	if c.MaxConcurrent < 0 || c.DefaultTimeout < 0 {
		return fmt.Errorf("%w: concurrency and timeout must be non-negative", ErrToolExecutionConfig)
	}
	return nil
}

// ToolOptions controls one registration, independently of model-facing metadata.
// Serial includes result processing and preserves same-name order within a run.
// It does not provide exclusivity against different tools or other runs.
type ToolOptions struct {
	Traits ToolTraits
	Serial bool
	// Timeout nil inherits DefaultTimeout; a pointer to zero disables that default.
	// Parent context deadlines always apply. The pointed value is copied.
	Timeout *time.Duration
	// Guard is an explicit shared gate across registrations and concurrent runs.
	// A nil guard imposes no cross-run limit. Do not copy a ToolGuard after use.
	Guard *ToolGuard
}

// Validate rejects negative timeouts and unknown effect declarations.
func (o ToolOptions) Validate() error {
	if o.Timeout != nil && *o.Timeout < 0 {
		return fmt.Errorf("%w: tool timeout must be non-negative", ErrToolExecutionConfig)
	}
	switch o.Traits.Effect {
	case ToolEffectUnknown, ToolEffectReadOnly, ToolEffectMutating, ToolEffectDestructive:
	default:
		return fmt.Errorf("%w: unknown tool effect %q", ErrToolExecutionConfig, o.Traits.Effect)
	}
	return nil
}

// clone snapshots pointer-valued registration settings while sharing the guard.
func (o ToolOptions) clone() ToolOptions {
	if o.Timeout != nil {
		value := *o.Timeout
		o.Timeout = &value
	}
	return o
}

// ToolOptionsProvider supplies registration settings. Implementations must be
// concurrency-safe; the executor snapshots their returned settings.
type ToolOptionsProvider interface{ ToolOptions() ToolOptions }

type configuredTool struct {
	Tool
	options ToolOptions
}

// ToolOptions returns a copy of the registration settings.
func (t *configuredTool) ToolOptions() ToolOptions { return t.options.clone() }

// WithToolOptions wraps a tool with validated copied options. Nested wrappers
// replace, rather than merge, options. The handler and Guard remain shared.
func WithToolOptions(tool Tool, options ToolOptions) (Tool, error) {
	if nilImplementation(tool) {
		return nil, fmt.Errorf("%w: tool is nil", ai.ErrInvalidToolDefinition)
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	if _, err := ToolDefinitions([]Tool{tool}); err != nil {
		return nil, err
	}
	return &configuredTool{Tool: tool, options: options.clone()}, nil
}

// optionsForTool validates a captured registration or supplies default settings.
func optionsForTool(tool Tool) (ToolOptions, error) {
	if provider, ok := tool.(ToolOptionsProvider); ok {
		options := provider.ToolOptions().clone()
		return options, options.Validate()
	}
	return ToolOptions{}, nil
}

// ToolGuard serializes registrations sharing this exact guard, including across
// workflows. Its zero value is ready for use; it is an in-process gate only.
type ToolGuard struct {
	mu      sync.Mutex
	held    bool
	changed chan struct{}
}

// tryAcquire returns ownership or a generation channel to await without reserving a worker.
func (g *ToolGuard) tryAcquire() (release func(), changed <-chan struct{}) {
	if g == nil {
		return func() {}, nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.changed == nil {
		g.changed = make(chan struct{})
	}
	if g.held {
		return nil, g.changed
	}
	g.held = true
	return func() {
		g.mu.Lock()
		defer g.mu.Unlock()
		g.held = false
		close(g.changed)
		g.changed = make(chan struct{})
	}, nil
}

// configuredTool delegates invocation without acquiring Guard itself. Admission
// belongs to Loop; direct Function/CallTool invocations bypass these settings.
func (t *configuredTool) Function(ctx context.Context, call ai.ToolCall) (string, error) {
	return t.Tool.Function(ctx, call)
}
