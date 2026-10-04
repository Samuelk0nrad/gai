package agent

import (
	"errors"
	"fmt"
	"reflect"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/context/tooldefinitions"
	"github.com/lace-ai/gai/loop"
)

var (
	// ErrInvalidExecutionConfig identifies an invalid effective run configuration.
	ErrInvalidExecutionConfig = errors.New("invalid agent execution configuration")
	// ErrTokenCounterNotConfigurable means a prompt builder cannot honor an explicit
	// counter selection or a model override's automatic counter selection.
	ErrTokenCounterNotConfigurable = errors.New("prompt builder does not implement TokenCounterSetter")
)

// Optional distinguishes inheritance from replacement with a possibly nil value.
// When Set is false, Value is ignored. When Set is true, Value replaces the
// definition setting, including nil. Only overrides need this presence marker.
type Optional[T any] struct {
	Set   bool
	Value T
}

// LimitsOverrides replaces individual limits without resetting omitted fields.
type LimitsOverrides struct {
	// MaxLoopIterations inherits when nil; zero selects the loop's default.
	MaxLoopIterations *int
	// MaxTokens inherits when nil; zero selects adapter/provider defaults.
	// This limits each generation, not the total workflow's usage.
	MaxTokens *int
}

// ExecutionOverrides changes the definition settings for one workflow.
// Nil pointers inherit; supplied objects replace atomically, except Limits,
// whose fields inherit independently. Configuration is copied by NewRun.
// Dependency implementations remain shared and must support concurrent use.
type ExecutionOverrides struct {
	// Model inherits when nil. A model is required after resolution.
	Model  ai.Model
	Limits LimitsOverrides
	// Tools inherits when nil; a non-nil slice replaces membership and order.
	// A non-nil empty slice disables all tools.
	Tools          []loop.Tool
	ToolChoice     *ai.ToolChoice
	ResponseFormat *ai.ResponseFormat
	Reasoning      *ai.ReasoningConfig
	// TokenCounter set to nil clears a custom counter and selects from the
	// effective model, falling back to ai.TextTokenEstimator. It never disables
	// counting. Counters supplied here must perform only local work.
	TokenCounter Optional[ai.TokenCounter]
	// RequestBudget replaces the entire budget policy. Set with nil restores
	// prompt-builder inheritance; an explicit zero policy disables limits.
	RequestBudget Optional[*ai.RequestBudgetConfig]
	// RetryPolicy set to nil disables the entire policy, including its timeouts.
	RetryPolicy Optional[*loop.RetryPolicy]
	// ToolResultProcessor set to nil disables the inherited processor.
	ToolResultProcessor Optional[loop.ToolResultProcessor]
}

type resolvedExecution struct {
	model                     ai.Model
	limits                    Limits
	tools                     []loop.Tool
	toolChoice                ai.ToolChoice
	responseFormat            ai.ResponseFormat
	reasoning                 ai.ReasoningConfig
	counter                   ai.TokenCounter
	requestBudget             *ai.RequestBudgetConfig
	retryPolicy               *loop.RetryPolicy
	toolResultProcessor       loop.ToolResultProcessor
	nativeTools               bool
	reconfigureTools          bool
	requireTokenCounterSetter bool
}

// resolveExecution owns the configuration used by a run. It deliberately
// validates after overlaying, so a valid override can replace an invalid default.
func resolveExecution(def Definition, overrides *ExecutionOverrides) (resolvedExecution, error) {
	r := resolvedExecution{
		model: def.Model, limits: def.Limits, tools: def.Tools,
		toolChoice: def.ToolChoice, responseFormat: def.ResponseFormat,
		reasoning: def.Reasoning, counter: def.TokenCounter,
		requestBudget: def.RequestBudget,
		retryPolicy:   def.RetryPolicy, toolResultProcessor: def.ToolResultProcessor,
		reconfigureTools: def.ToolChoice.Mode != "" || len(def.ToolChoice.Names) != 0,
	}
	if overrides != nil {
		if overrides.Model != nil {
			r.model = overrides.Model
			r.reconfigureTools = true
			r.requireTokenCounterSetter = true
		}
		if overrides.Limits.MaxTokens != nil {
			r.limits.MaxTokens = *overrides.Limits.MaxTokens
		}
		if overrides.Limits.MaxLoopIterations != nil {
			r.limits.MaxLoopIterations = *overrides.Limits.MaxLoopIterations
		}
		if overrides.Tools != nil {
			r.tools = overrides.Tools
			r.reconfigureTools = true
		}
		if overrides.ToolChoice != nil {
			r.toolChoice = *overrides.ToolChoice
			r.reconfigureTools = true
		}
		if overrides.ResponseFormat != nil {
			r.responseFormat = *overrides.ResponseFormat
		}
		if overrides.Reasoning != nil {
			r.reasoning = *overrides.Reasoning
		}
		if overrides.TokenCounter.Set {
			r.counter = overrides.TokenCounter.Value
			r.requireTokenCounterSetter = true
		}
		if overrides.RequestBudget.Set {
			r.requestBudget = overrides.RequestBudget.Value
		}
		if overrides.RetryPolicy.Set {
			r.retryPolicy = overrides.RetryPolicy.Value
		}
		if overrides.ToolResultProcessor.Set {
			r.toolResultProcessor = overrides.ToolResultProcessor.Value
		}
	}
	if nilDependency(r.model) {
		return resolvedExecution{}, loop.ErrModelNotConfigured
	}
	if r.limits.MaxTokens < 0 {
		return resolvedExecution{}, fmt.Errorf("%w: MaxTokens must be non-negative", ErrInvalidExecutionConfig)
	}
	if r.requestBudget != nil {
		if err := r.requestBudget.Validate(); err != nil {
			return resolvedExecution{}, fmt.Errorf("%w: %w", ErrInvalidExecutionConfig, err)
		}
		r.requestBudget = clonePointer(r.requestBudget)
	}
	if r.limits.MaxLoopIterations < 0 {
		return resolvedExecution{}, fmt.Errorf("%w: MaxLoopIterations must be non-negative", ErrInvalidExecutionConfig)
	}
	if r.counter != nil {
		if nilDependency(r.counter) {
			return resolvedExecution{}, fmt.Errorf("%w: counter is a typed nil", ErrInvalidExecutionConfig)
		}
		r.requireTokenCounterSetter = true
	}
	if r.toolResultProcessor != nil && nilDependency(r.toolResultProcessor) {
		return resolvedExecution{}, fmt.Errorf("%w: tool response processor is a typed nil", ErrInvalidExecutionConfig)
	}
	if r.retryPolicy != nil {
		if err := r.retryPolicy.Validate(); err != nil {
			return resolvedExecution{}, fmt.Errorf("execution.retry_policy: %w", err)
		}
	}
	if err := r.responseFormat.Validate(); err != nil {
		return resolvedExecution{}, fmt.Errorf("execution.response_format: %w", err)
	}
	r.nativeTools = usesNativeTools(r.model)
	transport := loop.ToolTransportText
	if r.nativeTools {
		transport = loop.ToolTransportNative
	}
	tools, err := loop.EffectiveTools(r.tools, r.toolChoice, transport)
	if err != nil {
		return resolvedExecution{}, fmt.Errorf("execution.tools: %w", err)
	}
	r.tools = cloneTools(tools)
	r.toolChoice = cloneToolChoice(r.toolChoice)
	r.responseFormat = cloneResponseFormat(r.responseFormat)
	r.retryPolicy = cloneRetryPolicy(r.retryPolicy)
	if r.counter == nil {
		if provider, ok := r.model.(ai.TokenCounterProvider); ok {
			r.counter = provider.TokenCounter()
		}
		if r.counter != nil && nilDependency(r.counter) {
			return resolvedExecution{}, fmt.Errorf("%w: model counter is a typed nil", ErrInvalidExecutionConfig)
		}
		if r.counter == nil {
			r.counter = ai.TextTokenEstimator{}
		}
	}
	return r, nil
}

// nilDependency detects typed nils without calling dependency methods.
func nilDependency(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	default:
		return false
	}
}

func cloneDefinition(def Definition) Definition {
	def.RequestBudget = clonePointer(def.RequestBudget)
	def.Tools = cloneTools(def.Tools)
	def.ToolChoice = cloneToolChoice(def.ToolChoice)
	def.ResponseFormat = cloneResponseFormat(def.ResponseFormat)
	def.RetryPolicy = cloneRetryPolicy(def.RetryPolicy)
	def.ToolDefinitionOptions = append([]tooldefinitions.Option(nil), def.ToolDefinitionOptions...)
	def.Middleware = append([]Middleware(nil), def.Middleware...)
	return def
}

func cloneRetryPolicy(policy *loop.RetryPolicy) *loop.RetryPolicy {
	if policy == nil {
		return nil
	}
	copy := *policy
	return &copy
}

func clonePointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func cloneExecution(overrides *ExecutionOverrides) *ExecutionOverrides {
	if overrides == nil {
		return nil
	}
	copy := *overrides
	copy.Limits.MaxTokens = clonePointer(overrides.Limits.MaxTokens)
	copy.Limits.MaxLoopIterations = clonePointer(overrides.Limits.MaxLoopIterations)
	copy.Tools = cloneTools(overrides.Tools)
	if overrides.ToolChoice != nil {
		choice := cloneToolChoice(*overrides.ToolChoice)
		copy.ToolChoice = &choice
	}
	if overrides.ResponseFormat != nil {
		format := cloneResponseFormat(*overrides.ResponseFormat)
		copy.ResponseFormat = &format
	}
	copy.Reasoning = clonePointer(overrides.Reasoning)
	copy.RequestBudget.Value = clonePointer(overrides.RequestBudget.Value)
	copy.RetryPolicy.Value = cloneRetryPolicy(overrides.RetryPolicy.Value)
	return &copy
}
