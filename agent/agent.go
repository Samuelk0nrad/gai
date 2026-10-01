package agent

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/tooldefinitions"
	"github.com/lace-ai/gai/loop"
)

var readRunID = rand.Read

// ErrPromptInputNotConfigurable means a prompt builder cannot receive the
// current run's input through SetInput(context.PromptInput).
var ErrPromptInputNotConfigurable = errors.New("prompt builder does not implement SetInput(context.PromptInput)")

// RunInput contains the application input for one agent run.
type RunInput struct {
	ID string
	// TraceContext explicitly selects approved trace-wide dimensions. Nil
	// inherits a trace context already present in ctx; non-nil replaces it.
	TraceContext *gai.TraceContext
	// Prompt separates genuine user content from structured machine context.
	Prompt gaictx.PromptInput
	// Execution overrides definition-level execution settings for this run.
	// Nil and an empty block both inherit all defaults.
	Execution *ExecutionOverrides
	// Meta carries application data such as user, session, or request IDs.
	Meta map[string]any
}

// Prompt creates and returns a run-owned prompt builder used by one workflow.
// Callers must return a fresh builder for every invocation; Agent mutates the
// returned builder while applying that run's input and execution configuration.
// The builder must implement SetInput(context.PromptInput); otherwise NewRun
// returns ErrPromptInputNotConfigurable. For text-based tools it must also
// implement PrependContextSource(context.Context, context.ContextSource) error
// unless HasContextSource already reports a tool_definitions source. Run
// overrides that change an existing tool source additionally require
// ReplaceContextSource and RemoveContextSource, as provided by context.Builder.
type Prompt func(ctx context.Context, input RunInput) (gaictx.PromptBuilder, error)

// Limits controls loop iterations and model output size.
type Limits struct {
	// MaxLoopIterations limits model/tool iterations. Zero uses the loop default.
	// Negative limits are invalid.
	MaxLoopIterations int
	// MaxTokens is the default output limit for each model generation.
	// Zero uses adapter/provider defaults. Negative limits are invalid.
	MaxTokens int
}

// Definition describes a reusable agent and its workflow middleware.
type Definition struct {
	// Name identifies the agent in diagnostics and is the default name used when
	// the agent is adapted into middleware.
	Name string
	// Model performs the agent's model calls.
	Model ai.Model
	// Tools are available to the model during loop execution. When the model's
	// ai.ModelDescriber descriptor reports native tools are supported, their
	// definitions and text-based invocation protocol are not added to the
	// prompt. Otherwise, the protocol is added as the first prompt context
	// source unless its builder already contains a tool_definitions source.
	Tools []loop.Tool
	// ToolChoice is the default tool-use policy. The zero value uses the
	// loop/provider default. Run overrides replace this policy atomically.
	ToolChoice ai.ToolChoice
	// ResponseFormat is the default output shape for each model generation.
	ResponseFormat ai.ResponseFormat
	// ToolDefinitionOptions configure the auto-prepended tool-definitions prompt
	// source used for Tools. The resolved ToolChoice takes precedence over
	// WithToolChoice options; use the ToolChoice field for execution policy.
	ToolDefinitionOptions []tooldefinitions.Option
	// Prompt builds run-specific instructions and context.
	Prompt Prompt
	// Limits configures loop execution defaults.
	Limits Limits
	// RetryPolicy enables classified model-generation retries. Nil disables retries.
	// Each workflow receives its own copy of the configured policy.
	RetryPolicy *loop.RetryPolicy
	// Reasoning configures model reasoning/thinking behavior for every model call.
	Reasoning ai.ReasoningConfig
	// TokenCounter supplies a local-only text counter. Nil selects the optional
	// model counter, then ai.TextTokenEstimator if no model counter is available.
	TokenCounter ai.TokenCounter
	// ToolResponseProcessor can transform tool responses before they enter the transcript.
	ToolResponseProcessor loop.ToolResponseProcessor
	// ObservationSink receives agent and workflow lifecycle events.
	ObservationSink gai.ObservationSink
	// Middleware transforms the run stream in declaration order.
	Middleware []Middleware
}

// Agent is a reusable definition that creates independent workflows.
type Agent struct {
	def Definition
}

// New snapshots the definition's mutable configuration. Models, tools,
// counters, processors, and callbacks remain caller-owned shared dependencies.
// The effective configuration is validated by NewRun.
func New(def Definition) *Agent {
	return &Agent{def: cloneDefinition(def)}
}

// NewRun builds a single-use workflow for input.
//
// Prompt construction happens before NewRun returns. Model execution and
// middleware processing begin when Workflow.Run or Workflow.RunEvents is called.
func (a *Agent) NewRun(ctx context.Context, input RunInput) (*Workflow, error) {
	input = cloneRunInput(input)
	if input.ID == "" {
		var err error
		input.ID, err = newRunID()
		if err != nil {
			return nil, err
		}
	}
	ctx, input = resolveRunTraceContext(ctx, input)
	ctx = gai.WithObservationRunID(ctx, input.ID)
	ctx, obs := newRunCreationObserver(ctx, a, input)
	if a != nil {
		if err := validateMiddleware(a.def.Middleware); err != nil {
			obs.Failed(ctx, "middleware_validation", err)
			obs.Finish(err)
			return nil, err
		}
	}
	if a == nil {
		obs.Failed(ctx, "execution_resolution", loop.ErrNilLoop)
		obs.Finish(loop.ErrNilLoop)
		return nil, loop.ErrNilLoop
	}
	execution, err := resolveExecution(a.def, input.Execution)
	if err != nil {
		obs.Failed(ctx, "execution_resolution", err)
		obs.Finish(err)
		return nil, err
	}
	obs.Resolved(execution)
	l, err := a.newLoop(ctx, input, execution)
	if err != nil {
		obs.Failed(ctx, "loop_creation", err)
		obs.Finish(err)
		return nil, err
	}
	workflow := newWorkflow(input, l, a.name(), a.debugSink(), a.middleware())
	obs.LoopConfigured(l.MaxLoopIterations)
	obs.Created(ctx)
	obs.Finish(nil)
	return workflow, nil
}

func newRunID() (string, error) {
	var bytes [16]byte
	if _, err := readRunID(bytes[:]); err != nil {
		return "", err
	}
	return "run_" + hex.EncodeToString(bytes[:]), nil
}

func resolveRunTraceContext(ctx context.Context, input RunInput) (context.Context, RunInput) {
	if input.TraceContext != nil {
		ctx = gai.WithTraceContext(ctx, *input.TraceContext)
	}
	if traceContext, ok := gai.TraceContextFromContext(ctx); ok {
		input.TraceContext = &traceContext
	}
	return ctx, input
}

func (a *Agent) name() string {
	if a == nil {
		return ""
	}
	return a.def.Name
}

func (a *Agent) debugSink() gai.ObservationSink {
	if a == nil {
		return nil
	}
	return a.def.ObservationSink
}

func (a *Agent) middleware() []Middleware {
	if a == nil {
		return nil
	}
	return a.def.Middleware
}

func (a *Agent) newLoop(ctx context.Context, input RunInput, execution resolvedExecution) (*loop.Loop, error) {
	if a.def.Prompt == nil {
		return nil, loop.ErrPromptNotConfigured
	}
	nativeTools := execution.nativeTools
	promptBuilder, err := a.def.Prompt(ctx, cloneRunInput(input))
	if err != nil {
		return nil, err
	}
	if nilDependency(promptBuilder) {
		return nil, loop.ErrPromptNotConfigured
	}
	inputSetter, ok := promptBuilder.(promptInputSetter)
	if !ok {
		return nil, ErrPromptInputNotConfigurable
	}
	inputSetter.SetInput(input.Prompt)
	lookup, hasContextSourceLookup := promptBuilder.(contextSourceLookup)
	hasToolDefinitions := hasContextSourceLookup && lookup.HasContextSource("tool_definitions")
	manager, hasContextSourceManager := promptBuilder.(contextSourceManager)
	if nativeTools && execution.reconfigureTools && hasToolDefinitions {
		if !hasContextSourceManager {
			return nil, fmt.Errorf("prompt builder cannot remove existing tool definitions")
		}
		if err := manager.RemoveContextSource(ctx, "tool_definitions"); err != nil {
			return nil, err
		}
	} else if !nativeTools {
		if execution.reconfigureTools && hasToolDefinitions && len(execution.tools) == 0 {
			if !hasContextSourceManager {
				return nil, fmt.Errorf("prompt builder cannot remove existing tool definitions")
			}
			if err := manager.RemoveContextSource(ctx, "tool_definitions"); err != nil {
				return nil, err
			}
		} else if len(execution.tools) > 0 && (!hasToolDefinitions || execution.reconfigureTools) {
			toolOptions := append([]tooldefinitions.Option(nil), a.def.ToolDefinitionOptions...)
			toolOptions = append(toolOptions, tooldefinitions.WithToolChoice(execution.toolChoice))
			toolSource, err := tooldefinitions.New(nil, toolSignatures(execution.tools), a.def.ObservationSink, toolOptions...)
			if err != nil {
				return nil, err
			}
			if hasToolDefinitions {
				if !hasContextSourceManager {
					return nil, fmt.Errorf("prompt builder cannot replace existing tool definitions")
				}
				if err := manager.ReplaceContextSource(ctx, "tool_definitions", toolSource); err != nil {
					return nil, err
				}
			} else {
				prepender, ok := promptBuilder.(contextSourcePrepender)
				if !ok {
					return nil, fmt.Errorf("prompt builder cannot prepend tool definitions: does not implement PrependContextSource")
				}
				if err := prepender.PrependContextSource(ctx, toolSource); err != nil {
					return nil, err
				}
			}
		}
	}
	if setter, ok := promptBuilder.(gaictx.TokenCounterSetter); ok {
		setter.SetTokenCounter(execution.counter)
	} else if execution.requireTokenCounterSetter {
		return nil, ErrTokenCounterNotConfigurable
	}

	l := loop.New(execution.model, execution.tools, promptBuilder, execution.toolResponseProcessor)
	l.ObservationSink = a.def.ObservationSink
	if !nativeTools {
		l.ToolTransport = loop.ToolTransportText
	}
	if execution.limits.MaxLoopIterations > 0 {
		l.MaxLoopIterations = execution.limits.MaxLoopIterations
	}
	l.MaxTokens = execution.limits.MaxTokens
	l.ResponseFormat = execution.responseFormat
	l.Reasoning = execution.reasoning
	l.ToolChoice = execution.toolChoice
	l.RetryPolicy = execution.retryPolicy
	return l, nil
}

func cloneTools(tools []loop.Tool) []loop.Tool {
	if tools == nil {
		return nil
	}
	cloned := make([]loop.Tool, len(tools))
	copy(cloned, tools)
	return cloned
}

func toolSignatures(tools []loop.Tool) []gaictx.ToolSignature {
	signatures := make([]gaictx.ToolSignature, len(tools))
	for index, tool := range tools {
		signatures[index] = tool
	}
	return signatures
}

func cloneToolChoice(choice ai.ToolChoice) ai.ToolChoice {
	cloned := choice
	cloned.Names = append([]string(nil), choice.Names...)
	return cloned
}

func usesNativeTools(model ai.Model) bool {
	if describer, ok := model.(ai.ModelDescriber); ok {
		return describer.Descriptor().SupportsNativeTools()
	}
	return false
}

// promptInputSetter is required by Agent to inject each run's input, but is not
// needed by loops whose prompt builders already own their input.
type promptInputSetter interface {
	SetInput(input gaictx.PromptInput)
}

// contextSourcePrepender is needed only when Agent injects text tool definitions.
type contextSourcePrepender interface {
	PrependContextSource(ctx context.Context, source gaictx.ContextSource) error
}

// contextSourceLookup is the optional agent-internal prompt-builder capability
// for inspecting named context sources.
type contextSourceLookup interface {
	HasContextSource(name string) bool
}

// contextSourceManager is the optional agent-internal prompt-builder capability
// for mutating named context sources. It is required only when a run changes an
// existing source.
type contextSourceManager interface {
	ReplaceContextSource(ctx context.Context, name string, source gaictx.ContextSource) error
	RemoveContextSource(ctx context.Context, name string) error
}
