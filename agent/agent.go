package agent

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/tooldefinitions"
	"github.com/lace-ai/gai/loop"
)

var readRunID = rand.Read

// RunInput contains the application input for one agent run.
type RunInput struct {
	ID string
	// TraceContext explicitly selects approved trace-wide dimensions. Nil
	// inherits a trace context already present in ctx; non-nil replaces it.
	TraceContext *gai.TraceContext
	// Prompt separates genuine user content from structured machine context.
	Prompt gaictx.PromptInput
	// MaxTokens overrides Definition.Limits.MaxTokens when it is positive.
	MaxTokens int
	// ResponseFormat requests the output shape for every model call in this run.
	ResponseFormat ai.ResponseFormat
	// Execution overrides selected definition-level execution settings for this run.
	// Tools replaces Definition.Tools when non-nil; an empty non-nil slice disables
	// definition-level tools. Nil inherits Definition.Tools.
	Execution ExecutionConfig
	// Meta carries application data such as user, session, or request IDs.
	Meta map[string]any
}

// ExecutionConfig provides optional configuration for one workflow.
//
// Tool slices are copied when NewRun is called, preserving tool membership and
// order for that workflow. Tool implementations themselves must remain safe for
// concurrent use.
type ExecutionConfig struct {
	Tools []loop.Tool
	// ToolChoice overrides the provider tool-choice setting when non-nil.
	ToolChoice *ai.ToolChoice
	// Reasoning overrides Definition.Reasoning when non-nil.
	Reasoning *ai.ReasoningConfig
}

// Prompt creates and returns a run-owned prompt builder used by one workflow.
// Callers must return a fresh builder for every invocation; Agent mutates the
// returned builder while applying that run's input and execution configuration.
type Prompt func(ctx context.Context, input RunInput) (gaictx.PromptBuilder, error)

// Limits controls loop iterations and model output size.
type Limits struct {
	// MaxLoopIterations limits model/tool iterations. Zero uses the loop default.
	MaxLoopIterations int
	// MaxTokens is the default model output limit for the agent.
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
	// prompt. Models without a descriptor use ai.NativeToolModel as a legacy
	// fallback. Otherwise, the protocol is added as the first prompt context
	// source unless its builder already contains a tool_definitions source.
	Tools []loop.Tool
	// ToolDefinitionOptions configure the auto-prepended tool-definitions prompt
	// source used for Tools.
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
	// Tokenizer overrides Model.Tokenizer when it is non-nil.
	Tokenizer ai.Tokenizer
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

// New creates an agent from def. Configuration is validated by NewRun.
func New(def Definition) *Agent {
	return &Agent{def: def}
}

// NewRun builds a single-use workflow for input.
//
// Prompt construction happens before NewRun returns. Model execution and
// middleware processing begin when Workflow.Run or Workflow.RunEvents is called.
func (a *Agent) NewRun(ctx context.Context, input RunInput) (*Workflow, error) {
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
	l, err := a.newLoop(ctx, input)
	if err != nil {
		obs.Failed(ctx, "loop_creation", err)
		obs.Finish(err)
		return nil, err
	}
	workflow := newWorkflow(input, l, a.name(), a.debugSink(), a.middleware())
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

func (a *Agent) newLoop(ctx context.Context, input RunInput) (*loop.Loop, error) {
	if a == nil {
		return nil, loop.ErrNilLoop
	}
	if a.def.Model == nil {
		return nil, loop.ErrModelNotConfigured
	}
	if a.def.Prompt == nil {
		return nil, loop.ErrPromptNotConfigured
	}
	nativeTools := usesNativeTools(a.def.Model)
	execution, err := resolveExecution(a.def.Tools, input.Execution, nativeTools)
	if err != nil {
		return nil, err
	}

	promptBuilder, err := a.def.Prompt(ctx, input)
	if err != nil {
		return nil, err
	}
	if promptBuilder == nil {
		return nil, loop.ErrPromptNotConfigured
	}
	promptBuilder.SetInput(input.Prompt)
	if !nativeTools {
		lookup, hasContextSourceLookup := promptBuilder.(contextSourceLookup)
		hasToolDefinitions := hasContextSourceLookup && lookup.HasContextSource("tool_definitions")
		manager, hasContextSourceManager := promptBuilder.(contextSourceManager)
		if (execution.toolsOverridden || execution.textToolsConfigured) && hasToolDefinitions && len(execution.tools) == 0 {
			if !hasContextSourceManager {
				return nil, fmt.Errorf("prompt builder cannot remove existing tool definitions")
			}
			if err := manager.RemoveContextSource(ctx, "tool_definitions"); err != nil {
				return nil, err
			}
		} else if len(execution.tools) > 0 && (!hasToolDefinitions || execution.toolsOverridden || execution.textToolsConfigured) {
			toolOptions := append([]tooldefinitions.Option(nil), a.def.ToolDefinitionOptions...)
			if execution.hasToolChoice {
				toolOptions = append(toolOptions, tooldefinitions.WithToolChoice(execution.toolChoice))
			}
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
			} else if err := promptBuilder.PrependContextSource(ctx, toolSource); err != nil {
				return nil, err
			}
		}
	}
	if setter, ok := promptBuilder.(gaictx.TokenizerSetter); ok {
		tokenizer := a.def.Tokenizer
		if tokenizer == nil {
			tokenizer = a.def.Model.Tokenizer()
		}
		if tokenizer != nil {
			setter.SetTokenizer(tokenizer)
		}
	}

	l := loop.New(a.def.Model, execution.tools, promptBuilder, a.def.ToolResponseProcessor)
	l.ObservationSink = a.def.ObservationSink
	if !nativeTools {
		l.ToolTransport = loop.ToolTransportText
	}
	if a.def.Limits.MaxLoopIterations > 0 {
		l.MaxLoopIterations = a.def.Limits.MaxLoopIterations
	}
	if input.MaxTokens > 0 {
		l.MaxTokens = input.MaxTokens
	} else {
		l.MaxTokens = a.def.Limits.MaxTokens
	}
	l.ResponseFormat = cloneResponseFormat(input.ResponseFormat)
	l.Reasoning = a.def.Reasoning
	if execution.hasToolChoice {
		l.ToolChoice = execution.toolChoice
	}
	if input.Execution.Reasoning != nil {
		l.Reasoning = *input.Execution.Reasoning
	}
	if a.def.RetryPolicy != nil {
		policy := *a.def.RetryPolicy
		l.RetryPolicy = &policy
	}
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

type executionResolution struct {
	tools               []loop.Tool
	toolChoice          ai.ToolChoice
	hasToolChoice       bool
	toolsOverridden     bool
	textToolsConfigured bool
}

// resolveExecution snapshots one run's tools and choice, then resolves the
// effective text-transport tool set before prompt or loop construction.
func resolveExecution(definitionTools []loop.Tool, config ExecutionConfig, nativeTools bool) (executionResolution, error) {
	resolved := executionResolution{
		tools:           cloneTools(definitionTools),
		toolsOverridden: config.Tools != nil,
	}
	if config.Tools != nil {
		resolved.tools = cloneTools(config.Tools)
	}
	if config.ToolChoice != nil {
		if err := config.ToolChoice.Validate(); err != nil {
			return executionResolution{}, err
		}
		resolved.toolChoice = cloneToolChoice(*config.ToolChoice)
		resolved.hasToolChoice = true
	}
	transport := loop.ToolTransportNative
	if !nativeTools {
		transport = loop.ToolTransportText
	}
	effectiveTools, err := loop.EffectiveTools(resolved.tools, resolved.toolChoice, transport)
	if err != nil {
		return executionResolution{}, err
	}
	resolved.tools = effectiveTools
	if nativeTools || !resolved.hasToolChoice {
		return resolved, nil
	}
	resolved.textToolsConfigured = resolved.toolChoice.Mode == ai.ToolChoiceNone || resolved.toolChoice.Mode == ai.ToolChoiceRequired
	return resolved, nil
}

func usesNativeTools(model ai.Model) bool {
	if describer, ok := model.(ai.ModelDescriber); ok {
		return describer.Descriptor().SupportsNativeTools()
	}
	native, ok := model.(interface{ NativeTools() bool })
	return ok && native.NativeTools()
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
