package loop

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

const (
	defaultMaxLoopIterations = 8
)

// ToolTransportMode controls whether a loop sends tool definitions through the
// provider-native AIRequest.Tools field. It does not affect Loop.Tools, which
// always remains available to resolve and execute tool calls.
type ToolTransportMode uint8

const (
	// ToolTransportNative sends Loop.Tools as AIRequest.Tools by default.
	ToolTransportNative ToolTransportMode = iota
	// ToolTransportText omits AIRequest.Tools for text tool protocols.
	ToolTransportText
)

// ToolResultProcessor transforms a result before it enters the conversation.
// Returning an error terminates the run. To reject output without terminating,
// return a ToolResult containing a safe Err and a nil processing error.
// Implementations are shared dependencies and must be concurrency-safe.
type ToolResultProcessor interface {
	Process(context.Context, ToolPolicyInput, ToolResult) (ToolResult, error)
}

// ToolResultProcessorFunc adapts a function to ToolResultProcessor.
type ToolResultProcessorFunc func(context.Context, ToolPolicyInput, ToolResult) (ToolResult, error)

// Process invokes the adapter or rejects a nil processor function.
func (f ToolResultProcessorFunc) Process(ctx context.Context, input ToolPolicyInput, result ToolResult) (ToolResult, error) {
	if f == nil {
		return ToolResult{}, fmt.Errorf("%w: processor function is nil", ErrToolResultProcess)
	}
	return f(ctx, input, result)
}

// Loop coordinates prompt construction, model generation, and tool execution.
//
// Use New for initialized defaults. A Loop stores run state in Iterations and
// must not be run concurrently or reused without explicitly clearing that
// state.
type Loop struct {
	// ObservationSink receives finalized normalized loop observations.
	ObservationSink gai.ObservationSink
	// Iterations contains completed model/tool interaction rounds.
	Iterations []Iteration
	// Model generates tokens for each iteration.
	Model ai.Model
	// Tools contains the functions available to the model.
	Tools []Tool
	// ToolChoice controls whether and how the model may call Tools.
	ToolChoice ai.ToolChoice
	// ToolTransport controls whether Tools are serialized into AIRequest.Tools.
	// The default is ToolTransportNative; Tools remain executable in either mode.
	ToolTransport ToolTransportMode
	// MaxLoopIterations limits model/tool interaction rounds.
	MaxLoopIterations int
	// MaxTokens limits model output for each generation request.
	MaxTokens int
	// RequestBudget replaces the prompt builder's inherited request window.
	// Nil inherits a builder exposing RequestBudget; zero limits disable checks.
	RequestBudget *ai.RequestBudgetConfig
	// TokenCounter is the local request estimator override. Nil selects the
	// builder counter, model counter, then generic estimator.
	TokenCounter ai.TokenCounter
	// ResponseFormat requests the output shape for each model generation.
	ResponseFormat ai.ResponseFormat
	// Reasoning configures model reasoning/thinking behavior for each model generation.
	Reasoning ai.ReasoningConfig
	// RetryPolicy enables classified retries. Nil disables retries.
	RetryPolicy *RetryPolicy
	// PromptBuilder constructs the prompt for each iteration.
	PromptBuilder gaictx.PromptBuilder
	// ToolResultProcessor optionally processes tool responses before they are recorded, emitted, or captured by loop telemetry.
	ToolResultProcessor ToolResultProcessor
}

// Validate applies default iteration limits and checks required loop dependencies.
func (l *Loop) Validate() error {
	if l == nil {
		return ErrNilLoop
	}
	if l.MaxLoopIterations <= 0 {
		l.MaxLoopIterations = defaultMaxLoopIterations
	}
	if l.Model == nil {
		return ErrModelNotConfigured
	}
	if l.PromptBuilder == nil {
		return ErrPromptNotConfigured
	}
	if l.RequestBudget != nil {
		if err := l.RequestBudget.Validate(); err != nil {
			return err
		}
	}
	if l.MaxTokens < 0 {
		return fmt.Errorf("%w: negative output limit", ai.ErrInvalidRequestBudget)
	}
	if l.ToolResultProcessor != nil && nilImplementation(l.ToolResultProcessor) {
		return fmt.Errorf("%w: processor is a typed nil", ErrToolResultProcess)
	}
	if l.RetryPolicy != nil {
		if err := l.RetryPolicy.Validate(); err != nil {
			return err
		}
	}
	if err := l.ResponseFormat.Validate(); err != nil {
		return err
	}
	if _, err := EffectiveTools(l.Tools, l.ToolChoice, l.ToolTransport); err != nil {
		return err
	}
	return nil
}

// EffectiveTools validates and resolves the run-scoped executable tool set for
// a transport and tool choice. Text transport omits disabled tools and limits
// named choices to the tools rendered in its prompt. Native transport preserves
// the configured set except that required named choices are similarly limited
// for provider-side choice handling.
func EffectiveTools(tools []Tool, choice ai.ToolChoice, transport ToolTransportMode) ([]Tool, error) {
	switch transport {
	case ToolTransportNative, ToolTransportText:
	default:
		return nil, fmt.Errorf("invalid tool transport mode: %d", transport)
	}
	if err := choice.Validate(); err != nil {
		return nil, err
	}
	if _, err := ToolDefinitions(tools); err != nil {
		return nil, err
	}
	if choice.Mode == ai.ToolChoiceRequired {
		if len(tools) == 0 {
			return nil, ErrRequiredToolNotConfigured
		}
		for _, requiredName := range choice.Names {
			if !slices.ContainsFunc(tools, func(tool Tool) bool {
				return tool != nil && tool.Name() == requiredName
			}) {
				return nil, fmt.Errorf("%w: %q", ErrRequiredToolNotConfigured, requiredName)
			}
		}
	}
	if transport == ToolTransportText && choice.Mode == ai.ToolChoiceNone {
		return []Tool{}, nil
	}
	if len(choice.Names) > 0 && (transport == ToolTransportText || choice.Mode == ai.ToolChoiceRequired) {
		return toolsNamed(tools, choice.Names), nil
	}
	return tools, nil
}

// New constructs a Loop with the default iteration limit.
func New(model ai.Model, tools []Tool, promptBuilder gaictx.PromptBuilder, toolResultProcessor ToolResultProcessor) *Loop {
	l := &Loop{
		Model:               model,
		Tools:               tools,
		MaxLoopIterations:   defaultMaxLoopIterations,
		PromptBuilder:       promptBuilder,
		ToolResultProcessor: toolResultProcessor,
	}
	return l
}

type pendingToolCall struct {
	partIndex int
	call      ai.ToolCall
}

// Run starts asynchronous model and tool execution.
//
// The returned channel carries every token, retry, iteration, and terminal
// event in the exact order it occurred. Tokens are forwarded in real time;
// when an attempt is retried, consumers that keep visible token state must
// discard that attempt's tokens on its RetryEvent. Callers must consume the
// channel until it closes or cancel ctx.
func (l *Loop) Run(ctx context.Context) <-chan Event {
	events := make(chan Event, 32)

	if err := l.Validate(); err != nil {
		events <- ErrorEvent(err)
		close(events)
		return events
	}

	go l.run(ctx, events)
	return events
}

// run owns the asynchronous execution flow after Run has synchronously
// validated the public configuration.
func (l *Loop) run(ctx context.Context, events chan<- Event) {
	l.executeRun(ctx, events)
}

// buildAttemptRequest builds exactly the provider request for one model attempt.
// It deliberately runs before the attempt deadline is installed: prompt building
// belongs to iteration preparation, not model generation.
func (l *Loop) buildAttemptRequest(ctx context.Context, toolDefinitions []ai.ToolDefinition, requiredToolCallSatisfied bool) (ai.AIRequest, error) {
	request, err := l.PromptBuilder.BuildRequest(ctx, attemptConversation{loop: l})
	if err != nil {
		return ai.AIRequest{}, err
	}

	toolChoice := l.ToolChoice
	if toolChoice.Mode == ai.ToolChoiceRequired && requiredToolCallSatisfied {
		toolChoice = ai.ToolChoice{Mode: ai.ToolChoiceAuto}
	}
	if l.ToolTransport == ToolTransportText || len(toolDefinitions) == 0 {
		// Text tool calling exposes definitions through message context. A neutral
		// choice is likewise required when native tool definitions are absent.
		toolChoice = ai.ToolChoice{}
	}
	request.MaxTokens = l.MaxTokens
	request.Tools = toolDefinitions
	request.ToolChoice = toolChoice
	request.ResponseFormat = l.ResponseFormat
	request.Reasoning = l.Reasoning
	// Snapshot the canonical request after applying execution settings.
	request = request.Copy()
	if err := request.Validate(); err != nil {
		return ai.AIRequest{}, err
	}
	return request, nil
}

func userMessageForIteration(promptBuilder gaictx.PromptBuilder, index int) *ai.Message {
	if index != 0 || promptBuilder == nil {
		return nil
	}
	input := promptBuilder.Input()
	if len(input.User) == 0 {
		return nil
	}
	return &ai.Message{Role: ai.RoleUser, Parts: ai.CloneParts(input.User)}
}

// hasPermittedToolCall reports whether the model requested at least one valid
// configured tool allowed by the required tool choice, with no rejected calls.
// Calls for unavailable, malformed, or unselected tools reject the response.
func hasPermittedToolCall(toolCalls []pendingToolCall, tools []Tool, allowedNames []string) bool {
	hasPermittedCall := false
	for _, toolCall := range toolCalls {
		if err := toolCall.call.Validate(); err != nil {
			return false
		}
		if len(allowedNames) > 0 && !slices.Contains(allowedNames, toolCall.call.Name) {
			return false
		}
		configured := false
		for _, tool := range tools {
			if tool != nil && tool.Name() == toolCall.call.Name {
				configured = true
				break
			}
		}
		if !configured {
			return false
		}
		hasPermittedCall = true
	}
	return hasPermittedCall
}

// toolsNamed selects registered tools without changing their configured order.
func toolsNamed(tools []Tool, names []string) []Tool {
	selected := make([]Tool, 0, len(names))
	for _, tool := range tools {
		if tool != nil && slices.Contains(names, tool.Name()) {
			selected = append(selected, tool)
		}
	}
	return selected
}

// executeToolCalls records tool responses on iteration. Tool execution
// failures are stored in ToolResult.Err and are not returned. Only framework
// or tool-response processing failures are returned.
func (l *Loop) executeToolCalls(ctx context.Context, iteration *Iteration, toolCalls []pendingToolCall, tools []Tool, events chan<- Event, iterationCount, attemptID, retryCount int) error {
	var wg sync.WaitGroup
	// A failed later start event must not leave tools mutating a snapshot.
	defer wg.Wait()
	var toolErr error
	var toolErrMu sync.Mutex

	for _, tc := range toolCalls {
		if events != nil {
			if err := sendEvent(ctx, events, ToolStartEvent(iterationCount, attemptID, retryCount, tc.call)); err != nil {
				return err
			}
		}
		wg.Add(1)
		go func(tc pendingToolCall) {
			defer wg.Done()

			toolRes, duration, processErr := processObservedTool(ctx, ToolPolicyInput{Call: tc.call}, tools, l.ToolResultProcessor, l.ObservationSink)
			if processErr != nil {
				toolErrMu.Lock()
				if toolErr == nil {
					toolErr = processErr
				}
				toolErrMu.Unlock()
				if events != nil {
					if err := sendEvent(ctx, events, ToolErrorEvent(iterationCount, attemptID, retryCount, tc.call, nil, duration, processErr)); err != nil {
						toolErrMu.Lock()
						if toolErr == nil {
							toolErr = err
						}
						toolErrMu.Unlock()
					}
				}
				return
			}
			iteration.Parts[tc.partIndex].ToolResp = toolRes
			if events != nil {
				if err := sendEvent(ctx, events, ToolResultEvent(iterationCount, attemptID, retryCount, tc.call, toolRes, duration)); err != nil {
					toolErrMu.Lock()
					if toolErr == nil {
						toolErr = err
					}
					toolErrMu.Unlock()
				}
			}
		}(tc)
	}
	wg.Wait()
	if toolErr == nil {
		for _, tc := range toolCalls {
			response := iteration.Parts[tc.partIndex].ToolResp
			if response == nil {
				continue
			}
			result := ai.ToolResult{ToolCallID: tc.call.ID, Name: tc.call.Name, Parts: ai.TextParts(response.Text)}
			if err := response.Err; err != nil {
				result.IsError = true
				result.Parts = ai.TextParts(err.Error())
			}
			iteration.Conversation = append(iteration.Conversation, ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &result}}})
		}
	}
	return toolErr
}

func sendLoopError(ctx context.Context, events chan<- Event, state *loopRunState, err error) {
	if state != nil {
		state.fail(err)
	}
	_ = sendEvent(ctx, events, ErrorEvent(err))
}

func sendAttemptError(ctx context.Context, events chan<- Event, state *loopRunState, iterationCount, attemptID, retryCount int, attemptIteration *Iteration, err error) {
	if state != nil {
		state.fail(err)
	}
	_ = sendEvent(ctx, events, AttemptErrorEvent(iterationCount, attemptID, retryCount, attemptIteration, err))
}

func sendLoopCanceled(ctx context.Context, events chan<- Event, state *loopRunState, err error) {
	if state != nil {
		state.cancel(err)
	}
	sendTerminalEvent(ctx, events, CanceledEvent(err))
}

func sendAttemptCanceled(ctx context.Context, events chan<- Event, state *loopRunState, iterationCount, attemptID, retryCount int, attemptIteration *Iteration, err error) {
	if state != nil {
		state.cancel(err)
	}
	sendTerminalEvent(ctx, events, AttemptCanceledEvent(iterationCount, attemptID, retryCount, attemptIteration, err))
}

func sendTerminalEvent(_ context.Context, events chan<- Event, event Event) {
	select {
	case events <- event:
	default:
	}
}

func cancellationError(ctx context.Context, err error) error {
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	return nil
}

// Messages returns the completed iterations as ordered conversation messages.
func (l *Loop) Messages() []ai.Message {
	var msgs []ai.Message

	for _, i := range l.Iterations {
		msgs = append(msgs, i.Messages()...)
	}

	return msgs
}

// attemptConversation excludes only the input recorded by the loop itself;
// PromptBuilder.Input supplies that message. No content-based deduplication is
// applied to arbitrary conversation messages.
type attemptConversation struct{ loop *Loop }

func (c attemptConversation) Messages() []ai.Message {
	var messages []ai.Message
	for i := range c.loop.Iterations {
		messages = append(messages, c.loop.Iterations[i].DeltaMessages()...)
	}
	return messages
}
