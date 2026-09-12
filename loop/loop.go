package loop

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

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
	// ToolTransportNative sends Loop.Tools as AIRequest.Tools. This is the
	// default to preserve direct loop.New compatibility.
	ToolTransportNative ToolTransportMode = iota
	// ToolTransportText omits AIRequest.Tools for prompt-rendered tool protocols.
	ToolTransportText
)

// ToolResponseProcessor can inspect or modify a tool response before the loop
// records it and builds the next prompt. Implementations must be safe for
// concurrent use.
type ToolResponseProcessor interface {
	// Process handles the response produced for req.
	Process(req ai.ToolCall, res *ToolResponse) error
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
	// ResponseFormat requests the output shape for each model generation.
	ResponseFormat ai.ResponseFormat
	// Reasoning configures model reasoning/thinking behavior for each model generation.
	Reasoning ai.ReasoningConfig
	// RetryPolicy enables classified retries. Nil disables retries.
	RetryPolicy *RetryPolicy
	// PromptBuilder constructs the prompt for each iteration.
	PromptBuilder gaictx.PromptBuilder
	// ToolResponseProcessor optionally processes tool responses after they are recorded on the iteration and before it is persisted.
	ToolResponseProcessor ToolResponseProcessor
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
func New(model ai.Model, tools []Tool, promptBuilder gaictx.PromptBuilder, toolResponseProcessor ToolResponseProcessor) *Loop {
	l := &Loop{
		Model:                 model,
		Tools:                 tools,
		MaxLoopIterations:     defaultMaxLoopIterations,
		PromptBuilder:         promptBuilder,
		ToolResponseProcessor: toolResponseProcessor,
	}
	return l
}

type pendingToolCall struct {
	partIndex int
	call      ai.ToolCall
}

// renderedPromptRequest creates the compatibility request used by the loop.
// Conversation state remains in Prompt until a provider-native message path is
// introduced deliberately; this boundary keeps that future change separate
// from the current rendered-prompt behavior.
func renderedPromptRequest(prompt string, maxTokens int, tools []ai.ToolDefinition, toolChoice ai.ToolChoice, responseFormat ai.ResponseFormat, reasoning ai.ReasoningConfig) ai.AIRequest {
	return ai.AIRequest{
		Prompt:         prompt,
		MaxTokens:      maxTokens,
		Tools:          tools,
		ToolChoice:     toolChoice,
		ResponseFormat: responseFormat,
		Reasoning:      reasoning,
	}
}

// NativeMessages returns the provider-neutral history for completed iterations.
// Prompt builders combine it with their own base user message.
func (l *Loop) NativeMessages() []ai.RequestMessage {
	var messages []ai.RequestMessage
	if l == nil {
		return nil
	}
	iterations := l.Iterations
	for _, iteration := range iterations {
		var text string
		var toolCalls []ai.RequestToolCall
		var toolResults []ai.RequestMessage
		for _, part := range iteration.Parts {
			switch part.Type {
			case IterationTypeResponse:
				if part.Response != nil && part.Response.Text != "" {
					text += part.Response.Text
				}
			case IterationTypeToolCall, IterationTypeToolError:
				if part.ToolReq == nil {
					continue
				}
				toolCalls = append(toolCalls, ai.RequestToolCall{
					ID:               part.ToolReq.ID,
					Name:             part.ToolReq.Name,
					Arguments:        append([]byte(nil), part.ToolReq.Args...),
					ThoughtSignature: append([]byte(nil), part.ToolReq.ThoughtSignature...),
				})
				if part.ToolResp != nil {
					result := ai.RequestToolResult{ToolCallID: part.ToolReq.ID, Name: part.ToolReq.Name, Content: part.ToolResp.TextValue()}
					if err := part.ToolResp.ErrorValue(); err != nil {
						result.Content = err.Error()
						result.IsError = true
					}
					toolResults = append(toolResults, ai.RequestMessage{Role: ai.RequestMessageRoleTool, ToolResult: &result})
				}
			}
		}
		if text != "" || len(toolCalls) > 0 {
			messages = append(messages, ai.RequestMessage{Role: ai.RequestMessageRoleAssistant, Text: text, ToolCalls: toolCalls})
			messages = append(messages, toolResults...)
		}
	}
	return messages
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
	callerCtx := ctx
	if l.RetryPolicy != nil && l.RetryPolicy.TotalTimeout > 0 {
		var totalCancel context.CancelFunc
		ctx, totalCancel = context.WithTimeout(ctx, l.RetryPolicy.TotalTimeout)
		defer totalCancel()
	}
	ctx, runState := newLoopRunState(ctx, l)
	defer close(events)
	defer runState.finish()
	if err := ctx.Err(); err != nil {
		sendLoopCanceled(ctx, events, runState, err)
		return
	}

	executionTools, err := EffectiveTools(l.Tools, l.ToolChoice, l.ToolTransport)
	if err != nil {
		sendLoopError(ctx, events, runState, err)
		return
	}

	var (
		toolDefinitions []ai.ToolDefinition
	)
	if l.ToolTransport == ToolTransportNative {
		toolDefinitions, err = ToolDefinitions(executionTools)
		if err != nil {
			if cancelErr := cancellationError(ctx, err); cancelErr != nil {
				sendLoopCanceled(ctx, events, runState, cancelErr)
				return
			}
			sendLoopError(ctx, events, runState, err)
			return
		}
	}

	_, err = l.PromptBuilder.BuildContext(ctx)
	if err != nil {
		if cancelErr := cancellationError(ctx, err); cancelErr != nil {
			sendLoopCanceled(ctx, events, runState, cancelErr)
			return
		}
		sendLoopError(ctx, events, runState, fmt.Errorf("%w: %w", ErrBuildPrompt, err))
		return
	}
	if err := ctx.Err(); err != nil {
		sendLoopCanceled(ctx, events, runState, err)
		return
	}

	requiredToolCallSatisfied := l.ToolChoice.Mode != ai.ToolChoiceRequired
	// Retain the input until an iteration is accepted. A rejected required-tool
	// response consumes an iteration slot but must not lose conversation input.
	userMessage := userMessageForIteration(l.PromptBuilder, 0)
	for i := range l.MaxLoopIterations {
		iteration := Iteration{Count: i + 1}
		var toolCalls []pendingToolCall
		var deferredTokens []ai.Token
		var iterState *loopIterationState
		var iterationErr error
		var iterCtx context.Context
		var cancel context.CancelFunc
		deferTokens := (l.ToolTransport == ToolTransportText && (!requiredToolCallSatisfied ||
			(l.ToolChoice.Mode == ai.ToolChoiceRequired && len(l.ToolChoice.Names) > 0))) ||
			(l.ToolTransport == ToolTransportNative && len(l.ToolChoice.Names) > 0)

		for attempt := 1; ; attempt++ {
			attemptIteration := Iteration{Count: iteration.Count, UserMessage: userMessage}
			toolCalls = nil
			deferredTokens = nil

			iterCtx, iterState = runState.startIteration(ctx, iteration.Count, attempt)
			iterCtx, cancel = context.WithCancel(iterCtx)
			attemptID := iterState.attemptID()
			if err := iterCtx.Err(); err != nil {
				sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, err)
				cancel()
				iterState.markCanceled(err)
				iterState.finish(nil)
				return
			}
			if err := sendEvent(ctx, events, AttemptStartEvent(iteration.Count, attemptID, runState.retryCount)); err != nil {
				if cancelErr := cancellationError(iterCtx, err); cancelErr != nil {
					sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, cancelErr)
					iterState.markCanceled(cancelErr)
				} else {
					sendAttemptError(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, err)
				}
				cancel()
				iterState.finish(nil)
				return
			}

			request, err := l.buildAttemptRequest(iterCtx, toolDefinitions, requiredToolCallSatisfied)
			if err != nil {
				if cancelErr := cancellationError(iterCtx, err); cancelErr != nil {
					sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, cancelErr)
					cancel()
					iterState.markCanceled(cancelErr)
					iterState.finish(nil)
					return
				}
				iterationErr = fmt.Errorf("%w: %w", ErrBuildPrompt, err)
				sendAttemptError(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, iterationErr)
				cancel()
				iterState.finish(iterationErr)
				return
			}

			modelCtx := iterCtx
			var attemptDeadline context.Context
			if l.RetryPolicy != nil && l.RetryPolicy.AttemptTimeout > 0 {
				var attemptCancel context.CancelFunc
				attemptDeadline, attemptCancel = context.WithTimeout(iterCtx, l.RetryPolicy.AttemptTimeout)
				previousCancel := cancel
				cancel = func() { attemptCancel(); previousCancel() }
				modelCtx = attemptDeadline
			}
			tokens := l.Model.GenerateStream(modelCtx, request)

			retrying := false
			var retryErr error
			for t := range tokens {
				if t.Err != nil {
					retryErr = t.Err
					attemptTimeout := attemptDeadline != nil && errors.Is(attemptDeadline.Err(), context.DeadlineExceeded) && callerCtx.Err() == nil && ctx.Err() == nil
					var providerErr *ai.ProviderError
					if attemptTimeout && errors.Is(t.Err, context.DeadlineExceeded) && !errors.As(t.Err, &providerErr) {
						t.Err = ErrAttemptTimeout
						retryErr = t.Err
					} else if !attemptTimeout {
						if cancelErr := cancellationError(iterCtx, t.Err); cancelErr != nil {
							sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, cancelErr)
							cancel()
							iterState.markCanceled(cancelErr)
							iterState.finish(nil)
							return
						}
					}

					retryLimit := 0
					canRetry := false
					retryable := false
					if l.RetryPolicy != nil {
						retryLimit = l.RetryPolicy.MaxRetries
						retryable = l.RetryPolicy.isRetryable(t.Err)
						canRetry = l.RetryPolicy.hasRetryBudget(runState.retryCount) && retryable
					}
					if canRetry {
						retrying = true
						break
					}

					if l.RetryPolicy == nil || !retryable {
						iterationErr = t.Err
					} else {
						iterationErr = fmt.Errorf("%w: limit=%d: %w", ErrMaxRetries, retryLimit, t.Err)
					}
					sendAttemptError(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, iterationErr)
					cancel()
					iterState.finish(iterationErr)
					return
				}

				if t.Type == ai.TokenTypeToolCall && l.ToolChoice.Mode == ai.ToolChoiceNone {
					// A provider can still emit a tool-call token after tools are
					// disabled. Do not expose or retain a disabled call.
					continue
				}
				if t.Type == ai.TokenTypeToolCall && t.ToolCall != nil {
					attemptIteration.AppendToken(t)

					toolReq := t.ToolCall
					partIdx := len(attemptIteration.Parts) - 1

					toolCalls = append(toolCalls, pendingToolCall{
						partIndex: partIdx,
						call:      *toolReq,
					})
				} else {
					attemptIteration.AppendToken(t)
				}
				if deferTokens {
					deferredTokens = append(deferredTokens, t)
					continue
				}
				runState.recordToken(t)
				iterState.recordToken(t)
				if err := sendEvent(ctx, events, TokenEvent(iteration.Count, attemptID, runState.retryCount, t)); err != nil {
					if cancelErr := cancellationError(iterCtx, err); cancelErr != nil {
						sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, cancelErr)
						iterState.markCanceled(cancelErr)
					} else {
						sendAttemptError(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, err)
					}
					cancel()
					iterState.finish(nil)
					return
				}
			}

			attemptTimeout := attemptDeadline != nil && errors.Is(attemptDeadline.Err(), context.DeadlineExceeded) && callerCtx.Err() == nil && ctx.Err() == nil
			if attemptTimeout && !retrying {
				retryErr = ErrAttemptTimeout
				if l.RetryPolicy.ShouldRetry(runState.retryCount, ErrAttemptTimeout) {
					retrying = true
				} else {
					iterationErr = fmt.Errorf("%w: limit=%d: %w", ErrMaxRetries, l.RetryPolicy.MaxRetries, ErrAttemptTimeout)
					sendAttemptError(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, iterationErr)
					cancel()
					iterState.finish(iterationErr)
					return
				}
			}
			if err := iterCtx.Err(); err != nil && !(retrying && attemptTimeout) {
				sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, err)
				cancel()
				iterState.markCanceled(err)
				iterState.finish(nil)
				return
			}
			if !retrying {
				iteration = attemptIteration
				break
			}

			runState.retry()
			delay := time.Duration(0)
			if l.RetryPolicy != nil {
				delay = l.RetryPolicy.Backoff(runState.retryCount-1, retryErr)
			}
			iterState.recordIteration(attemptIteration)
			iterState.markRetrying(runState.retryCount, retryReason(retryErr), delay)
			if err := sendEvent(ctx, events, RetryEvent(iteration.Count, attemptID, runState.retryCount, retryReason(retryErr), delay, attemptIteration)); err != nil {
				if cancelErr := cancellationError(iterCtx, err); cancelErr != nil {
					sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, cancelErr)
					iterState.markCanceled(cancelErr)
				} else {
					sendAttemptError(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, err)
				}
				cancel()
				iterState.finish(nil)
				return
			}
			if l.RetryPolicy != nil {
				if delay > 0 {
					// The backoff is scoped to the run context, not the failed
					// attempt context. Cancel the attempt before waiting so a model
					// producer cannot remain blocked while the retry is delayed.
					cancel()
					if err := l.RetryPolicy.wait(ctx, delay); err != nil {
						if cancelErr := cancellationError(ctx, err); cancelErr != nil {
							sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, cancelErr)
							cancel()
							iterState.markCanceled(cancelErr)
							iterState.finish(nil)
							return
						}
						iterationErr = fmt.Errorf("wait for retry: %w", err)
						sendAttemptError(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, iterationErr)
						cancel()
						iterState.finish(iterationErr)
						return
					}
				}
			}
			cancel()
			iterState.finish(nil)
		}

		if deferTokens && (!requiredToolCallSatisfied || len(toolCalls) > 0) &&
			!hasPermittedToolCall(toolCalls, l.Tools, l.ToolChoice.Names) {
			// A text-transport response that does not satisfy a required tool
			// call is not part of the conversation and must not be observable.
			cancel()
			if err := sendEvent(ctx, events, DiscardEvent(iteration.Count, iterState.attemptID(), runState.retryCount, iteration)); err != nil {
				iterState.finish(nil)
				return
			}
			runState.resetRetries()
			iterState.finish(iterationErr)
			continue
		}
		if l.ToolChoice.Mode == ai.ToolChoiceNone {
			// Providers and text protocols can still emit a tool-call token even
			// when tools are disabled. Never dispatch such a call.
			toolCalls = nil
		}
		for _, token := range deferredTokens {
			runState.recordToken(token)
			iterState.recordToken(token)
			if err := sendEvent(ctx, events, TokenEvent(iteration.Count, iterState.attemptID(), runState.retryCount, token)); err != nil {
				if cancelErr := cancellationError(iterCtx, err); cancelErr != nil {
					sendAttemptCanceled(ctx, events, runState, iteration.Count, iterState.attemptID(), runState.retryCount, &iteration, cancelErr)
					iterState.markCanceled(cancelErr)
				} else {
					sendAttemptError(ctx, events, runState, iteration.Count, iterState.attemptID(), runState.retryCount, &iteration, err)
				}
				cancel()
				iterState.finish(nil)
				return
			}
		}

		if err := l.executeToolCalls(iterCtx, &iteration, toolCalls, executionTools, events, iteration.Count, iterState.attemptID(), runState.retryCount); err != nil {
			if cancelErr := cancellationError(iterCtx, err); cancelErr != nil {
				sendAttemptCanceled(ctx, events, runState, iteration.Count, iterState.attemptID(), runState.retryCount, &iteration, cancelErr)
				cancel()
				iterState.markCanceled(cancelErr)
				iterState.finish(nil)
				return
			}
			iterationErr = err
			sendAttemptError(ctx, events, runState, iteration.Count, iterState.attemptID(), runState.retryCount, &iteration, iterationErr)
			cancel()
			iterState.finish(iterationErr)
			return
		}
		if err := iterCtx.Err(); err != nil {
			sendAttemptCanceled(ctx, events, runState, iteration.Count, iterState.attemptID(), runState.retryCount, &iteration, err)
			cancel()
			iterState.markCanceled(err)
			iterState.finish(nil)
			return
		}

		iterState.recordToolResponses(iteration)

		attemptID := iterState.attemptID()
		retryCount := runState.retryCount

		if err := sendEvent(ctx, events, IterationDoneEvent(iteration, attemptID, retryCount)); err != nil {
			if cancelErr := cancellationError(iterCtx, err); cancelErr != nil {
				sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, retryCount, &iteration, cancelErr)
				iterState.markCanceled(cancelErr)
			} else {
				sendAttemptError(ctx, events, runState, iteration.Count, attemptID, retryCount, &iteration, err)
			}
			cancel()
			iterState.finish(nil)
			return
		}
		l.Iterations = append(l.Iterations, iteration)
		userMessage = nil
		if hasPermittedToolCall(toolCalls, l.Tools, l.ToolChoice.Names) {
			requiredToolCallSatisfied = true
		}
		runState.resetRetries()
		if l.ToolTransport == ToolTransportText && len(toolCalls) == 0 && !requiredToolCallSatisfied {
			// Text transport cannot rely on provider enforcement. Keep the
			// run active until the required tool call has been observed.
			cancel()
			iterState.finish(iterationErr)
			continue
		}
		if len(toolCalls) == 0 {
			cancel()
			iterState.markFinal()
			iterState.finish(nil)
			if err := sendEvent(ctx, events, DoneEvent()); err != nil {
				if cancelErr := cancellationError(ctx, err); cancelErr != nil {
					sendLoopCanceled(ctx, events, runState, cancelErr)
				}
			}
			return
		}
		cancel()
		iterState.finish(iterationErr)
	}

	if err := ctx.Err(); err != nil {
		sendLoopCanceled(ctx, events, runState, err)
		return
	}
	sendLoopError(ctx, events, runState, fmt.Errorf("%w: limit=%d", ErrMaxIterations, l.MaxLoopIterations))
}

// buildAttemptRequest builds exactly the provider request for one model attempt.
// It deliberately runs before the attempt deadline is installed: prompt building
// belongs to iteration preparation, not model generation.
func (l *Loop) buildAttemptRequest(ctx context.Context, toolDefinitions []ai.ToolDefinition, requiredToolCallSatisfied bool) (ai.AIRequest, error) {
	var (
		prompt         string
		nativeMessages []ai.RequestMessage
		err            error
	)
	if builder, ok := l.PromptBuilder.(gaictx.NativeMessageBuilder); ok {
		prompt, nativeMessages, err = builder.BuildRequest(ctx, l)
	} else {
		prompt, err = l.PromptBuilder.BuildPrompt(ctx, l)
	}
	if err != nil {
		return ai.AIRequest{}, err
	}

	toolChoice := l.ToolChoice
	if toolChoice.Mode == ai.ToolChoiceRequired && requiredToolCallSatisfied {
		toolChoice = ai.ToolChoice{Mode: ai.ToolChoiceAuto}
	}
	if l.ToolTransport == ToolTransportText || len(toolDefinitions) == 0 {
		// Text transport exposes tools through its rendered prompt. A neutral
		// choice is likewise required when native tool definitions are absent.
		toolChoice = ai.ToolChoice{}
	}
	request := renderedPromptRequest(prompt, l.MaxTokens, toolDefinitions, toolChoice, l.ResponseFormat, l.Reasoning)
	request.Messages = nativeMessages
	return request, nil
}

func userMessageForIteration(promptBuilder gaictx.PromptBuilder, index int) *gaictx.Message {
	if index != 0 || promptBuilder == nil {
		return nil
	}
	input := promptBuilder.Input()
	if input.User == nil {
		return nil
	}
	return &gaictx.Message{Role: gaictx.RoleUser, Content: input.User}
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
// failures are stored in ToolResponse.Err and are not returned. Only framework
// or tool-response processing failures are returned.
func (l *Loop) executeToolCalls(ctx context.Context, iteration *Iteration, toolCalls []pendingToolCall, tools []Tool, events chan<- Event, iterationCount, attemptID, retryCount int) error {
	var wg sync.WaitGroup
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

			toolRes, duration := callObservedTool(ctx, tc.call, tools, l.ObservationSink)
			iteration.Parts[tc.partIndex].ToolResp = toolRes
			if l.ToolResponseProcessor != nil {
				if err := l.ToolResponseProcessor.Process(tc.call, toolRes); err != nil {
					processErr := fmt.Errorf("%w: %w", ErrToolResponseProcess, err)
					toolErrMu.Lock()
					if toolErr == nil {
						toolErr = processErr
					}
					toolErrMu.Unlock()
					if events != nil {
						if err := sendEvent(ctx, events, ToolErrorEvent(iterationCount, attemptID, retryCount, tc.call, toolRes, duration, processErr)); err != nil {
							toolErrMu.Lock()
							if toolErr == nil {
								toolErr = err
							}
							toolErrMu.Unlock()
						}
					}
					return
				}
			}
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
func (l *Loop) Messages() []gaictx.Message {
	var msgs []gaictx.Message

	for _, i := range l.Iterations {
		msgs = append(msgs, i.Messages()...)
	}

	return msgs
}
