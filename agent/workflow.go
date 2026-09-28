package agent

import (
	"context"
	"errors"
	"reflect"
	"sync"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

var (
	ErrWorkflowAlreadyRun      = errors.New("workflow has already been run")
	ErrWorkflowNotConfigured   = errors.New("workflow is not configured")
	ErrMiddlewareNotConfigured = errors.New("middleware is not configured")
)

// Stream is the streaming output transformed by middleware. Implementations
// must consume or forward all three channels and close every returned channel.
// Tokens are forwarded in real time; a later retry status can mark an already
// forwarded attempt discardable.
type Stream struct {
	// Tokens contains model text, thoughts, and tool-call tokens.
	Tokens <-chan ai.Token
	// Statuses contains primary-agent iteration updates. AgentMiddleware does not
	// expose iteration updates from its nested agent.
	Statuses <-chan loop.IterationInformation
	// Errors contains primary and middleware failures.
	Errors <-chan error
}

// AgentResult contains canonical accepted output. AttemptedTokens and
// AttemptedText retain the raw real-time stream, including discarded retries.
type AgentResult struct {
	Tokens          []ai.Token
	Text            string
	Reasoning       string
	AttemptedTokens []ai.Token
	AttemptedText   string
	Messages        []gaictx.Message
	Iterations      []loop.Iteration
	// Usage totals accepted iteration usage. BilledUsage includes discarded retries.
	Usage       ai.Usage
	BilledUsage ai.Usage
	Errors      []error
	// Canceled reports that the agent stopped because its context ended.
	Canceled bool
	// CancellationErr contains context.Canceled or context.DeadlineExceeded.
	CancellationErr error
}

// StageResult is the named result produced by agent middleware. Output records
// how the stage affected the visible workflow tokens.
type StageResult struct {
	Name   string
	Output OutputPolicy
	Result AgentResult
}

// WorkflowResult is a snapshot of the complete workflow state. Primary never
// changes, while Tokens, Text, and Reasoning represent canonical output after
// the latest stage. AttemptedTokens and AttemptedText retain raw diagnostics.
type WorkflowResult struct {
	Input           RunInput
	Primary         AgentResult
	Tokens          []ai.Token
	Text            string
	Reasoning       string
	AttemptedTokens []ai.Token
	AttemptedText   string
	// Usage totals accepted iteration usage. BilledUsage includes discarded retries.
	Usage       ai.Usage
	BilledUsage ai.Usage
	Stages      []StageResult
	Errors      []error
	// Canceled reports that the workflow stopped because its context ended.
	Canceled bool
	// CancellationErr contains context.Canceled or context.DeadlineExceeded.
	CancellationErr error
	Complete        bool
}

// MiddlewareContext gives middleware access to the accumulated workflow result.
type MiddlewareContext struct {
	workflow *Workflow
}

// Result returns a safe snapshot of the current workflow state.
func (c *MiddlewareContext) Result() WorkflowResult {
	if c == nil || c.workflow == nil {
		return WorkflowResult{}
	}
	return c.workflow.Result()
}

// Middleware transforms one workflow stream into another. Middleware is applied
// in Definition.Middleware order and may forward, buffer, append, or replace any
// part of the upstream stream.
type Middleware interface {
	Process(ctx context.Context, run *MiddlewareContext, upstream Stream) Stream
}

// MiddlewareFunc adapts a function into Middleware.
type MiddlewareFunc func(ctx context.Context, run *MiddlewareContext, upstream Stream) Stream

func (f MiddlewareFunc) Process(ctx context.Context, run *MiddlewareContext, upstream Stream) Stream {
	return f(ctx, run, upstream)
}

type middlewareValidator interface {
	validate() error
}

// Workflow runs a configured agent loop and its stream middleware. A Workflow
// is single-use; create another with Agent.NewRun for a subsequent invocation.
type Workflow struct {
	loop       *loop.Loop
	middleware []Middleware
	name       string
	debug      gai.ObservationSink

	mu      sync.RWMutex
	started bool
	result  WorkflowResult
}

func newWorkflow(input RunInput, l *loop.Loop, name string, debug gai.ObservationSink, middleware []Middleware) *Workflow {
	input = cloneRunInput(input)
	return &Workflow{
		loop:       l,
		middleware: append([]Middleware(nil), middleware...),
		name:       name,
		debug:      debug,
		result:     WorkflowResult{Input: input},
	}
}

func validateMiddleware(middleware []Middleware) error {
	for _, item := range middleware {
		if middlewareIsNil(item) {
			return ErrMiddlewareNotConfigured
		}
		if validator, ok := item.(middlewareValidator); ok {
			if err := validator.validate(); err != nil {
				return err
			}
		}
	}
	return nil
}

func middlewareIsNil(item Middleware) bool {
	if item == nil {
		return true
	}

	value := reflect.ValueOf(item)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

// RunEvents starts the workflow and returns its ordered primary-agent event stream.
// The stream is the preferred API for consumers that need causal ordering between
// tokens, retries, completed iterations, errors, and cancellation. It can be
// called only once. Middleware remains available through Run because the legacy
// three-channel Middleware interface cannot preserve a total event order.
func (w *Workflow) RunEvents(ctx context.Context) <-chan loop.Event {
	if err := w.begin(); err != nil {
		return failedEventStream(err)
	}
	ctx = applyWorkflowTraceContext(ctx, w)
	ctx, runObs := newAgentRunObserver(ctx, w)
	ctx, obs := newWorkflowObserver(ctx, w)
	obs.Started(ctx)
	return w.captureEvents(ctx, w.loop.Run(ctx), obs, runObs)
}

// Run starts the workflow and returns the final transformed stream. Deprecated:
// prefer RunEvents for new consumers. Callers must consume all three channels.
// It can be called only once.
func (w *Workflow) Run(ctx context.Context) (<-chan ai.Token, <-chan loop.IterationInformation, <-chan error) {
	if err := w.begin(); err != nil {
		return failedStream(err)
	}
	ctx = applyWorkflowTraceContext(ctx, w)
	ctx, runObs := newAgentRunObserver(ctx, w)
	ctx, obs := newWorkflowObserver(ctx, w)
	events := w.loop.Run(ctx)

	obs.Started(ctx)
	stream := w.capturePrimary(ctx, loopEventsToStream(ctx, events), obs)
	run := &MiddlewareContext{workflow: w}
	for _, middleware := range w.middleware {
		stream = middleware.Process(ctx, run, stream)
	}
	stream = w.captureFinal(ctx, stream, obs, runObs)
	return stream.Tokens, stream.Statuses, stream.Errors
}

func applyWorkflowTraceContext(ctx context.Context, workflow *Workflow) context.Context {
	if workflow == nil {
		return ctx
	}
	if workflow.result.Input.TraceContext != nil {
		ctx = gai.WithTraceContext(ctx, *workflow.result.Input.TraceContext)
	}
	return gai.WithObservationRunID(ctx, workflow.result.Input.ID)
}

func loopEventsToStream(ctx context.Context, events <-chan loop.Event) Stream {
	tokens := make(chan ai.Token, 16)
	statuses := make(chan loop.IterationInformation, 16)
	errs := make(chan error, 1)

	go func() {
		defer close(tokens)
		defer close(statuses)
		defer close(errs)

		for event := range events {
			switch event.Type {
			case loop.EventToken:
				if event.Token != nil {
					send(ctx, tokens, *event.Token)
				}
			case loop.EventRetry:
				status := loop.IterationInformation{
					IterationCount:   event.IterationCount,
					AttemptID:        event.AttemptID,
					RetryCount:       event.RetryCount,
					PartCount:        event.PartCount,
					Retrying:         true,
					DiscardIteration: true,
				}
				if event.Iteration != nil {
					status.Iteration = *event.Iteration
				}
				send(ctx, statuses, status)
			case loop.EventDiscard:
				status := loop.IterationInformation{
					IterationCount:   event.IterationCount,
					AttemptID:        event.AttemptID,
					RetryCount:       event.RetryCount,
					PartCount:        event.PartCount,
					DiscardIteration: true,
				}
				if event.Iteration != nil {
					status.Iteration = *event.Iteration
				}
				send(ctx, statuses, status)
			case loop.EventIterationDone:
				status := loop.IterationInformation{
					IterationCount: event.IterationCount,
					AttemptID:      event.AttemptID,
					RetryCount:     event.RetryCount,
					PartCount:      event.PartCount,
				}
				if event.Iteration != nil {
					status.Iteration = *event.Iteration
				}
				send(ctx, statuses, status)
			case loop.EventError:
				if event.Iteration != nil {
					status := loop.IterationInformation{
						Iteration:        *event.Iteration,
						IterationCount:   event.IterationCount,
						AttemptID:        event.AttemptID,
						RetryCount:       event.RetryCount,
						PartCount:        event.PartCount,
						DiscardIteration: true,
					}
					send(ctx, statuses, status)
				}
				if event.Err != nil {
					send(ctx, errs, event.Err)
				}
			case loop.EventCanceled:
				status := loop.IterationInformation{
					IterationCount:   event.IterationCount,
					AttemptID:        event.AttemptID,
					RetryCount:       event.RetryCount,
					PartCount:        event.PartCount,
					DiscardIteration: true,
					Canceled:         true,
					CancellationErr:  event.Err,
				}
				if event.Iteration != nil {
					status.Iteration = *event.Iteration
				}
				sendTerminalStatus(statuses, status)
			}
		}
	}()

	return Stream{Tokens: tokens, Statuses: statuses, Errors: errs}
}

// Result returns a concurrency-safe snapshot. Complete is true after all three
// channels returned by Run have closed.
func (w *Workflow) Result() WorkflowResult {
	if w == nil {
		return WorkflowResult{}
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	return cloneWorkflowResult(w.result)
}

func (w *Workflow) capturePrimary(ctx context.Context, upstream Stream, obs *workflowObserver) Stream {
	return captureStream(ctx, upstream, func(captured capturedStream) {
		canceled, cancellationErr := captured.cancellation()
		tokens := iterationTokens(w.loop.Iterations)
		result := AgentResult{
			Tokens:          tokens,
			Text:            tokenText(tokens),
			Reasoning:       tokenReasoning(tokens),
			AttemptedTokens: cloneTokens(captured.Tokens),
			AttemptedText:   tokenText(captured.Tokens),
			Messages:        cloneMessages(w.loop.Messages()),
			Iterations:      cloneIterations(w.loop.Iterations),
			Usage:           iterationUsage(w.loop.Iterations),
			BilledUsage:     statusUsage(captured.Statuses),
			Errors:          append([]error(nil), captured.Errors...),
			Canceled:        canceled,
			CancellationErr: cancellationErr,
		}
		w.mu.Lock()
		w.result.Primary = result
		w.result.Tokens = cloneTokens(result.Tokens)
		w.result.Text = result.Text
		w.result.Reasoning = result.Reasoning
		w.result.AttemptedTokens = cloneTokens(captured.Tokens)
		w.result.AttemptedText = tokenText(captured.Tokens)
		w.result.Usage = result.Usage
		w.result.BilledUsage = result.BilledUsage
		w.result.Errors = append([]error(nil), captured.Errors...)
		w.result.Canceled = canceled
		w.result.CancellationErr = cancellationErr
		w.mu.Unlock()
		obs.PrimaryFinished(ctx, result)
	})
}

func (w *Workflow) captureFinal(ctx context.Context, upstream Stream, obs *workflowObserver, runObs *agentRunObserver) Stream {
	return captureStream(ctx, upstream, func(captured capturedStream) {
		canceled, cancellationErr := captured.cancellation()
		w.mu.Lock()
		w.result.AttemptedTokens = cloneTokens(captured.Tokens)
		w.result.AttemptedText = tokenText(captured.Tokens)
		w.result.Errors = append([]error(nil), captured.Errors...)
		if canceled {
			w.result.Canceled = true
			w.result.CancellationErr = cancellationErr
		}
		w.result.Complete = true
		result := cloneWorkflowResult(w.result)
		w.mu.Unlock()
		obs.Finished(ctx, result)
		runObs.Finished(result)
	})
}

func (w *Workflow) captureEvents(ctx context.Context, upstream <-chan loop.Event, obs *workflowObserver, runObs *agentRunObserver) <-chan loop.Event {
	events := make(chan loop.Event, 32)
	go func() {
		defer close(events)

		var tokens []ai.Token
		var errs []error
		var canceled bool
		var cancellationErr error
		var billedUsage ai.Usage
		for event := range upstream {
			switch event.Type {
			case loop.EventToken:
				if event.Token != nil {
					tokens = append(tokens, *event.Token)
				}
			case loop.EventError:
				if event.Iteration != nil {
					billedUsage.Add(event.Iteration.Usage)
				}
				if event.Err != nil {
					errs = append(errs, event.Err)
				}
			case loop.EventCanceled:
				if event.Iteration != nil {
					billedUsage.Add(event.Iteration.Usage)
				}
				canceled = true
				cancellationErr = event.Err
			case loop.EventRetry, loop.EventDiscard, loop.EventIterationDone:
				if event.Iteration != nil {
					billedUsage.Add(event.Iteration.Usage)
				}
			}
			events <- event
		}

		primary := AgentResult{
			Tokens:          cloneTokens(tokens),
			Text:            tokenText(tokens),
			Reasoning:       tokenReasoning(tokens),
			Messages:        cloneMessages(w.loop.Messages()),
			Iterations:      cloneIterations(w.loop.Iterations),
			Usage:           iterationUsage(w.loop.Iterations),
			BilledUsage:     billedUsage,
			Errors:          append([]error(nil), errs...),
			Canceled:        canceled,
			CancellationErr: cancellationErr,
		}
		w.mu.Lock()
		w.result.Primary = primary
		w.result.Tokens = cloneTokens(tokens)
		w.result.Text = primary.Text
		w.result.Reasoning = primary.Reasoning
		w.result.Usage = primary.Usage
		w.result.BilledUsage = primary.BilledUsage
		w.result.Errors = append([]error(nil), errs...)
		w.result.Canceled = canceled
		w.result.CancellationErr = cancellationErr
		w.result.Complete = true
		result := cloneWorkflowResult(w.result)
		w.mu.Unlock()

		obs.PrimaryFinished(ctx, primary)
		obs.Finished(ctx, result)
		runObs.Finished(result)
	}()
	return events
}

func (w *Workflow) addStage(stage StageResult, tokens []ai.Token) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.result.Stages = append(w.result.Stages, cloneStageResult(stage))
	w.result.Tokens = cloneTokens(tokens)
	w.result.Text = tokenText(tokens)
	w.result.Reasoning = tokenReasoning(tokens)
}

type streamRelay struct {
	Tokens   chan<- ai.Token
	Statuses chan<- loop.IterationInformation
	Errors   chan<- error
}

type capturedStream struct {
	Tokens   []ai.Token
	Statuses []loop.IterationInformation
	Errors   []error
}

func (s capturedStream) cancellation() (bool, error) {
	for _, status := range s.Statuses {
		if status.Canceled {
			return true, status.CancellationErr
		}
	}
	return false, nil
}

func drainStream(ctx context.Context, upstream Stream, relay streamRelay) capturedStream {
	var captured capturedStream
	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		for token := range upstream.Tokens {
			captured.Tokens = append(captured.Tokens, token)
			if relay.Tokens != nil {
				send(ctx, relay.Tokens, token)
			}
		}
	}()
	go func() {
		defer wg.Done()
		for status := range upstream.Statuses {
			captured.Statuses = append(captured.Statuses, status)
			if relay.Statuses != nil {
				if status.Canceled {
					sendTerminalStatus(relay.Statuses, status)
				} else {
					send(ctx, relay.Statuses, status)
				}
			}
		}
	}()
	go func() {
		defer wg.Done()
		for err := range upstream.Errors {
			if err == nil {
				continue
			}
			captured.Errors = append(captured.Errors, err)
			if relay.Errors != nil {
				send(ctx, relay.Errors, err)
			}
		}
	}()
	wg.Wait()
	return captured
}

func captureStream(ctx context.Context, upstream Stream, completed func(capturedStream)) Stream {
	tokens := make(chan ai.Token, 16)
	statuses := make(chan loop.IterationInformation, 16)
	errs := make(chan error, 1)

	go func() {
		captured := drainStream(ctx, upstream, streamRelay{
			Tokens:   tokens,
			Statuses: statuses,
			Errors:   errs,
		})
		completed(captured)
		close(tokens)
		close(statuses)
		close(errs)
	}()

	return Stream{Tokens: tokens, Statuses: statuses, Errors: errs}
}

func (w *Workflow) begin() error {
	if w == nil || w.loop == nil {
		return ErrWorkflowNotConfigured
	}

	w.mu.Lock()
	defer w.mu.Unlock()
	if w.started {
		return ErrWorkflowAlreadyRun
	}
	w.started = true
	return nil
}

func failedEventStream(err error) <-chan loop.Event {
	events := make(chan loop.Event, 1)
	events <- loop.ErrorEvent(err)
	close(events)
	return events
}

func failedStream(err error) (<-chan ai.Token, <-chan loop.IterationInformation, <-chan error) {
	tokens := make(chan ai.Token)
	statuses := make(chan loop.IterationInformation)
	errs := make(chan error, 1)
	close(tokens)
	close(statuses)
	errs <- err
	close(errs)
	return tokens, statuses, errs
}

func send[T any](ctx context.Context, ch chan<- T, value T) {
	select {
	case ch <- value:
	case <-ctx.Done():
	}
}

func sendTerminalStatus(ch chan<- loop.IterationInformation, status loop.IterationInformation) {
	select {
	case ch <- status:
	default:
	}
}

func tokenText(tokens []ai.Token) string {
	var text []byte
	for _, token := range tokens {
		if token.Type != ai.TokenTypeText {
			continue
		}
		if token.Text != "" {
			text = append(text, token.Text...)
		} else {
			text = append(text, token.Data...)
		}
	}
	return string(text)
}

func tokenReasoning(tokens []ai.Token) string {
	var reasoning []byte
	for _, token := range tokens {
		if token.Type != ai.TokenTypeThought {
			continue
		}
		if token.Text != "" {
			reasoning = append(reasoning, token.Text...)
		} else {
			reasoning = append(reasoning, token.Data...)
		}
	}
	return string(reasoning)
}

func iterationTokens(iterations []loop.Iteration) []ai.Token {
	var tokens []ai.Token
	for _, iteration := range iterations {
		for _, part := range iteration.Parts {
			if part.Response != nil {
				if part.Response.Reasoning != "" {
					tokens = append(tokens, ai.Token{Type: ai.TokenTypeThought, Text: part.Response.Reasoning})
				}
				if part.Response.Text != "" {
					tokens = append(tokens, ai.Token{Type: ai.TokenTypeText, Text: part.Response.Text})
				}
			}
			if part.ToolReq != nil {
				tokens = append(tokens, ai.Token{Type: ai.TokenTypeToolCall, ToolCall: cloneToolCall(part.ToolReq)})
			}
		}
	}
	return tokens
}

func cloneRunInput(input RunInput) RunInput {
	cloned := input
	if input.TraceContext != nil {
		traceContext := *input.TraceContext
		traceContext.Tags = append([]string(nil), input.TraceContext.Tags...)
		if input.TraceContext.Metadata != nil {
			traceContext.Metadata = make(map[string]string, len(input.TraceContext.Metadata))
			for key, value := range input.TraceContext.Metadata {
				traceContext.Metadata[key] = value
			}
		}
		cloned.TraceContext = &traceContext
	}
	cloned.Prompt = input.Prompt.Clone()
	cloned.Execution = cloneExecution(input.Execution)
	if input.Meta != nil {
		cloned.Meta = make(map[string]any, len(input.Meta))
		for key, value := range input.Meta {
			cloned.Meta[key] = value
		}
	}
	return cloned
}

func cloneResponseFormat(format ai.ResponseFormat) ai.ResponseFormat {
	cloned := format
	cloned.Schema = append([]byte(nil), format.Schema...)
	return cloned
}

func cloneTokens(tokens []ai.Token) []ai.Token {
	cloned := make([]ai.Token, len(tokens))
	for i, token := range tokens {
		cloned[i] = token
		cloned[i].Data = append([]byte(nil), token.Data...)
		cloned[i].ToolCall = cloneToolCall(token.ToolCall)
	}
	return cloned
}

func cloneToolCall(call *ai.ToolCall) *ai.ToolCall {
	if call == nil {
		return nil
	}
	cloned := *call
	cloned.Args = append([]byte(nil), call.Args...)
	return &cloned
}

func cloneMessages(messages []gaictx.Message) []gaictx.Message {
	cloned := make([]gaictx.Message, len(messages))
	for i, message := range messages {
		cloned[i] = message
		if message.TokenCount != nil {
			cloned[i].TokenCount = make(map[string]int, len(message.TokenCount))
			for tokenizer, count := range message.TokenCount {
				cloned[i].TokenCount[tokenizer] = count
			}
		}
	}
	return cloned
}

func statusUsage(statuses []loop.IterationInformation) ai.Usage {
	var usage ai.Usage
	for _, status := range statuses {
		usage.Add(status.Iteration.Usage)
	}
	return usage
}

func iterationUsage(iterations []loop.Iteration) ai.Usage {
	var usage ai.Usage
	for _, iteration := range iterations {
		usage.Add(iteration.Usage)
	}
	return usage
}

func cloneIterations(iterations []loop.Iteration) []loop.Iteration {
	cloned := make([]loop.Iteration, len(iterations))
	for i, iteration := range iterations {
		cloned[i] = iteration
		if iteration.UserMessage != nil {
			message := cloneMessages([]gaictx.Message{*iteration.UserMessage})[0]
			cloned[i].UserMessage = &message
		}
		cloned[i].Parts = make([]loop.IterationPart, len(iteration.Parts))
		for j, part := range iteration.Parts {
			cloned[i].Parts[j] = part
			if part.Response != nil {
				response := *part.Response
				cloned[i].Parts[j].Response = &response
			}
			cloned[i].Parts[j].ToolReq = cloneToolCall(part.ToolReq)
			cloned[i].Parts[j].ToolResp = cloneToolResponse(part.ToolResp)
		}
	}
	return cloned
}

func cloneToolResponse(response *loop.ToolResponse) *loop.ToolResponse {
	if response == nil {
		return nil
	}
	cloned := *response
	if response.Text != nil {
		text := *response.Text
		cloned.Text = &text
	}
	if response.Err != nil {
		err := *response.Err
		cloned.Err = &err
	}
	return &cloned
}

func cloneAgentResult(result AgentResult) AgentResult {
	result.Tokens = cloneTokens(result.Tokens)
	result.AttemptedTokens = cloneTokens(result.AttemptedTokens)
	result.Messages = cloneMessages(result.Messages)
	result.Iterations = cloneIterations(result.Iterations)
	result.Errors = append([]error(nil), result.Errors...)
	return result
}

func cloneStageResult(stage StageResult) StageResult {
	stage.Result = cloneAgentResult(stage.Result)
	return stage
}

func cloneWorkflowResult(result WorkflowResult) WorkflowResult {
	result.Input = cloneRunInput(result.Input)
	result.Primary = cloneAgentResult(result.Primary)
	result.Tokens = cloneTokens(result.Tokens)
	result.AttemptedTokens = cloneTokens(result.AttemptedTokens)
	result.Errors = append([]error(nil), result.Errors...)
	result.Stages = append([]StageResult(nil), result.Stages...)
	for i := range result.Stages {
		result.Stages[i] = cloneStageResult(result.Stages[i])
	}
	return result
}
