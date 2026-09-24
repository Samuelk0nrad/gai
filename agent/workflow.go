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
	ErrWorkflowNotStarted      = errors.New("workflow has not been started")
	ErrWorkflowNotConfigured   = errors.New("workflow is not configured")
	ErrMiddlewareNotConfigured = errors.New("middleware is not configured")
)

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
	Usage           ai.Usage
	BilledUsage     ai.Usage
	Errors          []error
	Canceled        bool
	CancellationErr error
}

// StageResult is the named result produced by agent middleware.
type StageResult struct {
	Name   string
	Output OutputPolicy
	Result AgentResult
}

// WorkflowResult is a concurrency-safe snapshot of complete workflow state.
type WorkflowResult struct {
	Input RunInput

	// Output is the canonical visible workflow output after middleware.
	Output []OutputPart
	// Tokens is retained as a convenience view of text and reasoning output.
	Tokens          []ai.Token
	Text            string
	Reasoning       string
	AttemptedTokens []ai.Token
	AttemptedText   string

	Primary AgentResult
	Stages  []StageResult

	Usage           ai.Usage
	BilledUsage     ai.Usage
	Errors          []error
	Canceled        bool
	CancellationErr error
	Complete        bool
}

// MiddlewareContext gives middleware access to the accumulated workflow result
// and its source for newly-created events.
type MiddlewareContext struct {
	workflow *Workflow
	source   EventSource
}

func (c *MiddlewareContext) Result() WorkflowResult {
	if c == nil || c.workflow == nil {
		return WorkflowResult{}
	}
	return c.workflow.Result()
}

// Source returns this middleware's execution-local event source.
func (c *MiddlewareContext) Source() EventSource {
	if c == nil {
		return EventSource{}
	}
	return c.source
}

// Middleware transforms one ordered workflow event stream into another.
// Implementations consume upstream, preserve causal order for forwarded
// non-output events, close their returned channel, and never emit a terminal
// workflow event.
type Middleware interface {
	Process(context.Context, *MiddlewareContext, <-chan Event) <-chan Event
}

// MiddlewareFunc adapts a function into Middleware.
type MiddlewareFunc func(context.Context, *MiddlewareContext, <-chan Event) <-chan Event

func (f MiddlewareFunc) Process(ctx context.Context, run *MiddlewareContext, upstream <-chan Event) <-chan Event {
	return f(ctx, run, upstream)
}

type middlewareValidator interface {
	validate() error
}

// Workflow is the single-use execution handle returned by Agent.NewRun.
type Workflow struct {
	loop       *loop.Loop
	middleware []Middleware
	name       string
	debug      gai.ObservationSink

	mu          sync.RWMutex
	started     bool
	done        chan struct{}
	primaryDone chan struct{}
	terminalErr error
	result      WorkflowResult
}

func newWorkflow(input RunInput, l *loop.Loop, name string, debug gai.ObservationSink, middleware []Middleware) *Workflow {
	return &Workflow{
		loop:        l,
		middleware:  append([]Middleware(nil), middleware...),
		name:        name,
		debug:       debug,
		done:        make(chan struct{}),
		primaryDone: make(chan struct{}),
		result:      WorkflowResult{Input: cloneRunInput(input)},
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

// Run starts the canonical workflow pipeline, drains it, and returns its final
// result and terminal error.
func (w *Workflow) Run(ctx context.Context) (WorkflowResult, error) {
	events, err := w.start(ctx)
	if err != nil {
		return w.Result(), err
	}
	for range events {
	}
	return w.Wait()
}

// RunEvents starts the same ordered middleware-aware pipeline used by Run.
// Callers must drain the returned channel before Wait can complete. On
// cancellation, an abandoned stream retains its terminal event in place of any
// pending non-terminal event so completion does not depend on a receiver.
func (w *Workflow) RunEvents(ctx context.Context) <-chan Event {
	events, err := w.start(ctx)
	if err != nil {
		return failedEventStream(err)
	}
	return events
}

// Wait waits for an already-started workflow without consuming its event
// stream. It is safe for repeated and concurrent calls.
func (w *Workflow) Wait() (WorkflowResult, error) {
	if w == nil {
		return WorkflowResult{}, ErrWorkflowNotConfigured
	}
	w.mu.RLock()
	started := w.started
	done := w.done
	w.mu.RUnlock()
	if !started {
		return w.Result(), ErrWorkflowNotStarted
	}
	<-done
	w.mu.RLock()
	err := w.terminalErr
	w.mu.RUnlock()
	return w.Result(), err
}

// Result returns a non-blocking, defensively cloned snapshot.
func (w *Workflow) Result() WorkflowResult {
	if w == nil {
		return WorkflowResult{}
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	return cloneWorkflowResult(w.result)
}

func (w *Workflow) start(ctx context.Context) (<-chan Event, error) {
	if err := w.begin(); err != nil {
		return nil, err
	}
	ctx = applyWorkflowTraceContext(ctx, w)
	ctx, runObs := newAgentRunObserver(ctx, w)
	ctx, obs := newWorkflowObserver(ctx, w)
	obs.Started(ctx)

	stream := w.mapPrimary(ctx, w.loop.Run(ctx), obs)
	for index, middleware := range w.middleware {
		source := EventSource{Kind: SourceMiddleware, Index: index + 1, Name: middlewareName(middleware, index)}
		run := &MiddlewareContext{workflow: w, source: source}
		stream = middleware.Process(ctx, run, stream)
		stream = w.captureMiddlewareOutput(ctx, stream)
	}
	return w.finalize(ctx, stream, obs, runObs), nil
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

func middlewareName(middleware Middleware, index int) string {
	if named, ok := middleware.(*AgentMiddleware); ok {
		return named.name()
	}
	return "middleware"
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

func failedEventStream(err error) <-chan Event {
	events := make(chan Event, 1)
	events <- Event{Type: EventError, Source: EventSource{Kind: SourceWorkflow}, Err: err}
	close(events)
	return events
}

func sendWorkflowEvent(ctx context.Context, out chan<- Event, event Event, deliver bool) bool {
	if !deliver {
		return false
	}
	// Prefer a receiver that is already waiting, even when cancellation and the
	// send become ready together. This keeps internal pipeline stages draining
	// while allowing an abandoned public stream to be released by cancellation.
	select {
	case out <- event:
		return true
	default:
	}
	select {
	case out <- event:
		return true
	case <-ctx.Done():
		return false
	}
}

func sendTerminalWorkflowEvent(ctx context.Context, out chan Event, event Event, deliver bool) {
	if sendWorkflowEvent(ctx, out, event, deliver) || ctx.Err() == nil {
		return
	}
	// The finalizer owns out and gives it one bounded slot. Cancellation may
	// evict one pending non-terminal event, but the terminal outcome is retained
	// for a delayed consumer without letting an abandoned stream block Wait.
	select {
	case <-out:
	default:
	}
	out <- event
}

type attemptKey struct {
	source    int
	iteration int
	attempt   int
}

func eventAttemptKey(event Event) attemptKey {
	return attemptKey{source: event.Source.Index, iteration: event.IterationCount, attempt: event.AttemptID}
}

type primaryAccumulator struct {
	attempted []ai.Token
	accepted  []loop.Iteration
	billed    ai.Usage
	billedKey map[attemptKey]struct{}
	errs      []error
	canceled  bool
	cancelErr error
}

func (a *primaryAccumulator) account(event Event) {
	if event.Iteration == nil {
		return
	}
	key := eventAttemptKey(event)
	if _, exists := a.billedKey[key]; exists {
		return
	}
	a.billedKey[key] = struct{}{}
	a.billed.Add(event.Iteration.Usage)
}

func (a *primaryAccumulator) result() AgentResult {
	tokens := iterationTokens(a.accepted)
	var messages []gaictx.Message
	for _, iteration := range a.accepted {
		messages = append(messages, iteration.Messages()...)
	}
	return AgentResult{
		Tokens:          tokens,
		Text:            tokenText(tokens),
		Reasoning:       tokenReasoning(tokens),
		AttemptedTokens: cloneTokens(a.attempted),
		AttemptedText:   tokenText(a.attempted),
		Messages:        cloneMessages(messages),
		Iterations:      cloneIterations(a.accepted),
		Usage:           iterationUsage(a.accepted),
		BilledUsage:     a.billed,
		Errors:          append([]error(nil), a.errs...),
		Canceled:        a.canceled,
		CancellationErr: a.cancelErr,
	}
}

func (w *Workflow) mapPrimary(ctx context.Context, upstream <-chan loop.Event, obs *workflowObserver) <-chan Event {
	out := make(chan Event)
	source := EventSource{Kind: SourcePrimary, Index: 0, Name: w.name}
	go func() {
		defer close(out)
		defer close(w.primaryDone)
		acc := primaryAccumulator{billedKey: make(map[attemptKey]struct{})}
		deliver := sendWorkflowEvent(ctx, out, Event{Type: EventStageStart, Source: source}, true)
		var terminal Event
		for low := range upstream {
			event, emit := mapLoopEvent(low, source)
			if low.Type == loop.EventToken && low.Token != nil {
				acc.attempted = append(acc.attempted, cloneTokens([]ai.Token{*low.Token})[0])
			}
			switch low.Type {
			case loop.EventRetry, loop.EventDiscard:
				acc.account(event)
			case loop.EventIterationDone:
				acc.account(event)
				if low.Iteration != nil {
					acc.accepted = append(acc.accepted, cloneIterations([]loop.Iteration{*low.Iteration})[0])
				}
			case loop.EventError:
				if low.Err != nil {
					acc.errs = append(acc.errs, low.Err)
				}
				terminal = Event{Type: EventStageFinish, Source: source, IterationCount: low.IterationCount, AttemptID: low.AttemptID, RetryCount: low.RetryCount, PartCount: low.PartCount, Iteration: cloneIterationPtr(low.Iteration), StageOutcome: StageFailed, Err: low.Err}
				acc.account(terminal)
			case loop.EventCanceled:
				acc.canceled = true
				acc.cancelErr = low.Err
				terminal = Event{Type: EventStageFinish, Source: source, IterationCount: low.IterationCount, AttemptID: low.AttemptID, RetryCount: low.RetryCount, PartCount: low.PartCount, Iteration: cloneIterationPtr(low.Iteration), StageOutcome: StageCanceled, Err: low.Err}
				acc.account(terminal)
			case loop.EventDone:
				terminal = Event{Type: EventStageFinish, Source: source, StageOutcome: StageSucceeded}
			}
			if emit {
				deliver = sendWorkflowEvent(ctx, out, cloneEvent(event), deliver)
			}
		}
		if terminal.Type == "" {
			terminal = Event{Type: EventStageFinish, Source: source, StageOutcome: StageFailed, Err: errors.New("loop stream closed without a terminal event")}
			acc.errs = append(acc.errs, terminal.Err)
		}
		primary := acc.result()
		w.mu.Lock()
		w.result.Primary = cloneAgentResult(primary)
		w.result.Usage = primary.Usage
		w.result.BilledUsage = primary.BilledUsage
		w.result.Errors = append([]error(nil), primary.Errors...)
		w.result.Canceled = primary.Canceled
		w.result.CancellationErr = primary.CancellationErr
		w.setVisibleOutputLocked(outputPartsFromTokens(primary.Tokens))
		w.result.AttemptedTokens = cloneTokens(primary.AttemptedTokens)
		w.result.AttemptedText = primary.AttemptedText
		w.mu.Unlock()
		obs.PrimaryFinished(ctx, primary)
		sendWorkflowEvent(ctx, out, cloneEvent(terminal), deliver)
	}()
	return out
}

func mapLoopEvent(low loop.Event, source EventSource) (Event, bool) {
	event := Event{
		Source:         source,
		IterationCount: low.IterationCount,
		AttemptID:      low.AttemptID,
		RetryCount:     low.RetryCount,
		PartCount:      low.PartCount,
		Iteration:      cloneIterationPtr(low.Iteration),
		ToolCall:       cloneToolCall(low.ToolCall),
		ToolResponse:   cloneToolResponse(low.ToolResponse),
		RetryReason:    low.RetryReason,
		RetryDelay:     low.RetryDelay,
		Duration:       low.Duration,
		Err:            low.Err,
	}
	switch low.Type {
	case loop.EventAttemptStart:
		event.Type = EventAttemptStart
	case loop.EventToken:
		if low.Token == nil {
			return Event{}, false
		}
		text := low.Token.Text
		if text == "" {
			text = string(low.Token.Data)
		}
		switch low.Token.Type {
		case ai.TokenTypeText:
			event.Type = EventOutput
			event.Output = &OutputPart{Kind: OutputText, Text: text}
		case ai.TokenTypeThought:
			event.Type = EventOutput
			event.Output = &OutputPart{Kind: OutputReasoning, Text: text}
		default:
			return Event{}, false
		}
	case loop.EventRetry:
		event.Type = EventRetry
	case loop.EventDiscard:
		event.Type = EventDiscard
	case loop.EventIterationDone:
		event.Type = EventIterationDone
	case loop.EventToolStart:
		event.Type = EventToolStart
	case loop.EventToolResult:
		event.Type = EventToolResult
	case loop.EventToolError:
		event.Type = EventToolError
	default:
		return Event{}, false
	}
	return event, true
}

func cloneIterationPtr(iteration *loop.Iteration) *loop.Iteration {
	if iteration == nil {
		return nil
	}
	cloned := cloneIterations([]loop.Iteration{*iteration})[0]
	return &cloned
}

type outputRecord struct {
	key  attemptKey
	part OutputPart
}

type workflowOutputAccumulator struct {
	outputs         []outputRecord
	invalid         map[attemptKey]struct{}
	stageErrs       []error
	canceled        bool
	cancellationErr error
}

func (a *workflowOutputAccumulator) invalidate(key attemptKey) {
	if a.invalid == nil {
		a.invalid = make(map[attemptKey]struct{})
	}
	a.invalid[key] = struct{}{}
}

func (a *workflowOutputAccumulator) add(event Event) {
	switch event.Type {
	case EventOutput:
		if event.Output != nil {
			a.outputs = append(a.outputs, outputRecord{
				key:  eventAttemptKey(event),
				part: cloneOutputPart(*event.Output),
			})
		}
	case EventRetry, EventDiscard:
		a.invalidate(eventAttemptKey(event))
	case EventStageFinish:
		if event.StageOutcome != StageFailed && event.StageOutcome != StageCanceled {
			return
		}
		if event.AttemptID != 0 {
			a.invalidate(eventAttemptKey(event))
		}
		if event.StageOutcome == StageFailed && event.Err != nil && event.StageReason != stageReasonRecorded {
			a.stageErrs = append(a.stageErrs, event.Err)
		}
		if event.StageOutcome == StageCanceled && !a.canceled {
			a.canceled = true
			a.cancellationErr = event.Err
		}
	}
}

func (a *workflowOutputAccumulator) result() (accepted []OutputPart, attempted []OutputPart, stageErrs []error, canceled bool, cancellationErr error) {
	for _, output := range a.outputs {
		attempted = append(attempted, cloneOutputPart(output.part))
		if _, discarded := a.invalid[output.key]; !discarded {
			accepted = append(accepted, cloneOutputPart(output.part))
		}
	}
	return accepted, attempted, append([]error(nil), a.stageErrs...), a.canceled, a.cancellationErr
}

func (w *Workflow) captureMiddlewareOutput(ctx context.Context, upstream <-chan Event) <-chan Event {
	out := make(chan Event)
	go func() {
		defer close(out)
		var accumulator workflowOutputAccumulator
		deliver := true
		for event := range upstream {
			accumulator.add(event)
			deliver = sendWorkflowEvent(ctx, out, cloneEvent(event), deliver)
		}
		output, _, stageErrs, canceled, cancellationErr := accumulator.result()
		w.mu.Lock()
		w.setVisibleOutputLocked(output)
		for _, err := range stageErrs {
			if err != nil && !containsError(w.result.Errors, err) {
				w.result.Errors = append(w.result.Errors, err)
			}
		}
		if canceled {
			w.result.Canceled = true
			w.result.CancellationErr = cancellationErr
		}
		w.mu.Unlock()
	}()
	return out
}

func (w *Workflow) finalize(ctx context.Context, upstream <-chan Event, obs *workflowObserver, runObs *agentRunObserver) <-chan Event {
	out := make(chan Event, 1)
	go func() {
		var accumulator workflowOutputAccumulator
		deliver := true
		for event := range upstream {
			if event.Type == EventDone || event.Type == EventError || event.Type == EventCanceled {
				continue
			}
			accumulator.add(event)
			deliver = sendWorkflowEvent(ctx, out, cloneEvent(event), deliver)
		}
		<-w.primaryDone

		output, attempted, stageErrs, _, _ := accumulator.result()
		w.mu.Lock()
		w.setVisibleOutputLocked(output)
		w.result.AttemptedTokens = outputPartsToTokens(attempted)
		w.result.AttemptedText = outputTextOnly(attempted)
		for _, err := range stageErrs {
			if err != nil && !containsError(w.result.Errors, err) {
				w.result.Errors = append(w.result.Errors, err)
			}
		}
		terminal := Event{Type: EventDone, Source: EventSource{Kind: SourceWorkflow}}
		switch {
		case w.result.Canceled:
			terminal.Type = EventCanceled
			terminal.Err = w.result.CancellationErr
			w.terminalErr = w.result.CancellationErr
		case len(w.result.Errors) > 0:
			terminal.Type = EventError
			terminal.Err = errors.Join(w.result.Errors...)
			w.terminalErr = terminal.Err
		}
		w.result.Complete = true
		result := cloneWorkflowResult(w.result)
		w.mu.Unlock()

		obs.Finished(ctx, result)
		runObs.Finished(result)
		sendTerminalWorkflowEvent(ctx, out, terminal, deliver)
		close(out)
		close(w.done)
	}()
	return out
}

func containsError(errs []error, target error) bool {
	for _, err := range errs {
		if err == target || (err != nil && target != nil && err.Error() == target.Error()) {
			return true
		}
	}
	return false
}

func (w *Workflow) setVisibleOutputLocked(output []OutputPart) {
	w.result.Output = cloneOutputParts(output)
	w.result.Text = outputTextOnly(output)
	w.result.Reasoning = outputReasoningOnly(output)
	w.result.Tokens = outputPartsToTokens(output)
}

func outputPartsFromTokens(tokens []ai.Token) []OutputPart {
	var output []OutputPart
	for _, token := range tokens {
		text := token.Text
		if text == "" {
			text = string(token.Data)
		}
		switch token.Type {
		case ai.TokenTypeText:
			output = append(output, OutputPart{Kind: OutputText, Text: text})
		case ai.TokenTypeThought:
			output = append(output, OutputPart{Kind: OutputReasoning, Text: text})
		}
	}
	return output
}

func outputPartsToTokens(output []OutputPart) []ai.Token {
	var tokens []ai.Token
	for _, part := range output {
		switch part.Kind {
		case OutputText:
			tokens = append(tokens, ai.Token{Type: ai.TokenTypeText, Text: part.Text})
		case OutputReasoning:
			tokens = append(tokens, ai.Token{Type: ai.TokenTypeThought, Text: part.Text})
		}
	}
	return tokens
}

func outputTextOnly(output []OutputPart) string {
	var text string
	for _, part := range output {
		if part.Kind == OutputText {
			text += part.Text
		}
	}
	return text
}

func outputReasoningOnly(output []OutputPart) string {
	var reasoning string
	for _, part := range output {
		if part.Kind == OutputReasoning {
			reasoning += part.Text
		}
	}
	return reasoning
}

func tokenText(tokens []ai.Token) string {
	return outputTextOnly(outputPartsFromTokens(tokens))
}

func tokenReasoning(tokens []ai.Token) string {
	return outputReasoningOnly(outputPartsFromTokens(tokens))
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
	cloned.ResponseFormat = cloneResponseFormat(input.ResponseFormat)
	cloned.Execution.Tools = cloneTools(input.Execution.Tools)
	if input.Execution.ToolChoice != nil {
		choice := cloneToolChoice(*input.Execution.ToolChoice)
		cloned.Execution.ToolChoice = &choice
	}
	if input.Execution.Reasoning != nil {
		reasoning := *input.Execution.Reasoning
		cloned.Execution.Reasoning = &reasoning
	}
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
	if tokens == nil {
		return nil
	}
	cloned := make([]ai.Token, len(tokens))
	for i, token := range tokens {
		cloned[i] = token
		cloned[i].Data = append([]byte(nil), token.Data...)
		cloned[i].ToolCall = cloneToolCall(token.ToolCall)
		if token.Completion != nil {
			completion := *token.Completion
			completion.Raw = append([]byte(nil), token.Completion.Raw...)
			cloned[i].Completion = &completion
		}
	}
	return cloned
}

func cloneToolCall(call *ai.ToolCall) *ai.ToolCall {
	if call == nil {
		return nil
	}
	cloned := *call
	cloned.Args = append([]byte(nil), call.Args...)
	cloned.ThoughtSignature = append([]byte(nil), call.ThoughtSignature...)
	return &cloned
}

func cloneMessages(messages []gaictx.Message) []gaictx.Message {
	if messages == nil {
		return nil
	}
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

func iterationUsage(iterations []loop.Iteration) ai.Usage {
	var usage ai.Usage
	for _, iteration := range iterations {
		usage.Add(iteration.Usage)
	}
	return usage
}

func cloneIterations(iterations []loop.Iteration) []loop.Iteration {
	if iterations == nil {
		return nil
	}
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
				response.Raw = append([]byte(nil), part.Response.Raw...)
				response.ToolCalls = make([]ai.ToolCall, len(part.Response.ToolCalls))
				for k := range part.Response.ToolCalls {
					response.ToolCalls[k] = *cloneToolCall(&part.Response.ToolCalls[k])
				}
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
	result.Output = cloneOutputParts(result.Output)
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

func (w *Workflow) addStage(stage StageResult) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.result.Stages = append(w.result.Stages, cloneStageResult(stage))
}
