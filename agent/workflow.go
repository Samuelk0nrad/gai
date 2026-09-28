package agent

import (
	"context"
	"errors"
	"reflect"
	"sync"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/loop"
)

var (
	ErrWorkflowAlreadyRun      = errors.New("workflow has already been run")
	ErrWorkflowNotStarted      = errors.New("workflow has not been started")
	ErrWorkflowNotConfigured   = errors.New("workflow is not configured")
	ErrMiddlewareNotConfigured = errors.New("middleware is not configured")
)

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

	mu                 sync.RWMutex
	middlewareDone     sync.WaitGroup
	started            bool
	done               chan struct{}
	primaryDone        chan struct{}
	terminalErr        error
	deliveryIncomplete bool
	stageErrorsSeen    map[stageErrorKey]struct{}
	result             WorkflowResult
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
