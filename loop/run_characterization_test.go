package loop_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/testutil/mocks"
)

type failingAttemptPromptBuilder struct {
	stubPromptBuilder
	err error
}

type failingContextPromptBuilder struct {
	stubPromptBuilder
	err error
}

func (b *failingContextPromptBuilder) BuildContext(context.Context) ([]gaictx.Part, error) {
	return nil, b.err
}

func (b *failingAttemptPromptBuilder) BuildPrompt(context.Context, gaictx.Conversation) (string, error) {
	return "", b.err
}

func (b *failingAttemptPromptBuilder) BuildRequest(context.Context, gaictx.Conversation) (string, []ai.RequestMessage, error) {
	return "", nil, b.err
}

type attemptTimeoutModel struct{}

type blockingAfterTokenModel struct{}

type limitMutatingModel struct {
	delegate *scriptedStreamModel
	mutate   func()
	once     sync.Once
}

func (m *limitMutatingModel) Name() string { return m.delegate.Name() }
func (m *limitMutatingModel) Generate(ctx context.Context, request ai.AIRequest) (*ai.AIResponse, error) {
	return m.delegate.Generate(ctx, request)
}
func (m *limitMutatingModel) GenerateStream(ctx context.Context, request ai.AIRequest) <-chan ai.Token {
	m.once.Do(m.mutate)
	return m.delegate.GenerateStream(ctx, request)
}
func (m *limitMutatingModel) Close() error            { return m.delegate.Close() }
func (m *limitMutatingModel) Tokenizer() ai.Tokenizer { return m.delegate.Tokenizer() }

func (blockingAfterTokenModel) Name() string { return "blocking-after-token-model" }
func (blockingAfterTokenModel) Generate(context.Context, ai.AIRequest) (*ai.AIResponse, error) {
	return &ai.AIResponse{}, nil
}
func (blockingAfterTokenModel) GenerateStream(ctx context.Context, _ ai.AIRequest) <-chan ai.Token {
	tokens := make(chan ai.Token)
	go func() {
		defer close(tokens)
		tokens <- ai.Token{Type: ai.TokenTypeText, Text: "partial"}
		<-ctx.Done()
	}()
	return tokens
}
func (blockingAfterTokenModel) Close() error            { return nil }
func (blockingAfterTokenModel) Tokenizer() ai.Tokenizer { return &mocks.MockTokenizer{} }

func (attemptTimeoutModel) Name() string { return "attempt-timeout-model" }
func (attemptTimeoutModel) Generate(context.Context, ai.AIRequest) (*ai.AIResponse, error) {
	return &ai.AIResponse{}, nil
}
func (attemptTimeoutModel) GenerateStream(ctx context.Context, _ ai.AIRequest) <-chan ai.Token {
	tokens := make(chan ai.Token, 1)
	go func() {
		defer close(tokens)
		<-ctx.Done()
		tokens <- ai.Token{Err: ctx.Err()}
	}()
	return tokens
}
func (attemptTimeoutModel) Close() error            { return nil }
func (attemptTimeoutModel) Tokenizer() ai.Tokenizer { return &mocks.MockTokenizer{} }

func eventTypes(events []loop.Event) []loop.EventType {
	types := make([]loop.EventType, len(events))
	for i := range events {
		types[i] = events[i].Type
	}
	return types
}

func requireEventTypes(t *testing.T, events []loop.Event, want ...loop.EventType) {
	t.Helper()
	if got := eventTypes(events); !reflect.DeepEqual(got, want) {
		t.Fatalf("event types = %v, want %v", got, want)
	}
}

func requireAttemptMetadata(t *testing.T, event loop.Event, iteration, attempt, retry int) {
	t.Helper()
	if event.IterationCount != iteration || event.AttemptID != attempt || event.RetryCount != retry {
		t.Fatalf("event metadata = iteration %d attempt %d retry %d, want %d/%d/%d: %#v",
			event.IterationCount, event.AttemptID, event.RetryCount, iteration, attempt, retry, event)
	}
}

func TestLoopCharacterizationSuccessEventSequence(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{{{Type: ai.TokenTypeText, Text: "done"}}}}
	l := loop.New(model, nil, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1

	events := collectLoopEvents(t, l, context.Background())
	requireEventTypes(t, events,
		loop.EventAttemptStart,
		loop.EventToken,
		loop.EventIterationDone,
		loop.EventDone,
	)
	for _, index := range []int{0, 1, 2} {
		requireAttemptMetadata(t, events[index], 1, 1, 0)
	}
	if events[2].Iteration == nil || events[2].Iteration.UserMessage == nil || events[2].PartCount != 1 {
		t.Fatalf("completed iteration snapshot = %#v, want user input and one part", events[2])
	}
	if got := events[2].Iteration.Parts[0].Response.Text; got != "done" {
		t.Fatalf("completed text = %q, want done", got)
	}
}

func TestLoopCharacterizationRetryEventSequence(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{
			{Type: ai.TokenTypeText, Text: "partial"},
			{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("temporary")}},
		},
		{{Type: ai.TokenTypeText, Text: "final"}},
	}}
	l := loop.New(model, nil, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}

	events := collectLoopEvents(t, l, context.Background())
	requireEventTypes(t, events,
		loop.EventAttemptStart,
		loop.EventToken,
		loop.EventRetry,
		loop.EventAttemptStart,
		loop.EventToken,
		loop.EventIterationDone,
		loop.EventDone,
	)
	for _, index := range []int{0, 1} {
		requireAttemptMetadata(t, events[index], 1, 1, 0)
	}
	requireAttemptMetadata(t, events[2], 1, 1, 1)
	for _, index := range []int{3, 4, 5} {
		requireAttemptMetadata(t, events[index], 1, 2, 1)
	}
	if events[2].Iteration == nil || events[2].PartCount != 1 || events[2].Iteration.Parts[0].Response.Text != "partial" {
		t.Fatalf("retry snapshot = %#v, want immutable partial attempt", events[2])
	}
	if events[5].Iteration == nil || events[5].Iteration.Parts[0].Response.Text != "final" {
		t.Fatalf("accepted snapshot = %#v, want final attempt", events[5])
	}
}

func TestLoopCharacterizationRequiredToolDiscardEventSequence(t *testing.T) {
	t.Parallel()

	toolCall := ai.Token{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{
		ID: "call-1", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"payload"}`),
	}}
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Type: ai.TokenTypeText, Text: "discard me"}},
		{toolCall},
		{{Type: ai.TokenTypeText, Text: "done"}},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.ToolTransport = loop.ToolTransportText
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired}
	l.MaxLoopIterations = 3

	events := collectLoopEvents(t, l, context.Background())
	requireEventTypes(t, events,
		loop.EventAttemptStart,
		loop.EventDiscard,
		loop.EventAttemptStart,
		loop.EventToken,
		loop.EventToolStart,
		loop.EventToolResult,
		loop.EventIterationDone,
		loop.EventAttemptStart,
		loop.EventToken,
		loop.EventIterationDone,
		loop.EventDone,
	)
	requireAttemptMetadata(t, events[0], 1, 1, 0)
	requireAttemptMetadata(t, events[1], 1, 1, 0)
	if events[1].PartCount != 1 || events[1].Iteration == nil || len(events[1].Iteration.Parts) != 0 || events[1].Iteration.UserMessage != nil {
		t.Fatalf("discard snapshot = %#v, want sanitized metadata only", events[1])
	}
	for _, index := range []int{2, 3, 4, 5, 6} {
		requireAttemptMetadata(t, events[index], 2, 1, 0)
	}
	for _, index := range []int{7, 8, 9} {
		requireAttemptMetadata(t, events[index], 3, 1, 0)
	}
}

func TestLoopCharacterizationToolErrorEventSequence(t *testing.T) {
	t.Parallel()

	toolCall := ai.Token{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{
		ID: "call-1", Type: "function", Name: "failure", Args: json.RawMessage(`{"text":"payload"}`),
	}}
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{toolCall},
		{{Type: ai.TokenTypeText, Text: "done"}},
	}}
	l := loop.New(model, []loop.Tool{sentinelErrorTool{}}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 2

	events := collectLoopEvents(t, l, context.Background())
	requireEventTypes(t, events,
		loop.EventAttemptStart,
		loop.EventToken,
		loop.EventToolStart,
		loop.EventToolError,
		loop.EventIterationDone,
		loop.EventAttemptStart,
		loop.EventToken,
		loop.EventIterationDone,
		loop.EventDone,
	)
	for _, index := range []int{0, 1, 2, 3, 4} {
		requireAttemptMetadata(t, events[index], 1, 1, 0)
	}
	if !errors.Is(events[3].Err, events[3].ToolResponse.ErrorValue()) {
		t.Fatalf("tool error event = %#v, want returned tool error", events[3])
	}
	for _, index := range []int{5, 6, 7} {
		requireAttemptMetadata(t, events[index], 2, 1, 0)
	}
}

func TestLoopCharacterizationPromptFailureEventSequence(t *testing.T) {
	t.Parallel()

	promptErr := errors.New("build request failed")
	builder := &failingAttemptPromptBuilder{
		stubPromptBuilder: stubPromptBuilder{userPrompt: "Initial prompt"},
		err:               promptErr,
	}
	l := loop.New(&scriptedStreamModel{}, nil, builder, nil)
	l.MaxLoopIterations = 1

	events := collectLoopEvents(t, l, context.Background())
	requireEventTypes(t, events, loop.EventAttemptStart, loop.EventError)
	requireAttemptMetadata(t, events[0], 1, 1, 0)
	requireAttemptMetadata(t, events[1], 1, 1, 0)
	if !errors.Is(events[1].Err, loop.ErrBuildPrompt) || !errors.Is(events[1].Err, promptErr) {
		t.Fatalf("prompt error = %v, want ErrBuildPrompt wrapping sentinel", events[1].Err)
	}
	if events[1].Iteration == nil || events[1].Iteration.UserMessage == nil || events[1].PartCount != 0 {
		t.Fatalf("prompt failure snapshot = %#v, want user input and no parts", events[1])
	}
}

func TestLoopCharacterizationContextPreparationFailureEventSequence(t *testing.T) {
	t.Parallel()

	contextErr := errors.New("build context failed")
	builder := &failingContextPromptBuilder{
		stubPromptBuilder: stubPromptBuilder{userPrompt: "Initial prompt"},
		err:               contextErr,
	}
	l := loop.New(&scriptedStreamModel{}, nil, builder, nil)

	events := collectLoopEvents(t, l, context.Background())
	requireEventTypes(t, events, loop.EventError)
	if !errors.Is(events[0].Err, loop.ErrBuildPrompt) || !errors.Is(events[0].Err, contextErr) {
		t.Fatalf("context error = %v, want ErrBuildPrompt wrapping sentinel", events[0].Err)
	}
	if events[0].IterationCount != 0 || events[0].AttemptID != 0 || events[0].Iteration != nil {
		t.Fatalf("context preparation failure must precede attempts, got %#v", events[0])
	}
}

func TestLoopCharacterizationValidationErrorIsSynchronous(t *testing.T) {
	t.Parallel()

	l := &loop.Loop{}
	eventStream := l.Run(context.Background())
	select {
	case event, ok := <-eventStream:
		if !ok || event.Type != loop.EventError || !errors.Is(event.Err, loop.ErrModelNotConfigured) {
			t.Fatalf("validation event = %#v, open=%t; want model configuration error", event, ok)
		}
	default:
		t.Fatal("validation error was not available before Run returned")
	}
	if _, ok := <-eventStream; ok {
		t.Fatal("validation event stream must close synchronously")
	}
}

func TestLoopCharacterizationIterationLimitIsSnapshotted(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name         string
		initialLimit int
		mutatedLimit int
		wantRequests int
		wantMaxError bool
	}{
		{name: "lowered during run", initialLimit: 2, mutatedLimit: 1, wantRequests: 2},
		{name: "raised during run", initialLimit: 1, mutatedLimit: 2, wantRequests: 1, wantMaxError: true},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			delegate := &scriptedStreamModel{sequences: [][]ai.Token{
				{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{
					ID: "call-1", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"payload"}`),
				}}},
				{{Type: ai.TokenTypeText, Text: "done"}},
			}}
			var l *loop.Loop
			model := &limitMutatingModel{delegate: delegate}
			model.mutate = func() { l.MaxLoopIterations = test.mutatedLimit }
			l = loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
			l.MaxLoopIterations = test.initialLimit

			err := loopError(collectLoopEvents(t, l, context.Background()))
			if got := errors.Is(err, loop.ErrMaxIterations); got != test.wantMaxError {
				t.Fatalf("ErrMaxIterations = %t, want %t (error %v)", got, test.wantMaxError, err)
			}
			if got := len(delegate.Requests()); got != test.wantRequests {
				t.Fatalf("model requests = %d, want snapshotted limit behavior with %d", got, test.wantRequests)
			}
		})
	}
}

func TestLoopCharacterizationCallerCancellationEventSequence(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	l := loop.New(blockingAfterTokenModel{}, nil, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1

	eventStream := l.Run(ctx)
	events := []loop.Event{<-eventStream, <-eventStream}
	cancel()
	for event := range eventStream {
		events = append(events, event)
	}
	requireEventTypes(t, events, loop.EventAttemptStart, loop.EventToken, loop.EventCanceled)
	for _, event := range events {
		requireAttemptMetadata(t, event, 1, 1, 0)
	}
	if !errors.Is(events[2].Err, context.Canceled) || events[2].Iteration == nil || events[2].PartCount != 1 {
		t.Fatalf("cancellation event = %#v, want partial attempt snapshot", events[2])
	}
}

func TestLoopCharacterizationAttemptTimeoutEventSequence(t *testing.T) {
	t.Parallel()

	l := loop.New(attemptTimeoutModel{}, nil, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 0, AttemptTimeout: 5 * time.Millisecond}

	events := collectLoopEvents(t, l, context.Background())
	requireEventTypes(t, events, loop.EventAttemptStart, loop.EventError)
	requireAttemptMetadata(t, events[0], 1, 1, 0)
	requireAttemptMetadata(t, events[1], 1, 1, 0)
	if !errors.Is(events[1].Err, loop.ErrAttemptTimeout) || !errors.Is(events[1].Err, loop.ErrMaxRetries) {
		t.Fatalf("timeout error = %v, want exhausted attempt timeout", events[1].Err)
	}
	if events[1].Iteration == nil || events[1].Iteration.UserMessage == nil || events[1].PartCount != 0 {
		t.Fatalf("timeout snapshot = %#v, want empty attempt with retained input", events[1])
	}
}
