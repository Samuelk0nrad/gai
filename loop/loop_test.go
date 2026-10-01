package loop_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/testutil/mocks"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

type wrapStreamModel struct {
	ai.Model
}

func (m wrapStreamModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	return ai.DetectToolCallsInStream(ctx, m.Model.GenerateStream(ctx, req), nil)
}

type scriptedStreamModel struct {
	sequences     [][]ai.Token
	delays        []time.Duration
	ignoreContext bool
	idx           int
	mu            sync.Mutex
	requests      []ai.AIRequest
	requestTimes  []time.Time
}

type cancelAfterTokenModel struct {
	cancel context.CancelFunc
}

type retryCancellationModel struct {
	attemptCanceled chan struct{}
	calls           atomic.Int32
}

type countingPromptBuilder struct {
	count atomic.Int32
}

type deadlineRecordingPromptBuilder struct {
	stubPromptBuilder
	hasDeadline atomic.Bool
}

type deadlineRecordingTool struct {
	hasDeadline atomic.Bool
}

type countingTool struct {
	calls atomic.Int32
}

type failingToolResponseProcessor struct {
	err error
}

type sentinelErrorTool struct{}

func (sentinelErrorTool) Name() string              { return "failure" }
func (sentinelErrorTool) Description() string       { return "Returns a test error." }
func (sentinelErrorTool) Params() ai.ToolParameters { return loop.NewEchoTool().Params() }
func (sentinelErrorTool) Function(context.Context, *ai.ToolCall) *loop.ToolResponse {
	return loop.NewToolError(errors.New("tool-output-sentinel-secret"))
}

func (p failingToolResponseProcessor) Process(req ai.ToolCall, res *loop.ToolResponse) error {
	return p.err
}

func (b *countingPromptBuilder) PrependContextSource(ctx context.Context, source gaictx.ContextSource) error {
	return nil
}

func (b *countingPromptBuilder) AppendContextSource(ctx context.Context, source gaictx.ContextSource) error {
	return nil
}

func (b *countingPromptBuilder) AppendContextSources(ctx context.Context, sources ...gaictx.ContextSource) error {
	return nil
}

func (b *countingPromptBuilder) AppendSystemInstructions(ctx context.Context, instructions ...gaictx.Part) error {
	return nil
}

func (b *countingPromptBuilder) BuildContext(ctx context.Context) ([]gaictx.Part, error) {
	return nil, nil
}

func (b *countingPromptBuilder) BuildPrompt(ctx context.Context, conv gaictx.Conversation) (string, error) {
	count := b.count.Add(1)
	return fmt.Sprintf("prompt-%d", count), nil
}

func (b *deadlineRecordingPromptBuilder) BuildPrompt(ctx context.Context, conv gaictx.Conversation) (string, error) {
	_, hasDeadline := ctx.Deadline()
	b.hasDeadline.Store(hasDeadline)
	return b.stubPromptBuilder.BuildPrompt(ctx, conv)
}

func (t *deadlineRecordingTool) Name() string { return "record-deadline" }
func (t *deadlineRecordingTool) Description() string {
	return "Records whether its context has a deadline."
}
func (t *deadlineRecordingTool) Params() ai.ToolParameters {
	return loop.NewEchoTool().Params()
}
func (t *deadlineRecordingTool) Function(ctx context.Context, _ *ai.ToolCall) *loop.ToolResponse {
	_, hasDeadline := ctx.Deadline()
	t.hasDeadline.Store(hasDeadline)
	return loop.NewToolSuccess("ok")
}

func (t *countingTool) Name() string              { return "count" }
func (t *countingTool) Description() string       { return "Counts invocations." }
func (t *countingTool) Params() ai.ToolParameters { return loop.NewEchoTool().Params() }
func (t *countingTool) Function(context.Context, *ai.ToolCall) *loop.ToolResponse {
	t.calls.Add(1)
	return loop.NewToolSuccess("ok")
}

func (b *deadlineRecordingPromptBuilder) BuildRequest(ctx context.Context, conv gaictx.Conversation) (string, []ai.RequestMessage, error) {
	prompt, err := b.BuildPrompt(ctx, conv)
	if err != nil {
		return "", nil, err
	}
	return prompt, []ai.RequestMessage{{Role: ai.RequestMessageRoleUser, Text: prompt}}, nil
}

func (b *countingPromptBuilder) Input() gaictx.PromptInput {
	return gaictx.PromptInput{User: gaictx.NewTextContent("Initial prompt")}
}

func (b *countingPromptBuilder) SetInput(input gaictx.PromptInput) {
}

type stubPromptBuilder struct {
	systemPrompt string
	userPrompt   string
	contextText  string
	buildContext func() string
}

type nonNilConversationPromptBuilder struct {
	stubPromptBuilder
}

func (b *nonNilConversationPromptBuilder) BuildPrompt(ctx context.Context, conv gaictx.Conversation) (string, error) {
	if conv == nil {
		return "", errors.New("conversation must not be nil")
	}
	return b.stubPromptBuilder.BuildPrompt(ctx, conv)
}

func (b *stubPromptBuilder) PrependContextSource(ctx context.Context, source gaictx.ContextSource) error {
	return nil
}

func (b *stubPromptBuilder) AppendContextSource(ctx context.Context, source gaictx.ContextSource) error {
	return nil
}

func (b *stubPromptBuilder) AppendContextSources(ctx context.Context, sources ...gaictx.ContextSource) error {
	return nil
}

func (b *stubPromptBuilder) AppendSystemInstructions(ctx context.Context, instructions ...gaictx.Part) error {
	return nil
}

func (b *stubPromptBuilder) BuildContext(ctx context.Context) ([]gaictx.Part, error) {
	if b.buildContext != nil {
		b.contextText = b.buildContext()
	}
	return nil, nil
}

func (b *stubPromptBuilder) BuildPrompt(ctx context.Context, conv gaictx.Conversation) (string, error) {
	var prompt strings.Builder
	if b.systemPrompt != "" {
		prompt.WriteString(b.systemPrompt)
		prompt.WriteString("\n")
	}
	if b.contextText != "" {
		prompt.WriteString(b.contextText)
		prompt.WriteString("\n")
	}
	if b.userPrompt != "" {
		prompt.WriteString(b.userPrompt)
		prompt.WriteString("\n")
	}
	if conv != nil {
		prompt.WriteString(renderTestMessages(conv.Messages()))
	}
	return prompt.String(), nil
}

func (b *stubPromptBuilder) BuildRequest(ctx context.Context, conv gaictx.Conversation) (string, []ai.RequestMessage, error) {
	prompt, err := b.BuildPrompt(ctx, conv)
	if err != nil {
		return "", nil, err
	}
	messages, err := testNativeMessages(ctx, b, conv, prompt)
	if err != nil {
		return "", nil, err
	}
	return prompt, messages, nil
}

type emptyPromptConversation struct{}

func (emptyPromptConversation) Messages() []gaictx.Message { return nil }

func testNativeMessages(ctx context.Context, builder gaictx.PromptBuilder, conv gaictx.Conversation, prompt string) ([]ai.RequestMessage, error) {
	var nativeMessages []ai.RequestMessage
	if native, ok := conv.(gaictx.NativeConversation); ok {
		nativeMessages = native.NativeMessages()
	}
	if len(nativeMessages) == 0 {
		return []ai.RequestMessage{{Role: ai.RequestMessageRoleUser, Text: prompt}}, nil
	}
	base, err := builder.BuildPrompt(ctx, emptyPromptConversation{})
	if err != nil {
		return nil, err
	}
	messages := []ai.RequestMessage{{Role: ai.RequestMessageRoleUser, Text: base}}
	messages = append(messages, nativeMessages...)
	return messages, nil
}

func (b *stubPromptBuilder) Input() gaictx.PromptInput {
	if b.userPrompt == "" {
		return gaictx.PromptInput{}
	}
	return gaictx.PromptInput{User: gaictx.NewTextContent(b.userPrompt)}
}

func (b *stubPromptBuilder) SetInput(input gaictx.PromptInput) {
	b.userPrompt = ""
	if input.User != nil {
		b.userPrompt = input.User.String()
	}
}

func (m *scriptedStreamModel) Name() string {
	return "scripted-stream-model"
}

func (m *scriptedStreamModel) Generate(ctx context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
	return &ai.AIResponse{}, nil
}

func (m *scriptedStreamModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token)

	go func() {
		defer close(out)
		m.mu.Lock()
		m.requests = append(m.requests, req)
		m.requestTimes = append(m.requestTimes, time.Now())
		m.mu.Unlock()

		if m.idx >= len(m.sequences) {
			return
		}
		seq := m.sequences[m.idx]
		var delay time.Duration
		if m.idx < len(m.delays) {
			delay = m.delays[m.idx]
		}
		m.idx++
		if delay > 0 {
			time.Sleep(delay)
		}

		for _, tok := range seq {
			if m.ignoreContext {
				out <- tok
				continue
			}
			select {
			case <-ctx.Done():
				return
			case out <- tok:
			}
		}
	}()

	return out
}

func (m *scriptedStreamModel) Close() error {
	return nil
}

func (m *scriptedStreamModel) TokenCounter() ai.TokenCounter {
	return &mocks.MockTokenCounter{}
}

func (m *scriptedStreamModel) Requests() []ai.AIRequest {
	m.mu.Lock()
	defer m.mu.Unlock()

	requests := make([]ai.AIRequest, len(m.requests))
	copy(requests, m.requests)
	return requests
}

func (m *scriptedStreamModel) RequestTimes() []time.Time {
	m.mu.Lock()
	defer m.mu.Unlock()

	times := make([]time.Time, len(m.requestTimes))
	copy(times, m.requestTimes)
	return times
}

func (m *cancelAfterTokenModel) Name() string {
	return "cancel-after-token-model"
}

func (m *cancelAfterTokenModel) Generate(context.Context, ai.AIRequest) (*ai.AIResponse, error) {
	return &ai.AIResponse{}, nil
}

func (m *cancelAfterTokenModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)
	go func() {
		defer close(out)
		out <- ai.Token{Type: ai.TokenTypeText, Data: []byte("partial")}
		m.cancel()
	}()
	return out
}

func (m *cancelAfterTokenModel) Close() error {
	return nil
}

func (m *cancelAfterTokenModel) TokenCounter() ai.TokenCounter {
	return &mocks.MockTokenCounter{}
}

func (m *retryCancellationModel) Name() string { return "retry-cancellation-model" }

func (m *retryCancellationModel) Generate(context.Context, ai.AIRequest) (*ai.AIResponse, error) {
	return &ai.AIResponse{}, nil
}

func (m *retryCancellationModel) GenerateStream(ctx context.Context, _ ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token)
	attempt := m.calls.Add(1)
	go func() {
		defer close(out)
		if attempt == 1 {
			out <- ai.Token{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient}}
			<-ctx.Done()
			close(m.attemptCanceled)
			return
		}
		out <- ai.Token{Type: ai.TokenTypeText, Data: []byte("done")}
	}()
	return out
}

func (m *retryCancellationModel) Close() error { return nil }

func (m *retryCancellationModel) TokenCounter() ai.TokenCounter {
	return &mocks.MockTokenCounter{}
}

func collectLoopEvents(t *testing.T, l *loop.Loop, ctx context.Context) []loop.Event {
	t.Helper()

	var events []loop.Event
	for event := range l.Run(ctx) {
		events = append(events, event)
	}
	return events
}

func loopError(events []loop.Event) error {
	var err error
	for _, event := range events {
		if event.Type == loop.EventError {
			err = event.Err
		}
	}
	return err
}

func TestLoopPropagatesReasoningToModelRequests(t *testing.T) {
	t.Parallel()

	for _, reasoning := range []ai.ReasoningConfig{
		{},
		{Enabled: true, IncludeThoughts: true, BudgetTokens: 128, Effort: ai.ReasoningEffortHigh},
	} {
		reasoning := reasoning
		t.Run(fmt.Sprintf("enabled=%t", reasoning.Enabled), func(t *testing.T) {
			t.Parallel()

			model := &scriptedStreamModel{sequences: [][]ai.Token{{{Type: ai.TokenTypeText, Text: "done"}}}}
			l := loop.New(model, nil, &countingPromptBuilder{}, nil)
			l.Reasoning = reasoning

			if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
				t.Fatalf("unexpected loop error: %v", err)
			}
			requests := model.Requests()
			if len(requests) != 1 {
				t.Fatalf("expected 1 model request, got %d", len(requests))
			}
			if requests[0].Reasoning != reasoning {
				t.Fatalf("expected reasoning %+v, got %+v", reasoning, requests[0].Reasoning)
			}
		})
	}
}

func TestLoopValidateRejectsMissingNamedRequiredTool(t *testing.T) {
	t.Parallel()

	l := loop.New(&scriptedStreamModel{}, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"missing"}}

	err := l.Validate()
	if !errors.Is(err, loop.ErrRequiredToolNotConfigured) {
		t.Fatalf("Validate() error = %v, want ErrRequiredToolNotConfigured", err)
	}
}

func TestLoopValidateRejectsRequiredToolChoiceWithoutTools(t *testing.T) {
	t.Parallel()

	l := loop.New(&scriptedStreamModel{}, nil, testPromptBuilder(), nil)
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired}

	err := l.Validate()
	if !errors.Is(err, loop.ErrRequiredToolNotConfigured) {
		t.Fatalf("Validate() error = %v, want ErrRequiredToolNotConfigured", err)
	}
}

func TestLoopValidateRejectsInvalidToolChoiceMode(t *testing.T) {
	t.Parallel()

	l := loop.New(&scriptedStreamModel{}, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceMode("requred")}

	if err := l.Validate(); err == nil {
		t.Fatal("Validate() succeeded with an invalid tool choice mode")
	}
}

func TestEffectiveToolsAppliesTextTransportChoice(t *testing.T) {
	t.Parallel()

	search := namedTestTool{name: "search"}
	weather := namedTestTool{name: "weather"}

	got, err := loop.EffectiveTools([]loop.Tool{search, weather}, ai.ToolChoice{
		Mode:  ai.ToolChoiceRequired,
		Names: []string{"weather"},
	}, loop.ToolTransportText)
	if err != nil {
		t.Fatalf("EffectiveTools() error = %v", err)
	}
	if len(got) != 1 || got[0].Name() != "weather" {
		t.Fatalf("EffectiveTools() = %#v, want only weather", got)
	}

	got, err = loop.EffectiveTools([]loop.Tool{search}, ai.ToolChoice{Mode: ai.ToolChoiceNone}, loop.ToolTransportText)
	if err != nil {
		t.Fatalf("EffectiveTools() error = %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("EffectiveTools() = %#v, want no tools", got)
	}
}

func TestEffectiveToolsRejectsUnknownTransport(t *testing.T) {
	t.Parallel()

	_, err := loop.EffectiveTools(
		[]loop.Tool{nil},
		ai.ToolChoice{Mode: ai.ToolChoiceAuto},
		loop.ToolTransportMode(255),
	)
	if err == nil {
		t.Fatal("EffectiveTools() succeeded with an unknown transport mode")
	}
	if !strings.Contains(err.Error(), "invalid tool transport mode") {
		t.Fatalf("EffectiveTools() error = %v, want invalid transport error", err)
	}
}

func TestLoopValidateTextTransportRejectsInvalidToolDefinitions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		tools []loop.Tool
	}{
		{
			name:  "non-canonical name",
			tools: []loop.Tool{namedTestTool{name: " search "}},
		},
		{
			name: "duplicate names",
			tools: []loop.Tool{
				namedTestTool{name: "search"},
				namedTestTool{name: "search"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := loop.New(&scriptedStreamModel{}, tt.tools, testPromptBuilder(), nil)
			l.ToolTransport = loop.ToolTransportText

			if err := l.Validate(); !errors.Is(err, ai.ErrInvalidToolDefinition) {
				t.Fatalf("Validate() error = %v, want ErrInvalidToolDefinition", err)
			}
		})
	}
}

func TestLoopDowngradesRequiredToolChoiceAfterToolCall(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"payload"}`)}}},
		{{Type: ai.TokenTypeText, Text: "done"}},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired}

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	requests := model.Requests()
	if len(requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(requests))
	}
	if requests[0].ToolChoice.Mode != ai.ToolChoiceRequired {
		t.Fatalf("first tool choice = %#v, want required", requests[0].ToolChoice)
	}
	if requests[1].ToolChoice.Mode != ai.ToolChoiceAuto {
		t.Fatalf("second tool choice = %#v, want auto after the required call", requests[1].ToolChoice)
	}
}

func TestLoopNativeTransportDoesNotExecuteDifferentConfiguredToolOutsideNamedRequiredChoice(t *testing.T) {
	t.Parallel()

	unselected := &countingTool{}
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-count-before", Type: "function", Name: "count", Args: json.RawMessage(`{"text":"unselected before"}`)}}},
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-echo", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"selected"}`)}}},
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-count-after", Type: "function", Name: "count", Args: json.RawMessage(`{"text":"unselected after"}`)}}},
		{{Type: ai.TokenTypeText, Text: "done"}},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool(), unselected}, testPromptBuilder(), nil)
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"echo"}}

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	if calls := unselected.calls.Load(); calls != 0 {
		t.Fatalf("unselected tool calls = %d, want 0", calls)
	}
	requests := model.Requests()
	if len(requests) != 4 {
		t.Fatalf("requests = %d, want 4", len(requests))
	}
	if requests[0].ToolChoice.Mode != ai.ToolChoiceRequired || requests[1].ToolChoice.Mode != ai.ToolChoiceRequired || requests[2].ToolChoice.Mode != ai.ToolChoiceAuto || requests[3].ToolChoice.Mode != ai.ToolChoiceAuto {
		t.Fatalf("tool choices = %#v, %#v, %#v, %#v; want required, required, auto, auto", requests[0].ToolChoice, requests[1].ToolChoice, requests[2].ToolChoice, requests[3].ToolChoice)
	}
	for i, request := range requests {
		if len(request.Tools) != 1 || request.Tools[0].Name != "echo" {
			t.Fatalf("request %d tools = %#v, want only echo", i, request.Tools)
		}
	}
	for _, event := range events {
		if event.IterationCount == 1 && (event.Type == loop.EventToken || event.Type == loop.EventIterationDone) {
			t.Fatalf("unselected native tool call must not be observable, got %#v", event)
		}
	}
}

func TestLoopTextTransportDoesNotFinishBeforeRequiredToolCall(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Type: ai.TokenTypeText, Text: "I will answer without a tool."}},
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"payload"}`)}}},
		{{Type: ai.TokenTypeText, Text: "done"}},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.ToolTransport = loop.ToolTransportText
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired}

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	requests := model.Requests()
	if len(requests) != 3 {
		t.Fatalf("requests = %d, want 3", len(requests))
	}
	for i, request := range requests {
		if len(request.Tools) != 0 || request.ToolChoice.Mode != "" {
			t.Fatalf("text request %d must not use provider-native tools or tool choice: %#v", i, request)
		}
	}
	if strings.Contains(requests[1].Prompt, "I will answer without a tool.") {
		t.Fatalf("second prompt must not include the rejected response: %q", requests[1].Prompt)
	}
	for _, event := range events {
		if event.IterationCount == 1 && (event.Type == loop.EventToken || event.Type == loop.EventIterationDone) {
			t.Fatalf("rejected iteration must not be observable, got %#v", event)
		}
	}
	if len(l.Iterations) != 2 {
		t.Fatalf("persisted iterations = %d, want only accepted iterations", len(l.Iterations))
	}
	if l.Iterations[0].UserMessage == nil {
		t.Fatal("first accepted iteration must retain the original user message")
	}
	messages := l.Messages()
	if len(messages) == 0 || messages[0].Role != gaictx.RoleUser || messages[0].Content.String() != "Initial prompt" {
		t.Fatalf("messages must begin with the original user request, got %#v", messages)
	}
}

func TestLoopTextTransportDoesNotExecuteToolWhenChoiceIsNone(t *testing.T) {
	t.Parallel()

	tool := &countingTool{}
	model := &scriptedStreamModel{sequences: [][]ai.Token{{
		{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "count", Args: json.RawMessage(`{"text":"payload"}`)}},
	}}}
	l := loop.New(model, []loop.Tool{tool}, testPromptBuilder(), nil)
	l.ToolTransport = loop.ToolTransportText
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceNone}

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	if calls := tool.calls.Load(); calls != 0 {
		t.Fatalf("disabled tool calls = %d, want 0", calls)
	}
	for _, event := range events {
		if event.Type == loop.EventToken && event.Token.Type == ai.TokenTypeToolCall {
			t.Fatalf("disabled tool call must not be observable, got %#v", event)
		}
	}
	for _, iteration := range l.Iterations {
		for _, part := range iteration.Parts {
			if part.ToolReq != nil {
				t.Fatalf("disabled tool call must not be retained, got %#v", part.ToolReq)
			}
		}
	}
}

func TestLoopTextTransportRejectedResponseResetsRetryBudget(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("temporary before rejected response")}}},
		{{Type: ai.TokenTypeText, Text: "I will answer without a tool."}},
		{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("temporary after rejected response")}}},
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"payload"}`)}}},
		{{Type: ai.TokenTypeText, Text: "done"}},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.ToolTransport = loop.ToolTransportText
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired}
	l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	if got := len(model.Requests()); got != 5 {
		t.Fatalf("requests = %d, want 5", got)
	}
}

func TestLoopTextTransportDoesNotSatisfyRequiredToolChoiceWithUnknownTool(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "missing", Args: json.RawMessage(`{"text":"payload"}`)}}},
		{{Type: ai.TokenTypeText, Text: "done"}},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.ToolTransport = loop.ToolTransportText
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired}
	l.MaxLoopIterations = 2

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); !errors.Is(err, loop.ErrMaxIterations) {
		t.Fatalf("loop error = %v, want ErrMaxIterations after no permitted tool call", err)
	}
	if len(model.Requests()) != 2 {
		t.Fatalf("requests = %d, want 2", len(model.Requests()))
	}
	for _, event := range events {
		if event.Type == loop.EventDone {
			t.Fatalf("run must not finish after an unavailable tool call")
		}
	}
}

func TestLoopTextTransportDoesNotSatisfyNamedRequiredToolChoiceWithDifferentConfiguredTool(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-echo", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"payload"}`)}}},
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-failure", Type: "function", Name: "failure", Args: json.RawMessage(`{"text":"payload"}`)}}},
		{{Type: ai.TokenTypeText, Text: "done"}},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool(), sentinelErrorTool{}}, testPromptBuilder(), nil)
	l.ToolTransport = loop.ToolTransportText
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"failure"}}

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	if got := len(model.Requests()); got != 3 {
		t.Fatalf("requests = %d, want 3", got)
	}
	for _, event := range events {
		if event.IterationCount == 1 && (event.Type == loop.EventToken || event.Type == loop.EventIterationDone) {
			t.Fatalf("mismatched required tool call must not be observable, got %#v", event)
		}
	}
}

func TestLoopTextTransportDiscardsMixedRequiredToolResponse(t *testing.T) {
	t.Parallel()

	unselected := &countingTool{}
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{
			{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-echo-1", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"selected"}`)}},
			{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-count", Type: "function", Name: "count", Args: json.RawMessage(`{"text":"unselected"}`)}},
		},
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-echo-2", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"selected"}`)}}},
		{{Type: ai.TokenTypeText, Text: "done"}},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool(), unselected}, testPromptBuilder(), nil)
	l.ToolTransport = loop.ToolTransportText
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"echo"}}

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	if calls := unselected.calls.Load(); calls != 0 {
		t.Fatalf("unselected tool calls = %d, want 0", calls)
	}
	if requests := len(model.Requests()); requests != 3 {
		t.Fatalf("requests = %d, want 3 after discarding the mixed response", requests)
	}
	for _, event := range events {
		if event.IterationCount == 1 && (event.Type == loop.EventToken || event.Type == loop.EventIterationDone) {
			t.Fatalf("mixed response must not be observable, got %#v", event)
		}
	}
}

func TestLoopTextTransportRetainsNamedToolRestrictionAfterRequirementIsSatisfied(t *testing.T) {
	t.Parallel()

	unselected := &countingTool{}
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-echo", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"selected"}`)}}},
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-count", Type: "function", Name: "count", Args: json.RawMessage(`{"text":"unselected"}`)}}},
		{{Type: ai.TokenTypeText, Text: "done"}},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool(), unselected}, testPromptBuilder(), nil)
	l.ToolTransport = loop.ToolTransportText
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"echo"}}

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	if calls := unselected.calls.Load(); calls != 0 {
		t.Fatalf("unselected tool calls = %d, want 0", calls)
	}
	if requests := len(model.Requests()); requests != 3 {
		t.Fatalf("requests = %d, want 3 after discarding the later unselected call", requests)
	}
	for _, event := range events {
		if event.IterationCount == 2 && (event.Type == loop.EventToken || event.Type == loop.EventIterationDone) {
			t.Fatalf("later unselected tool call must not be observable, got %#v", event)
		}
	}
}

func TestLoopTextTransportDoesNotExposeMixedResponseWithUnknownTool(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{
			{Type: ai.TokenTypeText, Text: "I will answer without a tool."},
			{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-missing", Type: "function", Name: "missing", Args: json.RawMessage(`{"text":"payload"}`)}},
		},
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-echo", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"payload"}`)}}},
		{{Type: ai.TokenTypeText, Text: "done"}},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.ToolTransport = loop.ToolTransportText
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired}

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	requests := model.Requests()
	if len(requests) != 3 {
		t.Fatalf("requests = %d, want 3", len(requests))
	}
	if strings.Contains(requests[1].Prompt, "I will answer without a tool.") {
		t.Fatalf("second prompt must not include the rejected response: %q", requests[1].Prompt)
	}
	for _, event := range events {
		if event.IterationCount == 1 && (event.Type == loop.EventToken || event.Type == loop.EventIterationDone) {
			t.Fatalf("rejected iteration must not be observable, got %#v", event)
		}
	}
	if len(l.Iterations) != 2 {
		t.Fatalf("persisted iterations = %d, want only accepted iterations", len(l.Iterations))
	}
}

func loopEventsOfType(events []loop.Event, eventType loop.EventType) []loop.Event {
	var filtered []loop.Event
	for _, event := range events {
		if event.Type == eventType {
			filtered = append(filtered, event)
		}
	}
	return filtered
}

func TestLoopRejectsInvalidResponseFormatWithoutWrapping(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{}
	l := loop.New(model, nil, testPromptBuilder(), nil)
	l.ResponseFormat = ai.ResponseFormat{Type: ai.ResponseFormatJSONSchema}

	err := loopError(collectLoopEvents(t, l, context.Background()))
	if !errors.Is(err, ai.ErrInvalidResponseFormat) {
		t.Fatalf("expected invalid response format error, got %v", err)
	}
	if len(model.Requests()) != 0 {
		t.Fatalf("model must not be called for an invalid response format")
	}
}

func renderTestMessages(messages []gaictx.Message) string {
	var builder strings.Builder
	for i, message := range messages {
		builder.WriteString("<")
		builder.WriteString(string(message.Role))
		builder.WriteString(" key=")
		builder.WriteString(fmt.Sprintf("%d", i))
		builder.WriteString(">\n")
		builder.WriteString(message.Content.String())
		builder.WriteString("\n</")
		builder.WriteString(string(message.Role))
		builder.WriteString(">")
	}
	return builder.String()
}

func testPromptBuilder() gaictx.PromptBuilder {
	return &stubPromptBuilder{
		systemPrompt: "System prompt",
		userPrompt:   "Initial prompt",
	}
}

func TestLoop(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		iterations     []mocks.MockModelResponse
		wantIterations int
		wantError      bool
		maxIterations  int
	}{
		{
			name: "Single iteration",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: "Hello, World!"}, Err: nil},
			},
			wantIterations: 1,
			maxIterations:  8,
		},
		{
			name: "single Tool call",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: `{"id":"call-1","type":"function","name":"echo","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: "How are you?"}, Err: nil},
			},
			wantIterations: 2,
			maxIterations:  8,
		},
		{
			name: "Multiple iterations with tool calls",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: `{"id":"call-1","type":"function","name":"echo","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-2","type":"function","name":"echo","arguments":{"text":"another test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: "How are you?"}, Err: nil},
			},
			wantIterations: 3,
			maxIterations:  8,
		},
		{
			name: "Exceeding max iterations",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: `{"id":"call-1","type":"function","name":"echo","arguments":{"text":"test 1"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-2","type":"function","name":"echo","arguments":{"text":"test 2"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-3","type":"function","name":"echo","arguments":{"text":"test 3"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-4","type":"function","name":"echo","arguments":{"text":"test 4"}}`}, Err: nil},
			},
			wantIterations: 2,
			maxIterations:  2,
			wantError:      true,
		},
		{
			name: "Call wrong tool",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: `{"id":"call-1","type":"function","name":"nonexistent_tool","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: "Tool failed, stopping here."}, Err: nil},
			},
			wantIterations: 2,
			maxIterations:  8,
		},
		{
			name: "No tool calls after response",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: "Just a normal response."}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-1","type":"function","name":"nonexistent_tool","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-2","type":"function","name":"nonexistent_tool","arguments":{"text":"test"}}`}, Err: nil},
			},
			wantIterations: 1,
			maxIterations:  8,
		},
		{
			name: "Tool call with error",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: `{"id":"call-1","type":"function","name":"echo","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-2","type":"function","name":"echo","arguments":{"text":"second test"}}`}, Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("tool execution failed")}},
				{Res: ai.AIResponse{Text: `{"id":"call-3","type":"function","name":"echo","arguments":{"text":"third test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: "How are you?"}, Err: nil},
			},
			wantIterations: 3,
			maxIterations:  8,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model := &mocks.MockModel{}
			model.Responses = tt.iterations
			tools := []loop.Tool{loop.NewEchoTool()}
			l := loop.New(wrapStreamModel{Model: model}, tools, testPromptBuilder(), nil)
			l.MaxLoopIterations = tt.maxIterations
			l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}

			err := loopError(collectLoopEvents(t, l, context.Background()))

			if (err != nil) != tt.wantError {
				t.Fatalf("Loop failed: %v", err)
			}

			if len(l.Iterations) != tt.wantIterations {
				t.Fatalf("Expected %d iteration, got %d", tt.wantIterations, len(l.Iterations))
			}
		})
	}
}

func TestLoopHandlesManyToolCallsInOneIteration(t *testing.T) {
	t.Parallel()

	makeToolCalls := func(t *testing.T, n int, name string) []ai.Token {
		t.Helper()
		calls := make([]ai.Token, 0, n)
		for i := 0; i < n; i++ {
			args, err := json.Marshal(map[string]string{"text": "payload"})
			if err != nil {
				t.Fatalf("marshal args: %v", err)
			}

			calls = append(calls, ai.Token{
				Type: ai.TokenTypeToolCall,
				ToolCall: &ai.ToolCall{
					ID:   fmt.Sprintf("call-%d", i+1),
					Type: "function",
					Name: name,
					Args: args,
				},
			})
		}
		return calls
	}

	tests := []struct {
		name               string
		firstIteration     []ai.Token
		wantFirstParts     int
		wantToolErrors     int
		wantTotalIteration int
	}{
		{
			name:               "Exactly six valid tool calls",
			firstIteration:     makeToolCalls(t, 6, "echo"),
			wantFirstParts:     6,
			wantToolErrors:     0,
			wantTotalIteration: 2,
		},
		{
			name:               "Ten valid tool calls",
			firstIteration:     makeToolCalls(t, 10, "echo"),
			wantFirstParts:     10,
			wantToolErrors:     0,
			wantTotalIteration: 2,
		},
		{
			name:               "Six unknown tool calls produce tool errors",
			firstIteration:     makeToolCalls(t, 6, "unknown_tool"),
			wantFirstParts:     6,
			wantToolErrors:     6,
			wantTotalIteration: 2,
		},
		{
			name: "Mixed text and seven tool calls",
			firstIteration: append(
				[]ai.Token{{Type: ai.TokenTypeText, Data: []byte("prefix")}},
				makeToolCalls(t, 7, "echo")...,
			),
			wantFirstParts:     8,
			wantToolErrors:     0,
			wantTotalIteration: 2,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model := &scriptedStreamModel{
				sequences: [][]ai.Token{
					tt.firstIteration,
					{
						{Type: ai.TokenTypeText, Data: []byte("done")},
					},
				},
			}

			l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
			l.MaxLoopIterations = 3

			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			if err := loopError(collectLoopEvents(t, l, ctx)); err != nil {
				t.Fatalf("unexpected loop error: %v", err)
			}

			if len(l.Iterations) != tt.wantTotalIteration {
				t.Fatalf("expected %d iterations, got %d", tt.wantTotalIteration, len(l.Iterations))
			}

			if got := len(l.Iterations[0].Parts); got != tt.wantFirstParts {
				t.Fatalf("expected %d parts in first iteration, got %d", tt.wantFirstParts, got)
			}

			toolErrs := 0
			for i, part := range l.Iterations[0].Parts {
				if part.Type == loop.IterationTypeToolCall {
					if part.ToolResp == nil {
						t.Fatalf("part %d missing tool response", i)
					}
					if part.ToolResp.Err != nil {
						toolErrs++
					}
				}
			}

			if toolErrs != tt.wantToolErrors {
				t.Fatalf("expected %d tool errors, got %d", tt.wantToolErrors, toolErrs)
			}
		})
	}
}

func TestLoopWrapsToolPreprocessErrors(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{
				{
					Type: ai.TokenTypeToolCall,
					ToolCall: &ai.ToolCall{
						ID:   "call-1",
						Type: "function",
						Name: "echo",
						Args: json.RawMessage(`{"text":"payload"}`),
					},
				},
			},
		},
	}

	l := loop.New(
		model,
		[]loop.Tool{loop.NewEchoTool()},
		testPromptBuilder(),
		failingToolResponseProcessor{err: errors.New("reject tool response")},
	)

	events := collectLoopEvents(t, l, context.Background())
	err := loopError(events)
	if !errors.Is(err, loop.ErrToolResponseProcess) {
		t.Fatalf("error = %v, want ErrToolResponseProcess", err)
	}
	errorEvents := loopEventsOfType(events, loop.EventError)
	if len(errorEvents) != 1 {
		t.Fatalf("expected 1 error event, got %d", len(errorEvents))
	}
	if errorEvents[0].IterationCount != 1 || errorEvents[0].AttemptID != 1 {
		t.Fatalf("expected attempt metadata on error event, got %#v", errorEvents[0])
	}
	if errorEvents[0].Iteration == nil || errorEvents[0].Iteration.UserMessage == nil {
		t.Fatalf("expected failed tool-processing snapshot, got %#v", errorEvents[0].Iteration)
	}
	if len(errorEvents[0].Iteration.Parts) != 1 || errorEvents[0].Iteration.Parts[0].ToolResp == nil {
		t.Fatalf("expected failed tool-processing snapshot to retain tool response, got %#v", errorEvents[0].Iteration)
	}
	if len(l.Iterations) != 0 {
		t.Fatalf("expected preprocess failure to skip persisted iteration, got %d", len(l.Iterations))
	}
}

func TestLoopRetriesDoNotConsumeIterations(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("temporary 1")}}},
			{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("temporary 2")}}},
			{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("temporary 3")}}},
			{{Type: ai.TokenTypeText, Data: []byte("done")}},
		},
	}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 3}

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}

	if len(l.Iterations) != 1 {
		t.Fatalf("expected retry attempts to complete one iteration, got %d", len(l.Iterations))
	}
	if got := len(model.Requests()); got != 4 {
		t.Fatalf("expected 4 model attempts, got %d", got)
	}
	retries := loopEventsOfType(events, loop.EventRetry)
	iterations := loopEventsOfType(events, loop.EventIterationDone)
	if len(retries) != 3 || len(iterations) != 1 {
		t.Fatalf("expected 3 retries and 1 completed iteration, got retries=%d iterations=%d events=%#v", len(retries), len(iterations), events)
	}
	for i := 0; i < 3; i++ {
		event := retries[i]
		if event.IterationCount != 1 {
			t.Fatalf("retry event %d consumed an iteration: %#v", i, event)
		}
		if event.AttemptID != i+1 {
			t.Fatalf("retry event %d expected attempt %d, got %d", i, i+1, event.AttemptID)
		}
		if event.Iteration == nil || event.Iteration.UserMessage == nil {
			t.Fatalf("retry event %d should retain user message: %#v", i, event.Iteration)
		}
	}
	finalEvent := iterations[0]
	if finalEvent.IterationCount != 1 {
		t.Fatalf("expected final iteration count 1, got %d", finalEvent.IterationCount)
	}
	if finalEvent.AttemptID != 4 {
		t.Fatalf("expected final attempt 4, got %d", finalEvent.AttemptID)
	}
	if finalEvent.RetryCount != 3 {
		t.Fatalf("expected final retry count 3, got %d", finalEvent.RetryCount)
	}
	if l.Iterations[0].UserMessage == nil {
		t.Fatal("expected completed first iteration to retain user message")
	}
}

func TestLoopWithoutRetryPolicyDoesNotRetry(t *testing.T) {
	t.Parallel()

	streamErr := errors.New("temporary stream failure")
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Err: streamErr}},
		{{Type: ai.TokenTypeText, Data: []byte("must not be requested")}},
	}}
	l := loop.New(model, nil, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1

	events := collectLoopEvents(t, l, context.Background())
	err := loopError(events)
	if !errors.Is(err, streamErr) {
		t.Fatalf("error = %v, want original stream error", err)
	}
	if errors.Is(err, loop.ErrMaxRetries) {
		t.Fatalf("error = %v, must not report retry exhaustion without a policy", err)
	}
	for _, event := range events {
		if event.Type == loop.EventRetry {
			t.Fatal("nil retry policy must not emit EventRetry")
		}
	}
	if got := len(model.Requests()); got != 1 {
		t.Fatalf("model attempts = %d, want 1 without a retry policy", got)
	}
}

func TestLoopAttemptTimeoutDoesNotApplyToPromptConstruction(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{{{Type: ai.TokenTypeText, Data: []byte("done")}}}}
	promptBuilder := &deadlineRecordingPromptBuilder{stubPromptBuilder: stubPromptBuilder{userPrompt: "user"}}
	l := loop.New(model, nil, promptBuilder, nil)
	l.MaxLoopIterations = 1
	l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1, AttemptTimeout: time.Second}

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	if promptBuilder.hasDeadline.Load() {
		t.Fatal("prompt construction must not receive the model attempt deadline")
	}
}

func TestLoopAttemptTimeoutDoesNotApplyToToolExecution(t *testing.T) {
	t.Parallel()

	tool := &deadlineRecordingTool{}
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{
			Type: ai.TokenTypeToolCall,
			ToolCall: &ai.ToolCall{
				ID:   "call-1",
				Type: "function",
				Name: "record-deadline",
				Args: json.RawMessage(`{"text":"payload"}`),
			},
		}},
		{{Type: ai.TokenTypeText, Data: []byte("done")}},
	}}
	l := loop.New(model, []loop.Tool{tool}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 2
	l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1, AttemptTimeout: time.Second}

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	if tool.hasDeadline.Load() {
		t.Fatal("tool execution must not receive the model attempt deadline")
	}
}

func TestLoopAttemptTimeoutPreservesProviderRetryAfter(t *testing.T) {
	t.Parallel()

	retryAfter := 40 * time.Millisecond
	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, RetryAfter: retryAfter}}},
			{{Type: ai.TokenTypeText, Data: []byte("done")}},
		},
		delays:        []time.Duration{5 * time.Millisecond},
		ignoreContext: true,
	}
	l := loop.New(model, nil, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryPolicy = &loop.RetryPolicy{
		MaxRetries:        1,
		AttemptTimeout:    time.Millisecond,
		RespectRetryAfter: true,
	}

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	times := model.RequestTimes()
	if len(times) != 2 {
		t.Fatalf("expected 2 model attempts, got %d", len(times))
	}
	if elapsed := times[1].Sub(times[0]); elapsed < retryAfter {
		t.Fatalf("retry began after %s, want at least provider RetryAfter %s", elapsed, retryAfter)
	}
}

func TestLoopRetryObservabilityReportsClassificationAndDelay(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	})

	retryDelay := 7 * time.Millisecond
	var observationMu sync.Mutex
	var observations []gai.Observation
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Err: &ai.ProviderError{Kind: ai.ProviderErrorRateLimited}}},
		{{Type: ai.TokenTypeText, Data: []byte("done")}},
	}}
	l := loop.New(model, nil, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryPolicy = &loop.RetryPolicy{
		MaxRetries:     1,
		InitialBackoff: retryDelay,
		Wait:           func(context.Context, time.Duration) error { return nil },
	}
	l.ObservationSink = gai.ObservationSinkFunc(func(_ context.Context, observation gai.Observation) {
		observationMu.Lock()
		defer observationMu.Unlock()
		observations = append(observations, observation)
	})

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	retries := loopEventsOfType(events, loop.EventRetry)
	if len(retries) != 1 {
		t.Fatalf("retry events = %d, want 1", len(retries))
	}
	if retries[0].RetryReason != "rate_limited" || retries[0].RetryDelay != retryDelay {
		t.Fatalf("retry event observability = %#v, want reason rate_limited and delay %s", retries[0], retryDelay)
	}
	observationMu.Lock()
	retryObserved := false
	for _, observation := range observations {
		if observation.Name == "loop_retry_scheduled" && observation.Fields["iteration"] == 1 && observation.Fields["retry_reason"] == "rate_limited" && observation.Fields["retry_delay_ms"] == retryDelay.Milliseconds() {
			retryObserved = true
			break
		}
	}
	observationMu.Unlock()
	if !retryObserved {
		t.Fatalf("retry observation missing iteration, classification, or delay: %#v", observations)
	}

	for _, span := range recorder.Ended() {
		if span.Name() != "loop.iteration" {
			continue
		}
		attributes := map[string]any{}
		for _, attribute := range span.Attributes() {
			attributes[string(attribute.Key)] = attribute.Value.AsInterface()
		}
		if attributes["loop.retry_reason"] == "rate_limited" && attributes["loop.retry_delay_ms"] == int64(retryDelay/time.Millisecond) {
			return
		}
	}
	t.Fatalf("retrying iteration span missing classification or delay: %#v", recorder.Ended())
}

func TestLoopRetryUsesInjectedWait(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient}}},
			{{Type: ai.TokenTypeText, Data: []byte("done")}},
		},
	}
	var waited time.Duration
	l := loop.New(model, nil, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryPolicy = &loop.RetryPolicy{
		MaxRetries:     1,
		InitialBackoff: time.Hour,
		Wait: func(_ context.Context, delay time.Duration) error {
			waited = delay
			return nil
		},
	}

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	if waited != time.Hour {
		t.Fatalf("waited %s, want %s", waited, time.Hour)
	}
}

func TestLoopCancelsFailedAttemptBeforeRetryBackoff(t *testing.T) {
	model := &retryCancellationModel{attemptCanceled: make(chan struct{})}
	l := loop.New(model, nil, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryPolicy = &loop.RetryPolicy{
		MaxRetries:     1,
		InitialBackoff: time.Hour,
		Wait: func(_ context.Context, _ time.Duration) error {
			select {
			case <-model.attemptCanceled:
				return nil
			case <-time.After(100 * time.Millisecond):
				return errors.New("model attempt was still active during retry backoff")
			}
		},
	}

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	if calls := model.calls.Load(); calls != 2 {
		t.Fatalf("model attempts = %d, want 2", calls)
	}
}

func TestLoopTerminalRetryErrorReportsPolicyLimit(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{
		sequences: [][]ai.Token{{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient}}}},
	}
	l := loop.New(model, nil, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 0}

	err := loopError(collectLoopEvents(t, l, context.Background()))
	if !errors.Is(err, loop.ErrMaxRetries) {
		t.Fatalf("error = %v, want ErrMaxRetries", err)
	}
	if !strings.Contains(err.Error(), "limit=0") {
		t.Fatalf("error = %q, want active policy limit", err)
	}
}

func TestLoopNonRetryablePolicyErrorIsNotReportedAsRetryExhaustion(t *testing.T) {
	t.Parallel()

	providerErr := &ai.ProviderError{Kind: ai.ProviderErrorInvalidRequest, Err: errors.New("bad request")}
	model := &scriptedStreamModel{
		sequences: [][]ai.Token{{{Err: providerErr}}},
	}
	l := loop.New(model, nil, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 3}

	err := loopError(collectLoopEvents(t, l, context.Background()))
	if errors.Is(err, loop.ErrMaxRetries) {
		t.Fatalf("error = %v, must not report retry exhaustion", err)
	}
	var got *ai.ProviderError
	if !errors.As(err, &got) || got != providerErr {
		t.Fatalf("error = %v, want original provider error", err)
	}
}

func TestLoopPolicyWithNoRetriesReturnsNonRetryableProviderError(t *testing.T) {
	t.Parallel()

	providerErr := &ai.ProviderError{Kind: ai.ProviderErrorInvalidRequest, Err: errors.New("bad request")}
	model := &scriptedStreamModel{sequences: [][]ai.Token{{{Err: providerErr}}}}
	l := loop.New(model, nil, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 0}

	err := loopError(collectLoopEvents(t, l, context.Background()))
	if errors.Is(err, loop.ErrMaxRetries) {
		t.Fatalf("error = %v, must not report retry exhaustion", err)
	}
	var got *ai.ProviderError
	if !errors.As(err, &got) || got != providerErr {
		t.Fatalf("error = %v, want original provider error", err)
	}
}

func TestLoopPolicyReturnsNonRetryableErrorAfterBudgetIsConsumed(t *testing.T) {
	t.Parallel()

	providerErr := &ai.ProviderError{Kind: ai.ProviderErrorAuthentication, Err: errors.New("invalid credentials")}
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient}}},
		{{Err: providerErr}},
	}}
	l := loop.New(model, nil, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}

	err := loopError(collectLoopEvents(t, l, context.Background()))
	if errors.Is(err, loop.ErrMaxRetries) {
		t.Fatalf("error = %v, must not report retry exhaustion", err)
	}
	var got *ai.ProviderError
	if !errors.As(err, &got) || got != providerErr {
		t.Fatalf("error = %v, want original provider error", err)
	}
}

func TestLoopStreamErrorsIncludeAttemptMetadata(t *testing.T) {
	t.Parallel()

	fatalErr := errors.New("fatal stream error")
	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{{Err: fatalErr}},
		},
	}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1

	events := collectLoopEvents(t, l, context.Background())
	err := loopError(events)
	if !errors.Is(err, fatalErr) || errors.Is(err, loop.ErrMaxRetries) {
		t.Fatalf("error = %v, want original non-retry error", err)
	}
	errorEvents := loopEventsOfType(events, loop.EventError)
	if len(errorEvents) != 1 {
		t.Fatalf("expected 1 error event, got %d", len(errorEvents))
	}
	if errorEvents[0].IterationCount != 1 || errorEvents[0].AttemptID != 1 || errorEvents[0].RetryCount != 0 {
		t.Fatalf("expected attempt metadata on error event, got %#v", errorEvents[0])
	}
	if errorEvents[0].Iteration == nil || errorEvents[0].Iteration.UserMessage == nil {
		t.Fatalf("expected failed attempt snapshot to retain user message, got %#v", errorEvents[0].Iteration)
	}
	if !errors.Is(errorEvents[0].Err, fatalErr) || errors.Is(errorEvents[0].Err, loop.ErrMaxRetries) {
		t.Fatalf("expected original terminal error, got %q", errorEvents[0].Err)
	}
}

func TestLoopTerminalStreamErrorIncludesPartialAttemptIteration(t *testing.T) {
	t.Parallel()

	fatalErr := errors.New("fatal stream error")
	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{
				{Type: ai.TokenTypeText, Data: []byte("partial")},
				{Err: fatalErr},
			},
		},
	}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1

	events := collectLoopEvents(t, l, context.Background())
	err := loopError(events)
	if !errors.Is(err, fatalErr) || errors.Is(err, loop.ErrMaxRetries) {
		t.Fatalf("error = %v, want original non-retry error", err)
	}
	errorEvents := loopEventsOfType(events, loop.EventError)
	if len(errorEvents) != 1 {
		t.Fatalf("expected 1 error event, got %d", len(errorEvents))
	}
	errorEvent := errorEvents[0]
	if errorEvent.PartCount != 1 {
		t.Fatalf("expected partial attempt part count 1, got %d", errorEvent.PartCount)
	}
	if errorEvent.Iteration == nil {
		t.Fatal("expected partial attempt iteration on error event")
	}
	if got := errorEvent.Iteration.Parts[0].Response.Text; got != "partial" {
		t.Fatalf("expected partial attempt text, got %q", got)
	}
	if len(l.Iterations) != 0 {
		t.Fatalf("expected terminal stream failure to skip persisted iteration, got %d", len(l.Iterations))
	}
}

func TestLoopRetryStatusMarksPartialTokensDiscardable(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{
				{Type: ai.TokenTypeText, Data: []byte("partial")},
				{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("temporary")}},
			},
			{
				{Type: ai.TokenTypeText, Data: []byte("final")},
			},
		},
	}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}

	tokenEvents := loopEventsOfType(events, loop.EventToken)
	if len(tokenEvents) != 2 {
		t.Fatalf("expected partial and final token to stream, got %d", len(tokenEvents))
	}
	if string(tokenEvents[0].Token.Data) != "partial" || string(tokenEvents[1].Token.Data) != "final" {
		t.Fatalf("unexpected token events: %#v", tokenEvents)
	}
	if tokenEvents[0].AttemptID != 1 || tokenEvents[1].AttemptID != 2 {
		t.Fatalf("expected token attempt IDs 1 and 2, got %#v", tokenEvents)
	}
	retries := loopEventsOfType(events, loop.EventRetry)
	iterations := loopEventsOfType(events, loop.EventIterationDone)
	if len(retries) != 1 || len(iterations) != 1 {
		t.Fatalf("expected retry and final iteration events, got retries=%d iterations=%d", len(retries), len(iterations))
	}
	retryEvent := retries[0]
	if retryEvent.AttemptID != 1 {
		t.Fatalf("expected retry event for attempt 1, got %#v", retryEvent)
	}
	if retryEvent.PartCount != 1 {
		t.Fatalf("expected retry event to report partial attempt part count, got %d", retryEvent.PartCount)
	}
	if got := l.Iterations[0].Parts[0].Response.Text; got != "final" {
		t.Fatalf("expected persisted iteration to use successful attempt only, got %q", got)
	}
}

func TestLoopDoesNotRetryCanceledStream(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{{Err: context.Canceled}},
			{{Type: ai.TokenTypeText, Data: []byte("should not run")}},
		},
	}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("cancellation should not be an error event, got %v", err)
	}
	canceled := loopEventsOfType(events, loop.EventCanceled)
	if len(canceled) != 1 || !errors.Is(canceled[0].Err, context.Canceled) {
		t.Fatalf("expected one context.Canceled event, got %#v", canceled)
	}
	if got := len(model.Requests()); got != 1 {
		t.Fatalf("expected no retry after cancellation, got %d model requests", got)
	}
	if retries := loopEventsOfType(events, loop.EventRetry); len(retries) != 0 {
		t.Fatalf("expected no retry events after cancellation, got %#v", retries)
	}
}

func TestLoopCancelsWhenStreamClosesAfterContextCancellation(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	model := &cancelAfterTokenModel{cancel: cancel}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1

	events := collectLoopEvents(t, l, ctx)
	if err := loopError(events); err != nil {
		t.Fatalf("cancellation should not be an error event, got %v", err)
	}
	if done := loopEventsOfType(events, loop.EventDone); len(done) != 0 {
		t.Fatalf("canceled stream should not finish successfully, got %#v", done)
	}
	canceled := loopEventsOfType(events, loop.EventCanceled)
	if len(canceled) != 1 {
		t.Fatalf("expected one cancellation event, got %#v", events)
	}
	event := canceled[0]
	if !errors.Is(event.Err, context.Canceled) || event.Iteration == nil {
		t.Fatalf("expected canceled partial attempt, got %#v", event)
	}
	if event.Iteration.UserMessage == nil || len(event.Iteration.Parts) != 1 {
		t.Fatalf("expected canceled snapshot with user message and partial token, got %#v", event.Iteration)
	}
	if len(l.Iterations) != 0 {
		t.Fatalf("expected canceled partial attempt not to persist, got %#v", l.Iterations)
	}
}

func TestLoopAppendsIterationMessagesToIncrementalPrompt(t *testing.T) {
	t.Parallel()

	var buildCount atomic.Int32
	promptBuilder := &stubPromptBuilder{
		systemPrompt: "System prompt",
		userPrompt:   "Initial prompt",
		buildContext: func() string {
			count := buildCount.Add(1)
			return fmt.Sprintf("build-%d", count)
		},
	}

	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{
				{
					Type: ai.TokenTypeToolCall,
					ToolCall: &ai.ToolCall{
						ID:   "call-1",
						Type: "function",
						Name: "echo",
						Args: json.RawMessage(`{"text":"payload"}`),
					},
				},
			},
			{
				{Type: ai.TokenTypeText, Data: []byte("done")},
			},
		},
	}

	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, promptBuilder, nil)
	l.MaxLoopIterations = 3

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}

	if got := buildCount.Load(); got != 1 {
		t.Fatalf("expected incremental prompt builder to build sources once, got %d", got)
	}
	requests := model.Requests()
	if len(requests) != 2 {
		t.Fatalf("expected 2 model requests, got %d", len(requests))
	}
	if !strings.Contains(requests[0].Prompt, "System prompt") {
		t.Fatalf("expected system prompt in first request: %q", requests[0].Prompt)
	}
	if !strings.Contains(requests[0].Prompt, "build-1") || !strings.Contains(requests[1].Prompt, "build-1") {
		t.Fatalf("expected dynamic context to be reused: first=%q second=%q", requests[0].Prompt, requests[1].Prompt)
	}
	if !strings.Contains(requests[0].Prompt, "Initial prompt") {
		t.Fatalf("expected user prompt in first request: %q", requests[0].Prompt)
	}
	if strings.Contains(requests[0].Prompt, "payload") {
		t.Fatalf("first request should not contain future tool delta: %q", requests[0].Prompt)
	}
	if !strings.Contains(requests[1].Prompt, "payload") {
		t.Fatalf("second request should include appended tool delta: %q", requests[1].Prompt)
	}
	if len(l.Iterations) != 2 {
		t.Fatalf("expected 2 stored iterations, got %d", len(l.Iterations))
	}
	if l.Iterations[0].UserMessage == nil {
		t.Fatal("expected first stored iteration to retain user message")
	}
	if l.Iterations[1].UserMessage != nil {
		t.Fatalf("expected later stored iterations to omit user message, got %#v", l.Iterations[1].UserMessage)
	}
}

func TestLoopFallsBackToBuildPromptEveryIteration(t *testing.T) {
	t.Parallel()

	promptBuilder := &countingPromptBuilder{}
	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{
				{
					Type: ai.TokenTypeToolCall,
					ToolCall: &ai.ToolCall{
						ID:   "call-1",
						Type: "function",
						Name: "echo",
						Args: json.RawMessage(`{"text":"payload"}`),
					},
				},
			},
			{
				{Type: ai.TokenTypeText, Data: []byte("done")},
			},
		},
	}

	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, promptBuilder, nil)
	l.MaxLoopIterations = 3

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}

	if got := promptBuilder.count.Load(); got != 2 {
		t.Fatalf("expected prompt builder to render once per iteration, got %d", got)
	}
	requests := model.Requests()
	if len(requests) != 2 {
		t.Fatalf("expected 2 model requests, got %d", len(requests))
	}
	if requests[0].Prompt != "prompt-1" || requests[1].Prompt != "prompt-2" {
		t.Fatalf("expected rebuilt prompts, got first=%q second=%q", requests[0].Prompt, requests[1].Prompt)
	}
	for index, request := range requests {
		if len(request.Messages) != 0 {
			t.Fatalf("request %d expected rendered-prompt fallback, got native messages %#v", index, request.Messages)
		}
		if len(request.Tools) != 1 {
			t.Fatalf("request %d expected 1 tool definition, got %d", index, len(request.Tools))
		}
		if request.Tools[0].Name != "echo" {
			t.Fatalf("request %d expected echo tool definition, got %#v", index, request.Tools[0])
		}
	}
}

func TestLoopToolTransportControlsProviderToolDefinitions(t *testing.T) {
	tests := []struct {
		name       string
		transport  loop.ToolTransportMode
		wantTools  bool
		wantChoice bool
	}{
		{name: "default native transport", wantTools: true, wantChoice: true},
		{name: "text transport", transport: loop.ToolTransportText},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := &scriptedStreamModel{sequences: [][]ai.Token{{{Type: ai.TokenTypeText, Text: "done"}}}}
			l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
			l.ToolTransport = tt.transport
			if tt.wantChoice {
				l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired}
			}

			if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
				t.Fatalf("unexpected loop error: %v", err)
			}
			requests := model.Requests()
			if len(requests) != 1 {
				t.Fatalf("requests = %d, want 1", len(requests))
			}
			if got := len(requests[0].Tools) > 0; got != tt.wantTools {
				t.Fatalf("request tools = %#v, want present=%t", requests[0].Tools, tt.wantTools)
			}
			if got := requests[0].ToolChoice.Mode != ""; got != tt.wantChoice {
				t.Fatalf("request tool choice = %#v, want present=%t", requests[0].ToolChoice, tt.wantChoice)
			}
			if err := requests[0].Validate(); err != nil {
				t.Fatalf("request must validate: %v", err)
			}
			if len(l.Tools) != 1 || l.Tools[0].Name() != "echo" {
				t.Fatalf("loop lost executable tools: %#v", l.Tools)
			}
		})
	}
}

func TestLoopNeutralToolChoiceWithoutToolsIsTransportIndependent(t *testing.T) {
	for _, tt := range []struct {
		name      string
		transport loop.ToolTransportMode
		choice    ai.ToolChoiceMode
	}{
		{name: "native auto", choice: ai.ToolChoiceAuto},
		{name: "native none", choice: ai.ToolChoiceNone},
		{name: "text auto", transport: loop.ToolTransportText, choice: ai.ToolChoiceAuto},
		{name: "text none", transport: loop.ToolTransportText, choice: ai.ToolChoiceNone},
	} {
		t.Run(tt.name, func(t *testing.T) {
			model := &scriptedStreamModel{sequences: [][]ai.Token{{{Type: ai.TokenTypeText, Text: "done"}}}}
			l := loop.New(model, nil, testPromptBuilder(), nil)
			l.ToolTransport = tt.transport
			l.ToolChoice = ai.ToolChoice{Mode: tt.choice}

			if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
				t.Fatalf("unexpected loop error: %v", err)
			}
			requests := model.Requests()
			if len(requests) != 1 {
				t.Fatalf("requests = %d, want 1", len(requests))
			}
			if err := requests[0].Validate(); err != nil {
				t.Fatalf("request must validate: %v", err)
			}
			if requests[0].ToolChoice.Mode != "" {
				t.Fatalf("request tool choice = %#v, want empty without tools", requests[0].ToolChoice)
			}
		})
	}
}

func TestLoopNativeHistoryIncludesBaseRequestWithoutRenderedHistory(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"payload"}`), ThoughtSignature: []byte("opaque-thought-signature")}}},
		{{Type: ai.TokenTypeText, Data: []byte("done")}},
	}}
	promptBuilder := &stubPromptBuilder{systemPrompt: "system", userPrompt: "user"}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, promptBuilder, nil)
	l.MaxLoopIterations = 3

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	requests := model.Requests()
	if len(requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(requests))
	}
	second := requests[1]
	if len(second.Messages) != 3 {
		t.Fatalf("native messages = %#v, want base user plus assistant/tool history", second.Messages)
	}
	if base := second.Messages[0]; base.Role != ai.RequestMessageRoleUser || base.Text != "system\nuser\n" {
		t.Fatalf("base native message = %#v", base)
	}
	if strings.Contains(second.Messages[0].Text, "payload") {
		t.Fatalf("base native message duplicated rendered history: %q", second.Messages[0].Text)
	}
	if second.Messages[1].Role != ai.RequestMessageRoleAssistant || len(second.Messages[1].ToolCalls) != 1 || second.Messages[2].Role != ai.RequestMessageRoleTool {
		t.Fatalf("native history = %#v", second.Messages)
	}
	if got := string(second.Messages[1].ToolCalls[0].ThoughtSignature); got != "opaque-thought-signature" {
		t.Fatalf("thought signature = %q", got)
	}
	if !strings.Contains(second.Prompt, "payload") {
		t.Fatalf("complete rendered fallback omitted tool history: %q", second.Prompt)
	}
}

func TestLoopNativeRequestRendersInitialPromptOnce(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"payload"}`)}}},
		{{Type: ai.TokenTypeText, Data: []byte("done")}},
	}}
	renderer := &gaictx.SimpleRenderer{}
	var rendered []string
	if err := renderer.SetRenderResultCallback(context.Background(), func(_ []gaictx.Part, prompt string) {
		rendered = append(rendered, prompt)
	}); err != nil {
		t.Fatalf("SetRenderResultCallback failed: %v", err)
	}
	promptBuilder := gaictx.New(gaictx.Definition{
		Renderer:           renderer,
		SystemInstructions: []gaictx.Part{gaictx.NewTextPart("system")},
		PromptInput:        gaictx.PromptInput{User: gaictx.NewTextContent("user")},
	})
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, promptBuilder, nil)
	l.MaxLoopIterations = 3

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	if len(rendered) != 3 {
		t.Fatalf("render callbacks = %d, want one initial render and two distinct follow-up renders: %#v", len(rendered), rendered)
	}
	if rendered[0] != model.Requests()[0].Prompt {
		t.Fatalf("initial callback prompt = %q, want request prompt %q", rendered[0], model.Requests()[0].Prompt)
	}
	requests := model.Requests()
	if len(requests) != 2 || len(requests[0].Messages) != 1 || requests[0].Messages[0].Text != requests[0].Prompt {
		t.Fatalf("initial native request = %#v, want its sole native message to reuse the compatibility prompt", requests)
	}
	if !strings.Contains(requests[1].Prompt, "payload") || strings.Contains(requests[1].Messages[0].Text, "payload") {
		t.Fatalf("follow-up request must retain rendered fallback history and separate native base: %#v", requests[1])
	}
}

func TestLoopBuildsBasePromptWithNonNilConversation(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{{{Type: ai.TokenTypeText, Data: []byte("done")}}}}
	promptBuilder := &nonNilConversationPromptBuilder{stubPromptBuilder: stubPromptBuilder{systemPrompt: "system", userPrompt: "user"}}
	l := loop.New(model, nil, promptBuilder, nil)

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	requests := model.Requests()
	if len(requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(requests))
	}
	if len(requests[0].Messages) != 1 || requests[0].Messages[0].Text != "system\nuser\n" {
		t.Fatalf("base native message = %#v", requests[0].Messages)
	}
}

func TestLoopNativeHistoryGroupsParallelToolCallsInOneAssistantMessage(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{
			{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"first"}`)}},
			{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-2", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"second"}`)}},
		},
		{{Type: ai.TokenTypeText, Data: []byte("done")}},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, &stubPromptBuilder{systemPrompt: "system", userPrompt: "user"}, nil)
	l.MaxLoopIterations = 3

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	requests := model.Requests()
	if len(requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(requests))
	}
	second := requests[1]
	if len(second.Messages) != 4 {
		t.Fatalf("native messages = %#v, want base user, one assistant tool-call turn, and two results", second.Messages)
	}
	if assistant := second.Messages[1]; assistant.Role != ai.RequestMessageRoleAssistant || len(assistant.ToolCalls) != 2 {
		t.Fatalf("assistant tool-call turn = %#v, want two tool calls", assistant)
	}
	for i, message := range second.Messages[2:] {
		if message.Role != ai.RequestMessageRoleTool {
			t.Fatalf("tool result message %d = %#v", i, message)
		}
	}
}

func TestLoopNativeHistoryKeepsMixedTextAndToolCallsInOneAssistantMessage(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{
			{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"payload"}`)}},
			{Type: ai.TokenTypeText, Data: []byte("calling echo")},
		},
		{{Type: ai.TokenTypeText, Data: []byte("done")}},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, &stubPromptBuilder{systemPrompt: "system", userPrompt: "user"}, nil)
	l.MaxLoopIterations = 3

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}
	requests := model.Requests()
	if len(requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(requests))
	}
	second := requests[1]
	if len(second.Messages) != 3 {
		t.Fatalf("native messages = %#v, want base user, one mixed assistant turn, and one result", second.Messages)
	}
	assistant := second.Messages[1]
	if assistant.Role != ai.RequestMessageRoleAssistant || assistant.Text != "calling echo" || len(assistant.ToolCalls) != 1 {
		t.Fatalf("mixed assistant turn = %#v", assistant)
	}
	if result := second.Messages[2]; result.Role != ai.RequestMessageRoleTool || result.ToolResult == nil || result.ToolResult.ToolCallID != "call-1" {
		t.Fatalf("tool result = %#v", result)
	}
}

func TestIterationCountsLeadingThoughtTokens(t *testing.T) {
	t.Parallel()

	var iteration loop.Iteration
	iteration.AppendToken(ai.Token{Type: ai.TokenTypeThought, Text: "thinking", TokenUsage: 7})
	iteration.AppendToken(ai.Token{Type: ai.TokenTypeThought, Text: " more", TokenUsage: 3})
	iteration.AppendToken(ai.Token{Type: ai.TokenTypeText, Text: "answer", TokenUsage: 2})

	if len(iteration.Parts) != 1 {
		t.Fatalf("expected one response part, got %d", len(iteration.Parts))
	}
	response := iteration.Parts[0].Response
	if response == nil {
		t.Fatal("expected response part")
	}
	if response.Text != "answer" {
		t.Fatalf("unexpected visible text: %q", response.Text)
	}
	if response.Reasoning != "thinking more" {
		t.Fatalf("unexpected reasoning: %q", response.Reasoning)
	}
	if response.ReasoningTokens != 10 {
		t.Fatalf("expected reasoning tokens to include leading thought, got %d", response.ReasoningTokens)
	}
	if response.OutputTokens != 12 {
		t.Fatalf("unexpected output tokens: %d", response.OutputTokens)
	}
}

func TestIterationCompletionUsageUsesLatestProviderValues(t *testing.T) {
	t.Parallel()

	var iteration loop.Iteration
	iteration.AppendToken(ai.Token{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{
		Usage: ai.Usage{InputTokens: 10, OutputTokens: 4, ReasoningTokens: 2},
	}})
	iteration.AppendToken(ai.Token{Type: ai.TokenTypeCompletion, Completion: &ai.Completion{
		Usage: ai.Usage{InputTokens: 12, OutputTokens: 6, ReasoningTokens: 3},
	}})

	if iteration.Usage != (ai.Usage{InputTokens: 12, OutputTokens: 6, ReasoningTokens: 3}) {
		t.Fatalf("completion usage should use the latest provider values, got %#v", iteration.Usage)
	}
}

func TestLoopCreatesToolSpans(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	})

	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"payload"}`)}}},
		{{Type: ai.TokenTypeText, Text: "done"}},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	ctx := gai.WithContentCapturePolicy(context.Background(), gai.ContentCapturePolicy{
		ToolInput: gai.CaptureEnabled, ToolOutput: gai.CaptureEnabled,
		Redact: func(_ context.Context, _ gai.ContentKind, value []byte) ([]byte, error) {
			return []byte(strings.ReplaceAll(string(value), "payload", "[redacted]")), nil
		},
	})
	if err := loopError(collectLoopEvents(t, l, ctx)); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}

	for _, span := range recorder.Ended() {
		if span.Name() != "loop.tool" {
			continue
		}
		attributes := map[string]any{}
		for _, attribute := range span.Attributes() {
			attributes[string(attribute.Key)] = attribute.Value.AsInterface()
		}
		if attributes["tool.name"] != "echo" || attributes["tool.call_id"] != "call-1" || attributes["tool.status"] != "success" {
			t.Fatalf("unexpected tool span attributes: %#v", attributes)
		}
		if attributes["tool.input"] != `{"text":"[redacted]"}` || attributes["tool.output"] != "[redacted]" {
			t.Fatalf("tool span did not use the content policy: %#v", attributes)
		}
		if attributes["tool.input.redaction_applied"] != true || attributes["tool.output.redaction_applied"] != true {
			t.Fatalf("tool span is missing redaction metadata: %#v", attributes)
		}
		if attributes["gen_ai.operation.name"] != "execute_tool" || attributes["langfuse.observation.type"] != "tool" {
			t.Fatalf("tool span is missing semantic attributes: %#v", attributes)
		}
		for _, candidate := range recorder.Ended() {
			if candidate.Name() == "loop.iteration" && candidate.SpanContext().SpanID() == span.Parent().SpanID() {
				return
			}
		}
		t.Fatalf("tool span parent %s is not a loop.iteration span", span.Parent().SpanID())
	}
	t.Fatalf("expected loop.tool span, got %#v", recorder.Ended())
}

func TestLoopToolErrorDoesNotLeakIntoSpan(t *testing.T) {
	tests := []struct {
		name         string
		ctx          func() context.Context
		wantRedacted bool
	}{
		{name: "default policy", ctx: context.Background},
		{
			name: "enabled with redaction",
			ctx: func() context.Context {
				return gai.WithContentCapturePolicy(context.Background(), gai.ContentCapturePolicy{
					ToolOutput: gai.CaptureEnabled,
					Redact: func(_ context.Context, _ gai.ContentKind, value []byte) ([]byte, error) {
						return []byte(strings.ReplaceAll(string(value), "sentinel-secret", "[redacted]")), nil
					},
				})
			},
			wantRedacted: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			previousProvider := otel.GetTracerProvider()
			recorder := tracetest.NewSpanRecorder()
			provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
			otel.SetTracerProvider(provider)
			defer func() {
				_ = provider.Shutdown(context.Background())
				otel.SetTracerProvider(previousProvider)
			}()

			model := &scriptedStreamModel{sequences: [][]ai.Token{
				{{Type: ai.TokenTypeToolCall, ToolCall: &ai.ToolCall{ID: "call-error", Type: "function", Name: "failure", Args: json.RawMessage(`{"text":"payload"}`)}}},
				{{Type: ai.TokenTypeText, Text: "done"}},
			}}
			l := loop.New(model, []loop.Tool{sentinelErrorTool{}}, testPromptBuilder(), nil)
			if err := loopError(collectLoopEvents(t, l, tt.ctx())); err != nil {
				t.Fatalf("unexpected loop error: %v", err)
			}

			var observed strings.Builder
			for _, span := range recorder.Ended() {
				if span.Name() != "loop.tool" {
					continue
				}
				observed.WriteString(span.Status().Description)
				for _, attr := range span.Attributes() {
					observed.WriteString(string(attr.Key))
					observed.WriteString(attr.Value.String())
				}
				for _, event := range span.Events() {
					observed.WriteString(event.Name)
					for _, attr := range event.Attributes {
						observed.WriteString(string(attr.Key))
						observed.WriteString(attr.Value.String())
					}
				}
			}
			if strings.Contains(observed.String(), "sentinel-secret") {
				t.Fatalf("raw tool error reached OTel: %s", observed.String())
			}
			if !tt.wantRedacted && strings.Contains(observed.String(), "tool-output-") {
				t.Fatalf("disabled tool output reached OTel: %s", observed.String())
			}
			if got := strings.Contains(observed.String(), "[redacted]"); got != tt.wantRedacted {
				t.Fatalf("redacted tool output presence = %v, want %v: %s", got, tt.wantRedacted, observed.String())
			}
		})
	}
}

func TestIterationDeltaMessagesSkipsThoughtOnlyResponses(t *testing.T) {
	t.Parallel()

	var iteration loop.Iteration
	iteration.AppendToken(ai.Token{Type: ai.TokenTypeThought, Text: "thinking"})

	if messages := iteration.DeltaMessages(); len(messages) != 0 {
		t.Fatalf("expected no messages for thought-only response, got %#v", messages)
	}

	iteration.AppendToken(ai.Token{Type: ai.TokenTypeText, Text: "answer"})

	messages := iteration.DeltaMessages()
	if len(messages) != 1 {
		t.Fatalf("expected one assistant message after visible text, got %#v", messages)
	}
	if messages[0].Role != gaictx.RoleAssistant {
		t.Fatalf("expected assistant role, got %q", messages[0].Role)
	}
	if got := messages[0].Content.String(); got != "answer" {
		t.Fatalf("unexpected assistant content: %q", got)
	}
}
