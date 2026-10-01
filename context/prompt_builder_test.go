package context

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

type testContextSource struct {
	name   string
	budget int
	text   string
}

func (s *testContextSource) Name() string {
	return s.name
}

func (s *testContextSource) Function(ctx context.Context, tokenBudget int) (Part, error) {
	s.budget = tokenBudget
	return NewTextPart(s.text), nil
}

type emptyConversation struct{}

func (emptyConversation) Messages() []ai.Message {
	return nil
}

type debugTestTokenCounter struct{}

func (debugTestTokenCounter) Fidelity() ai.TokenCountFidelity { return ai.TokenCountEstimated }

func (debugTestTokenCounter) ID() string {
	return "debug.test"
}

func (debugTestTokenCounter) Tokenize(ctx context.Context, text string) ([]string, error) {
	return strings.Fields(text), nil
}

func (debugTestTokenCounter) CountTokens(ctx context.Context, text string) (int, error) {
	return len(strings.Fields(text)), nil
}

func TestNewPromptBuilderFromDefinition(t *testing.T) {
	t.Parallel()

	source := &testContextSource{name: "source", text: "context"}
	builder := New(Definition{
		SystemInstructions: []Part{NewTextPart("system")},
		ContextSources:     []ContextSource{source},
		PromptInput:        PromptInput{User: ai.TextParts("user")},
		TokenBudget:        12,
	})

	if builder.Renderer == nil {
		t.Fatal("expected default renderer")
	}
	if got := builder.Input().User[0].Text; got != "user" {
		t.Fatalf("expected user prompt %q, got %q", "user", got)
	}

	_, err := builder.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	if source.budget != 10 {
		t.Fatalf("expected source token budget 10 after estimating system instructions, got %d", source.budget)
	}

	prompt, err := renderBuilderRequest(builder, context.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("render request failed: %v", err)
	}

	systemIndex := strings.Index(prompt, "system")
	contextIndex := strings.Index(prompt, "context")
	userIndex := strings.LastIndex(prompt, "user\n")
	if systemIndex < 0 || contextIndex < 0 || userIndex < 0 {
		t.Fatalf("expected prompt to contain system, context, and user prompt: %q", prompt)
	}
	if !(systemIndex < contextIndex && contextIndex < userIndex) {
		t.Fatalf("expected system, context, user prompt order: %q", prompt)
	}
}

type messageConversation struct {
	messages []ai.Message
}

func (c messageConversation) Messages() []ai.Message {
	return c.messages
}

func TestBuildRequestPreservesRolesAndCanonicalConversation(t *testing.T) {
	t.Parallel()
	builder := New(Definition{SystemInstructions: []Part{NewTextPart("system")}, PromptInput: PromptInput{User: ai.TextParts("question")}})
	request, err := builder.BuildRequest(t.Context(), messageConversation{messages: []ai.Message{ai.TextMessage(ai.RoleAssistant, "answer")}})
	if err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 3 {
		t.Fatalf("messages = %#v", request.Messages)
	}
	for i, role := range []ai.Role{ai.RoleSystem, ai.RoleUser, ai.RoleAssistant} {
		if request.Messages[i].Role != role {
			t.Fatalf("message %d role = %q, want %q", i, request.Messages[i].Role, role)
		}
	}
	if !strings.Contains(request.Messages[0].Text(), "system") || request.Messages[1].Text() != "question" || request.Messages[2].Text() != "answer" {
		t.Fatalf("messages = %#v", request.Messages)
	}
	prompt, err := renderBuilderRequest(builder, t.Context(), messageConversation{messages: []ai.Message{ai.TextMessage(ai.RoleAssistant, "answer")}})
	if err != nil {
		t.Fatal(err)
	}
	want, err := ai.RenderMessages(t.Context(), request.Messages)
	if err != nil || prompt != want {
		t.Fatalf("fallback = %q, want canonical projection %q: %v", prompt, want, err)
	}
}

func TestRenderRequestPreservesStructuredConversationContent(t *testing.T) {
	t.Parallel()

	builder := New(Definition{
		SystemInstructions: []Part{NewTextPart("system")},
		PromptInput:        PromptInput{User: ai.TextParts("find docs")},
	})

	prompt, err := renderBuilderRequest(builder, context.Background(), messageConversation{
		messages: []ai.Message{
			{
				Role: ai.RoleAssistant, Parts: []ai.ContentPart{
					{Kind: ai.ContentToolCall,
						ToolCall: &ai.ToolCall{ID: "call_search",
							Type: "function", Name: "search", Args: []byte(`{"q":"lace"}`)}}},
			},
			{
				Role: ai.RoleTool, Parts: []ai.ContentPart{
					{Kind: ai.ContentToolResult,
						ToolResult: &ai.ToolResult{ToolCallID: "call_search", Name: "search", Parts: ai.TextParts("found <docs>")}}},
			},
		},
	})
	if err != nil {
		t.Fatalf("render request failed: %v", err)
	}

	expected := []string{
		`<user>`,
		`find docs`,
		`</user>`,
		`<assistant>`,
		`<tool_call id="call_search" name="search">`,
		`{&#34;q&#34;:&#34;lace&#34;}`,
		`<tool>`,
		`<tool_result id="call_search" name="search" is_error="false">`,
		`found &lt;docs&gt;`,
	}
	for _, fragment := range expected {
		if !strings.Contains(prompt, fragment) {
			t.Fatalf("expected prompt to contain %q:\n%s", fragment, prompt)
		}
	}
	rejected := []string{
		`<message role=`,
		`<user><text>`,
		`assistant: search`,
		`tool: search result`,
		`{&amp;#34;`,
		`Precomputed`,
		`cached`,
	}
	for _, fragment := range rejected {
		if strings.Contains(prompt, fragment) {
			t.Fatalf("expected prompt not to contain %q:\n%s", fragment, prompt)
		}
	}
}

func TestRenderRequestOrdersInputContextBeforeUserAndConversation(t *testing.T) {
	t.Parallel()

	observation, err := NewJSONPart("memory_observation", map[string]string{"fact": "stable"})
	if err != nil {
		t.Fatalf("NewJSONPart failed: %v", err)
	}
	builder := New(Definition{
		Renderer:           &SimpleRenderer{},
		SystemInstructions: []Part{NewTextPart("system")},
		ContextSources:     []ContextSource{&testContextSource{name: "source", text: "configured context"}},
		PromptInput: PromptInput{
			User: ai.TextParts("current user"),

			Context: []Part{observation},
		},
	})
	if _, err := builder.BuildContext(t.Context()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := renderBuilderRequest(builder, t.Context(), messageConversation{messages: []ai.Message{{Role: ai.RoleAssistant, Parts: ai.TextParts("assistant delta")}}})
	if err != nil {
		t.Fatalf("render request failed: %v", err)
	}

	ordered := []string{"system", "configured context", "memory_observation", "current user", "assistant delta"}
	previous := -1
	for _, fragment := range ordered {
		index := strings.Index(prompt, fragment)
		if index <= previous {
			t.Fatalf("prompt does not preserve structured input order at %q: %s", fragment, prompt)
		}
		previous = index
	}
}

type debugEventSink struct {
	events []gai.Observation
}

func (s *debugEventSink) Emit(ctx context.Context, e gai.Observation) {
	s.events = append(s.events, e)
}

type failingPart struct{}

func (failingPart) Name() string {
	return "failing"
}

func (failingPart) Tokens(ctx context.Context, counter ai.TokenCounter) (int, error) {
	return 0, errors.New("token count failed")
}

func (failingPart) Render(ctx context.Context) (RenderNode, error) {
	return RenderNode{}, errors.New("render failed")
}

func TestPromptBuilderEmitsExistingEventsWithoutSensitiveFieldsByDefault(t *testing.T) {
	t.Parallel()

	sink := &debugEventSink{}
	source := &testContextSource{name: "docs", text: "context"}
	builder := New(Definition{
		SystemInstructions: []Part{NewTextPart("system prompt")},
		ContextSources:     []ContextSource{source},
		PromptInput:        PromptInput{User: ai.TextParts("find docs")},
		TokenBudget:        10,
		ObservationSink:    sink,
	})
	builder.SetTokenCounter(debugTestTokenCounter{})

	if _, err := builder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	if _, err := renderBuilderRequest(builder, context.Background(), emptyConversation{}); err != nil {
		t.Fatalf("render request failed: %v", err)
	}

	var names []string
	for _, event := range sink.events {
		names = append(names, event.Name)
	}
	want := []string{
		"prompt_builder_context_build_started",
		"prompt_builder_source_included",
		"prompt_builder_context_build_finished",
		"renderer_render_started",
		"renderer_part_rendered",
		"renderer_render_finished",
		"renderer_render_started",
		"renderer_part_rendered",
		"renderer_render_finished",
	}
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("unexpected event names: got %v want %v", names, want)
	}

	renderEvent := sink.events[len(sink.events)-1]
	if _, ok := renderEvent.Fields["prompt"]; ok {
		t.Fatalf("expected prompt field to be omitted without sensitive debug")
	}
	if _, ok := renderEvent.Fields["prompt_structure"]; ok {
		t.Fatalf("expected prompt_structure field to be omitted without sensitive debug")
	}
}

func TestPromptBuilderSetObservationSinkUpdatesDefaultRenderer(t *testing.T) {
	ctx := gai.WithContentCapturePolicy(context.Background(), gai.ContentCapturePolicy{Prompt: gai.CaptureEnabled})

	t.Run("replacement", func(t *testing.T) {
		original := &debugEventSink{}
		replacement := &debugEventSink{}
		builder := New(Definition{
			SystemInstructions: []Part{NewTextPart("system")},
			PromptInput:        PromptInput{User: ai.TextParts("find docs")},
			ObservationSink:    original,
		})
		builder.SetObservationSink(replacement)

		if _, err := renderBuilderRequest(builder, ctx, emptyConversation{}); err != nil {
			t.Fatalf("render request failed: %v", err)
		}
		if len(original.events) != 0 {
			t.Fatalf("original sink received events after replacement: %#v", original.events)
		}
		assertRendererEvents(t, replacement.events)
	})

	t.Run("nil", func(t *testing.T) {
		original := &debugEventSink{}
		builder := New(Definition{
			SystemInstructions: []Part{NewTextPart("system")},
			PromptInput:        PromptInput{User: ai.TextParts("find docs")},
			ObservationSink:    original,
		})
		builder.SetObservationSink(nil)

		if _, err := renderBuilderRequest(builder, ctx, emptyConversation{}); err != nil {
			t.Fatalf("render request failed: %v", err)
		}
		if len(original.events) != 0 {
			t.Fatalf("original sink received events after removal: %#v", original.events)
		}
	})
}

func assertRendererEvents(t *testing.T, events []gai.Observation) {
	t.Helper()
	for _, event := range events {
		if event.Name == "renderer_render_finished" {
			return
		}
	}
	t.Fatalf("expected context renderer events, got %#v", events)
}

func TestPromptBuilderKeepsTokenErrorEvents(t *testing.T) {
	t.Parallel()

	sink := &debugEventSink{}
	builder := New(Definition{
		SystemInstructions: []Part{failingPart{}},
		ObservationSink:    sink,
	})
	builder.SetTokenCounter(debugTestTokenCounter{})

	if _, err := builder.SystemInstructionsTokens(context.Background()); err == nil {
		t.Fatal("expected token count error to reach the caller")
	}

	names := make([]string, 0, len(sink.events))
	for _, event := range sink.events {
		names = append(names, event.Name)
	}
	if !slices.Contains(names, "prompt_builder_token_count_failed") {
		t.Fatalf("expected token count failure event, got %v", names)
	}
}

func TestPromptBuilderReturnsCancellationBeforeBuilding(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	builder := New(Definition{
		ContextSources: []ContextSource{&testContextSource{name: "source", text: "context"}},
		PromptInput:    PromptInput{User: ai.TextParts("question")},
	})

	if _, err := builder.BuildContext(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("BuildContext error = %v, want context.Canceled", err)
	}
	if _, err := renderBuilderRequest(builder, ctx, emptyConversation{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("render request error = %v, want context.Canceled", err)
	}
}

type failingCountSource struct{}

func (failingCountSource) Name() string                                { return "failing" }
func (failingCountSource) Function(context.Context, int) (Part, error) { return failingPart{}, nil }

func TestBuildContextReturnsCountingErrorsFromEveryPartOrigin(t *testing.T) {
	for _, tc := range []struct {
		name string
		def  Definition
	}{
		{"system", Definition{TokenBudget: 100, SystemInstructions: []Part{failingPart{}}}},
		{"source", Definition{TokenBudget: 100, ContextSources: []ContextSource{failingCountSource{}}}},
		{"input", Definition{TokenBudget: 100, PromptInput: PromptInput{Context: []Part{failingPart{}}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			builder := New(tc.def)
			if _, err := builder.BuildContext(t.Context()); err == nil {
				t.Fatal("counting failure was ignored")
			}
		})
	}
}

func TestBuilderClearingCounterRestoresLocalEstimator(t *testing.T) {
	builder := New(Definition{TokenCounter: debugTestTokenCounter{}})
	builder.SetTokenCounter(nil)
	if _, ok := builder.TokenCounter().(ai.TextTokenEstimator); !ok {
		t.Fatalf("counter = %T", builder.TokenCounter())
	}
}

// Fresh builders may share immutable prompt parts through definitions or inputs.
func TestConcurrentBuildersCanShareImmutablePromptParts(t *testing.T) {
	named, err := NewNamedPart("context", strings.Repeat("input ", 2000))
	if err != nil {
		t.Fatal(err)
	}
	system := NewTextPart(strings.Repeat("instructions ", 2000))
	message := NewMessagePart(ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("shared message")})
	input := PromptInput{Context: []Part{named, message}}
	var group sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < 16; i++ {
		group.Add(1)
		go func() {
			defer group.Done()
			<-start
			builder := New(Definition{TokenBudget: 100000, SystemInstructions: []Part{system}, PromptInput: input.Clone()})
			if _, err := builder.BuildContext(t.Context()); err != nil {
				t.Error(err)
			}
		}()
	}
	close(start)
	group.Wait()
}

type budgetCountingPart struct {
	calls int
}

var errBudgetTestCount = errors.New("budget test count failed")

func (*budgetCountingPart) Name() string { return "text" }
func (p *budgetCountingPart) Tokens(context.Context, ai.TokenCounter) (int, error) {
	p.calls++
	return 0, errBudgetTestCount
}
func (*budgetCountingPart) Render(context.Context) (RenderNode, error) {
	return RenderNode{Type: "text", Value: "renderable context"}, nil
}

type budgetPartSource struct {
	part Part
	err  error
}

func (budgetPartSource) Name() string                                  { return "context" }
func (s budgetPartSource) Function(context.Context, int) (Part, error) { return s.part, s.err }

func TestBuildContextSkipsUnusedCountsWhenBudgetDisabled(t *testing.T) {
	for _, budget := range []int{0, -1} {
		t.Run(fmt.Sprint(budget), func(t *testing.T) {
			part := &budgetCountingPart{}
			sink := &debugEventSink{}
			builder := New(Definition{
				TokenBudget:        budget,
				SystemInstructions: []Part{part},
				ContextSources:     []ContextSource{budgetPartSource{part: part}},
				PromptInput:        PromptInput{Context: []Part{part}},
				ObservationSink:    sink,
			})
			parts, err := builder.BuildContext(t.Context())
			if err != nil || len(parts) != 2 || part.calls != 0 {
				t.Fatalf("BuildContext = %v, %v; count calls = %d", parts, err, part.calls)
			}
			prompt, err := renderBuilderRequest(builder, t.Context(), nil)
			if err != nil || strings.Count(prompt, "renderable context") != 3 {
				t.Fatalf("render request = %q, %v", prompt, err)
			}
			sawSource := false
			for _, event := range sink.events {
				if event.Name == "prompt_builder_source_included" {
					sawSource = true
					if event.Fields["tokens_counted"] != false || event.Fields["tokens"] != 0 {
						t.Fatalf("uncounted source observation = %v", event.Fields)
					}
				}
			}
			if !sawSource {
				t.Fatal("missing source-included observation")
			}
			sourceErr := errors.New("source failure")
			builder.ContextSources = []ContextSource{budgetPartSource{err: sourceErr}}
			if _, err := builder.BuildContext(t.Context()); !errors.Is(err, sourceErr) {
				t.Fatalf("source failure = %v, want %v", err, sourceErr)
			}
		})
	}
}

func TestBuildRequestPreservesOpaqueConversationAndOwnsPayloads(t *testing.T) {
	t.Parallel()
	message := ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "search", Args: []byte(`{"q":"x"}`), Extensions: []ai.Extension{{Namespace: "future", Type: "signature", Data: []byte(`"opaque"`), Required: true}}}}}}
	conv := messageConversation{messages: []ai.Message{message}}
	builder := New(Definition{PromptInput: PromptInput{User: ai.TextParts("question")}})
	request, err := builder.BuildRequest(t.Context(), conv)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(request.Messages[1], message) {
		t.Fatal("native request dropped opaque continuity state")
	}
	request.Messages[1].Parts[0].ToolCall.Args[6] = 'z'
	request.Messages[1].Parts[0].ToolCall.Extensions[0].Data[1] = 'X'
	if string(message.Parts[0].ToolCall.Args) != `{"q":"x"}` || string(message.Parts[0].ToolCall.Extensions[0].Data) != `"opaque"` {
		t.Fatal("request aliases conversation")
	}
	if _, err := renderBuilderRequest(builder, t.Context(), conv); err == nil {
		t.Fatal("fallback discarded required provider extension")
	}
}

func TestBuildRequestPreservesRepeatedIdenticalUserMessages(t *testing.T) {
	t.Parallel()
	user := ai.TextMessage(ai.RoleUser, "question")
	builder := New(Definition{PromptInput: PromptInput{User: user.Parts}})
	conv := messageConversation{messages: []ai.Message{user, ai.TextMessage(ai.RoleAssistant, "clarification"), ai.TextMessage(ai.RoleUser, "follow-up")}}
	request, err := builder.BuildRequest(t.Context(), conv)
	if err != nil {
		t.Fatal(err)
	}
	// Equal content does not establish that two messages have the same origin.
	// The builder preserves every caller-supplied conversation entry.
	if len(request.Messages) != 4 || request.Messages[0].Text() != "question" || request.Messages[1].Text() != "question" || request.Messages[3].Text() != "follow-up" {
		t.Fatalf("builder dropped a caller-supplied user message: %#v", request.Messages)
	}
}

func renderBuilderRequest(builder *Builder, ctx context.Context, conv Conversation) (string, error) {
	request, err := builder.BuildRequest(ctx, conv)
	if err != nil {
		return "", err
	}
	return ai.RenderMessages(ctx, request.Messages)
}

func TestBuildRequestRejectsEmptyConversation(t *testing.T) {
	t.Parallel()
	builder := New(Definition{})
	if _, err := builder.BuildRequest(t.Context(), nil); err == nil {
		t.Fatal("empty builder created a request without canonical messages")
	}
	if _, err := builder.BuildRequest(t.Context(), emptyConversation{}); err == nil {
		t.Fatal("empty conversation created a request without canonical messages")
	}
	builder.SetInput(PromptInput{User: ai.TextParts("")})
	request, err := builder.BuildRequest(t.Context(), nil)
	if err != nil || len(request.Messages) != 1 {
		t.Fatalf("explicit empty text should remain a canonical message: %#v, %v", request, err)
	}
}
