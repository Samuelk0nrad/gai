package agent_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/testutil/mocks"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestAgentWorkflowEmitsLifecycleEventsAndSpans(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(provider)
	defer func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	}()

	sink := &agentObservationSink{}
	post := agent.New(agent.Definition{
		Name:  "post",
		Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Message: ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "post"}}}}}}},
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})
	primary := agent.New(agent.Definition{
		Name:            "primary",
		Model:           &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Message: ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "primary"}}}}}}},
		ObservationSink: sink,
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
		Middleware: []agent.Middleware{agent.NewAgentMiddleware(post, agent.AgentMiddlewareConfig{
			Output: agent.PreserveOutput,
		})},
	})

	workflow, err := primary.NewRun(t.Context(), agent.RunInput{
		ID:     "run-1",
		Prompt: gaictx.PromptInput{User: ai.TextParts("question")},
		Meta:   map[string]any{"session_id": "session-1"},
	})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 0 {
		t.Fatalf("workflow errors: %v", consumed.errs)
	}

	wantEvents := []string{
		"agent_run_created",
		"agent_workflow_started",
		"agent_primary_finished",
		"agent_middleware_started",
		"agent_middleware_finished",
		"agent_workflow_finished",
	}
	for _, name := range wantEvents {
		if !sink.hasEvent(name) {
			t.Errorf("missing debug event %q; got %v", name, sink.names())
		}
	}
	for _, event := range sink.events {
		if event.RunID != "run-1" {
			t.Errorf("observation %q run ID = %q, want run-1", event.Name, event.RunID)
		}
	}
	created, ok := sink.event("agent_run_created")
	if !ok {
		t.Fatal("missing run-created event")
	}
	if _, leaked := created.Fields["user_input"]; leaked {
		t.Fatalf("non-sensitive sink received input text: %+v", created.Fields)
	}

	spanNames := map[string]bool{}
	for _, span := range recorder.Ended() {
		spanNames[span.Name()] = true
	}
	for _, name := range []string{"agent.run.create", "agent.workflow.run", "agent.middleware.run"} {
		if !spanNames[name] {
			t.Errorf("missing span %q; got %v", name, spanNames)
		}
	}
}

func TestAgentRunSpanIsParentOfWorkflow(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	})

	a := agent.New(agent.Definition{
		Name:  "primary",
		Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Message: ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "done"}}}}}}},
		Prompt: func(_ context.Context, _ agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})
	workflow, err := a.NewRun(t.Context(), agent.RunInput{ID: "run-42", Prompt: gaictx.PromptInput{User: ai.TextParts("question")}, Meta: map[string]any{"session_id": "session-1"}})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 0 {
		t.Fatalf("workflow errors: %v", consumed.errs)
	}

	var run, workflowSpan sdktrace.ReadOnlySpan
	for _, span := range recorder.Ended() {
		switch span.Name() {
		case "agent.run":
			run = span
		case "agent.workflow.run":
			workflowSpan = span
		}
	}
	if run == nil || workflowSpan == nil {
		t.Fatalf("expected agent.run and agent.workflow.run spans, got %#v", recorder.Ended())
	}
	if workflowSpan.Parent().SpanID() != run.SpanContext().SpanID() {
		t.Fatalf("workflow span parent = %s, want run span %s", workflowSpan.Parent().SpanID(), run.SpanContext().SpanID())
	}
	attributes := map[string]any{}
	for _, attribute := range run.Attributes() {
		attributes[string(attribute.Key)] = attribute.Value.AsInterface()
	}
	if attributes["agent.run_id"] != "run-42" || attributes["agent.meta_key_count"] != int64(1) {
		t.Fatalf("unexpected run attributes: %#v", attributes)
	}
	if _, leaked := attributes["agent.meta.session_id"]; leaked {
		t.Fatalf("run span leaked metadata value: %#v", attributes)
	}
}

func TestAgentContentCapturePolicySeparatesPromptCompletionAndReasoning(t *testing.T) {
	sink := &agentObservationSink{}
	a := agent.New(agent.Definition{
		Name:            "primary",
		ObservationSink: sink,
		Model:           &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Message: ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentReasoning, Text: "reason-secret"}, {Kind: ai.ContentText, Text: "answer-secret"}}}}}}},
		Prompt: func(_ context.Context, _ agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})
	ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{
		Prompt: gai.CaptureEnabled, Completion: gai.CaptureEnabled, Reasoning: gai.CaptureEnabled,
		Redact: func(_ context.Context, _ gai.ContentKind, value []byte) ([]byte, error) {
			return []byte(strings.ReplaceAll(string(value), "secret", "[redacted]")), nil
		},
	})
	workflow, err := a.NewRun(ctx, agent.RunInput{Prompt: gaictx.PromptInput{User: ai.TextParts("question-secret")}})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if consumed := consumeWorkflowContext(t, workflow, ctx); len(consumed.errs) != 0 {
		t.Fatalf("workflow errors: %v", consumed.errs)
	}

	created, ok := sink.event("agent_run_created")
	if !ok || created.Fields["user_input"] != "question-[redacted]" {
		t.Fatalf("captured prompt = %#v", created.Fields)
	}
	finished, ok := sink.event("agent_workflow_finished")
	if !ok || finished.Fields["output_text"] != "answer-[redacted]" || finished.Fields["reasoning"] != "reason-[redacted]" {
		t.Fatalf("captured output = %#v", finished.Fields)
	}
}

func TestAgentObservabilityReportsCreationAndMiddlewareFailures(t *testing.T) {
	sink := &agentObservationSink{}
	_, err := agent.New(agent.Definition{Name: "broken", ObservationSink: sink}).NewRun(t.Context(), textRunInput("question"))
	if err == nil {
		t.Fatal("expected run creation failure")
	}
	if event, ok := sink.event("agent_run_creation_failed"); !ok || event.Err != nil || event.Fields["outcome"] != "error" {
		t.Fatalf("missing creation failure event: %+v", event)
	}

	mapErr := errors.New("map input")
	post := agent.New(agent.Definition{
		Name:  "post",
		Model: &mocks.MockModel{},
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})
	primary := agent.New(agent.Definition{
		Name:            "primary",
		Model:           &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Message: ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "primary"}}}}}}},
		ObservationSink: sink,
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
		Middleware: []agent.Middleware{agent.NewAgentMiddleware(post, agent.AgentMiddlewareConfig{
			ErrorPolicy: agent.RecordError,
			MapInput: func(context.Context, agent.WorkflowResult) (agent.RunInput, error) {
				return agent.RunInput{}, mapErr
			},
		})},
	})
	workflow, err := primary.NewRun(t.Context(), textRunInput("question"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 0 {
		t.Fatalf("record-only middleware failure propagated: %v", consumed.errs)
	}
	event, ok := sink.event("agent_middleware_failed")
	if !ok || event.Err != nil || event.Fields["outcome"] != "error" {
		t.Fatalf("missing middleware failure event: %+v", event)
	}
}

type traceTestModel struct {
	mu      sync.Mutex
	scripts [][]ai.Token
	call    int
}

func (m *traceTestModel) Name() string { return "trace-test-model" }

func (m *traceTestModel) Generate(context.Context, ai.AIRequest) (*ai.AIResponse, error) {
	return &ai.AIResponse{}, nil
}

func (m *traceTestModel) GenerateStream(ctx context.Context, _ ai.AIRequest) <-chan ai.Token {
	ctx, span := gai.StartOperationSpan(ctx, "gai-test", "test.model", "test.operation", "generate")
	m.mu.Lock()
	call := m.call
	m.call++
	var script []ai.Token
	if call < len(m.scripts) {
		script = append([]ai.Token(nil), m.scripts[call]...)
	}
	m.mu.Unlock()
	out := make(chan ai.Token, len(script))
	go func() {
		defer close(out)
		defer span.End()
		for _, token := range script {
			select {
			case out <- token:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out
}

func (*traceTestModel) Close() error                  { return nil }
func (*traceTestModel) TokenCounter() ai.TokenCounter { return &mocks.MockTokenCounter{} }

type traceTestTool struct{ loop.Tool }

// Function adds an inner tool span to verify trace-context propagation through agent execution.
func (t traceTestTool) Function(ctx context.Context, call ai.ToolCall) (string, error) {
	ctx, span := gai.StartOperationSpan(ctx, "gai-test", "test.tool", "test.operation", "execute")
	defer span.End()
	return t.Tool.Function(ctx, call)
}

func TestTraceContextPropagatesAcrossRetriesToolsAndNestedMiddleware(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	})

	primaryModel := &traceTestModel{scripts: [][]ai.Token{
		{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("retry")}}},
		{{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "echo", Args: []byte(`{"text":"hello"}`)}}}},
		{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "primary"}}},
	}}
	postModel := &traceTestModel{scripts: [][]ai.Token{{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "post"}}}}}
	post := agent.New(agent.Definition{
		Name:  "post",
		Model: postModel,
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})
	primary := agent.New(agent.Definition{
		Name:        "primary",
		Model:       primaryModel,
		Tools:       []loop.Tool{traceTestTool{Tool: loop.NewEchoTool()}},
		RetryPolicy: &loop.RetryPolicy{MaxRetries: 1},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
		Middleware: []agent.Middleware{agent.NewAgentMiddleware(post, agent.AgentMiddlewareConfig{Output: agent.PreserveOutput})},
	})

	traceContext := &gai.TraceContext{
		Name:        "lace-chat",
		UserID:      "user-1",
		SessionID:   "session-1",
		Tags:        []string{"mobile"},
		Release:     "2026.08",
		Environment: "staging",
		Metadata:    map[string]string{"feature": "chat"},
	}
	workflow, err := primary.NewRun(t.Context(), agent.RunInput{
		ID:           "run-1",
		TraceContext: traceContext,
		Prompt:       gaictx.PromptInput{User: ai.TextParts("question")},
		Meta:         map[string]any{"private": "meta-secret"},
	})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	traceContext.UserID = "mutated-user"
	traceContext.Tags[0] = "mutated-tag"
	traceContext.Metadata["feature"] = "mutated-feature"
	if consumed := consumeWorkflow(t, workflow); len(consumed.errs) != 0 {
		t.Fatalf("workflow errors: %v", consumed.errs)
	}

	required := map[string]bool{
		"agent.run":            false,
		"agent.workflow.run":   false,
		"agent.middleware.run": false,
		"loop.run":             false,
		"loop.iteration":       false,
		"loop.tool":            false,
		"test.model.generate":  false,
		"test.tool.execute":    false,
	}
	for _, span := range recorder.Ended() {
		attributes := spanAttributeMap(span)
		if _, tracked := required[span.Name()]; tracked {
			required[span.Name()] = true
			if attributes["gai.trace.name"] != "lace-chat" || attributes["gai.trace.user_id"] != "user-1" || attributes["gai.trace.session_id"] != "session-1" || attributes["gai.trace.release"] != "2026.08" || attributes["gai.trace.environment"] != "staging" || attributes["gai.trace.metadata.feature"] != "chat" {
				t.Errorf("span %q missing trace context: %#v", span.Name(), attributes)
			}
			tags, ok := attributes["gai.trace.tags"].([]string)
			if !ok || len(tags) != 1 || tags[0] != "mobile" {
				t.Errorf("span %q tags = %#v", span.Name(), attributes["gai.trace.tags"])
			}
		}
		if strings.Contains(fmt.Sprint(attributes), "meta-secret") || strings.Contains(fmt.Sprint(attributes), "mutated-") {
			t.Errorf("span %q leaked unapproved or mutated values: %#v", span.Name(), attributes)
		}
	}
	for name, seen := range required {
		if !seen {
			t.Errorf("missing traced operation %q", name)
		}
	}
}

func TestRunInputTraceContextInheritsOrReplacesAtomically(t *testing.T) {
	a := agent.New(agent.Definition{
		Name:  "trace-context",
		Model: &mocks.MockModel{},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})
	inheritedCtx := gai.WithTraceContext(t.Context(), gai.TraceContext{UserID: "inherited", SessionID: "keep-only-on-inherit"})
	inherited, err := a.NewRun(inheritedCtx, textRunInput("inherit"))
	if err != nil {
		t.Fatalf("NewRun inherited: %v", err)
	}
	if got := inherited.Result().Input.TraceContext; got == nil || got.UserID != "inherited" || got.SessionID != "keep-only-on-inherit" {
		t.Fatalf("inherited trace context = %#v", got)
	}

	replacementInput := textRunInput("replace")
	replacementInput.TraceContext = &gai.TraceContext{UserID: "replacement"}
	replaced, err := a.NewRun(inheritedCtx, replacementInput)
	if err != nil {
		t.Fatalf("NewRun replaced: %v", err)
	}
	if got := replaced.Result().Input.TraceContext; got == nil || got.UserID != "replacement" || got.SessionID != "" {
		t.Fatalf("replacement retained inherited fields: %#v", got)
	}
}

func TestTraceContextSurvivesCanceledRunEventsContext(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	})

	a := agent.New(agent.Definition{
		Name:  "canceled",
		Model: &mocks.MockModel{},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})
	workflow, err := a.NewRun(t.Context(), agent.RunInput{
		TraceContext: &gai.TraceContext{UserID: "canceled-user"},
		Prompt:       gaictx.PromptInput{User: ai.TextParts("question")},
	})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	runCtx, cancel := context.WithCancel(context.Background())
	cancel()
	for range workflow.RunEvents(runCtx) {
	}

	want := map[string]bool{"agent.run": false, "agent.workflow.run": false, "loop.run": false}
	for _, span := range recorder.Ended() {
		if _, ok := want[span.Name()]; !ok {
			continue
		}
		want[span.Name()] = true
		if got := spanAttributeMap(span)["gai.trace.user_id"]; got != "canceled-user" {
			t.Errorf("span %q user = %#v", span.Name(), got)
		}
	}
	for name, seen := range want {
		if !seen {
			t.Errorf("missing canceled span %q", name)
		}
	}
}

func spanAttributeMap(span sdktrace.ReadOnlySpan) map[string]any {
	attributes := make(map[string]any, len(span.Attributes()))
	for _, attr := range span.Attributes() {
		attributes[string(attr.Key)] = attr.Value.AsInterface()
	}
	return attributes
}

type agentObservationSink struct {
	mu     sync.Mutex
	events []gai.Observation
}

func (s *agentObservationSink) Emit(_ context.Context, event gai.Observation) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, event)
}

func (s *agentObservationSink) hasEvent(name string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, event := range s.events {
		if event.Name == name {
			return true
		}
	}
	return false
}

func (s *agentObservationSink) event(name string) (gai.Observation, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, event := range s.events {
		if event.Name == name {
			return event, true
		}
	}
	return gai.Observation{}, false
}

func (s *agentObservationSink) names() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	names := make([]string, len(s.events))
	for i, event := range s.events {
		names[i] = event.Name
	}
	return names
}
