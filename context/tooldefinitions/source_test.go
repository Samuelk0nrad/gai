package tooldefinitions_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/tooldefinitions"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestSourceBuildsToolDefinitionsPart(t *testing.T) {
	sink := &captureSink{}
	source, err := tooldefinitions.New(&gaictx.XMLRenderer{}, []gaictx.ToolSignature{
		staticTool{name: "weather", description: "Gets current weather.", params: testParams("city")},
		staticTool{name: "search", description: "Searches documentation.", params: testParams("query")},
	}, sink)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if source.Name() != "tool_definitions" {
		t.Fatalf("Name = %q, want tool_definitions", source.Name())
	}

	part, err := source.Function(context.Background(), 2048)
	if err != nil {
		t.Fatalf("Function: %v", err)
	}
	rendered, err := (gaictx.XMLRenderer{}).Render(context.Background(), []gaictx.Part{part})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	for _, expected := range []string{
		`<tools>`, `<tool_usage>`, `&#34;function&#34;`, `&lt;tool-name&gt;`, `<tool-name>`, "search", `<description>`, "Searches documentation.", `<signature>`, `query`, "weather",
	} {
		if !strings.Contains(rendered, expected) {
			t.Errorf("rendered definitions missing %q:\n%s", expected, rendered)
		}
	}
	if strings.Index(rendered, "search") > strings.Index(rendered, "weather") {
		t.Fatalf("tools are not rendered deterministically:\n%s", rendered)
	}

	events := sink.Events()
	if len(events) != 2 || events[0].Name != "tool_definitions_build_started" || events[1].Name != "tool_definitions_build_finished" {
		t.Fatalf("unexpected debug events: %#v", events)
	}
	if got := events[1].Fields["tool_count"]; got != 2 {
		t.Fatalf("tool_count = %#v, want 2", got)
	}
}

func TestSourceRendersClearSimpleToolDefinitions(t *testing.T) {
	t.Parallel()

	source, err := tooldefinitions.New(&gaictx.SimpleRenderer{}, []gaictx.ToolSignature{
		staticTool{name: "search", description: "Searches the web.", params: testParams("query")},
	}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	part, err := source.Function(context.Background(), 2048)
	if err != nil {
		t.Fatalf("Function: %v", err)
	}

	rendered, err := (gaictx.SimpleRenderer{}).Render(context.Background(), []gaictx.Part{part})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	want := `<tools>
usage:
When a tool is required, output each tool call as a standalone JSON object using exactly this shape:
{"type":"function","name":"<tool-name>","arguments":{...}}

The name must match a listed tool, type must be exactly "function" and arguments must match its signature. Do not include an id, do not wrap the JSON in Markdown, and separate multiple calls with a blank line. If no tool is needed, respond normally. Do not repeat a completed tool call when its result is already present.
tool: search
description: Searches the web.
signature: {"type":"object","properties":{"query":{"type":"string"}}}
</tools>`
	if rendered != want {
		t.Fatalf("unexpected tool definitions:\nwant %q\n got %q", want, rendered)
	}
}

func TestSourceUsesSignatureOnlyRendererWithoutExposingStoredTools(t *testing.T) {
	source, err := tooldefinitions.New(mutatingRenderer{}, []gaictx.ToolSignature{
		staticTool{name: "search", description: "Searches the web.", params: testParams("query")},
	}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	for range 2 {
		part, err := source.Function(context.Background(), 2048)
		if err != nil {
			t.Fatalf("Function: %v", err)
		}
		rendered, err := (gaictx.SimpleRenderer{}).Render(context.Background(), []gaictx.Part{part})
		if err != nil {
			t.Fatalf("Render: %v", err)
		}
		if !strings.Contains(rendered, "tool: search") {
			t.Fatalf("stored tool was mutated by renderer:\n%s", rendered)
		}
	}
}

func TestSourceAllowsCustomUsageProtocol(t *testing.T) {
	t.Parallel()

	source, err := tooldefinitions.New(&gaictx.SimpleRenderer{}, []gaictx.ToolSignature{
		staticTool{name: "search", description: "Searches the web.", params: testParams("query")},
	}, nil, tooldefinitions.WithUsageProtocol("Call tools only when the user explicitly asks."))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	part, err := source.Function(context.Background(), 2048)
	if err != nil {
		t.Fatalf("Function: %v", err)
	}

	rendered, err := (gaictx.SimpleRenderer{}).Render(context.Background(), []gaictx.Part{part})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if !strings.Contains(rendered, "Call tools only when the user explicitly asks.") {
		t.Fatalf("custom usage protocol missing from render:\n%s", rendered)
	}
	if strings.Contains(rendered, `{"type":"function","name":"<tool-name>","arguments":{...}}`) {
		t.Fatalf("default usage protocol was not replaced:\n%s", rendered)
	}
}

func TestSourceRendersRequiredToolChoice(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		choice ai.ToolChoice
		want   string
	}{
		{
			name:   "auto",
			choice: ai.ToolChoice{Mode: ai.ToolChoiceAuto},
		},
		{
			name:   "required any tool",
			choice: ai.ToolChoice{Mode: ai.ToolChoiceRequired},
			want:   "You must make at least one tool call before producing a normal response.",
		},
		{
			name:   "required named tools",
			choice: ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"search", "weather"}},
			want:   "You must make at least one tool call before producing a normal response. You may call only these selected tools: search, weather.",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			source, err := tooldefinitions.New(&gaictx.SimpleRenderer{}, []gaictx.ToolSignature{
				staticTool{name: "search", description: "Searches the web.", params: testParams("query")},
			}, nil, tooldefinitions.WithToolChoice(tt.choice))
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			part, err := source.Function(context.Background(), 2048)
			if err != nil {
				t.Fatalf("Function: %v", err)
			}
			rendered, err := (gaictx.SimpleRenderer{}).Render(context.Background(), []gaictx.Part{part})
			if err != nil {
				t.Fatalf("Render: %v", err)
			}
			if tt.want == "" {
				if strings.Contains(rendered, "You must make at least one tool call before producing a normal response.") {
					t.Fatalf("unexpected required-tool instruction:\n%s", rendered)
				}
				return
			}
			if !strings.Contains(rendered, tt.want) {
				t.Fatalf("required-tool instruction missing:\n%s", rendered)
			}
		})
	}
}

func TestSourceErrorHandling(t *testing.T) {
	tests := []struct {
		name    string
		tools   []gaictx.ToolSignature
		options []tooldefinitions.Option
		wantErr error
	}{
		{name: "empty", wantErr: tooldefinitions.ErrToolsEmpty},
		{name: "nil tool", tools: []gaictx.ToolSignature{nil}, wantErr: tooldefinitions.ErrToolInvalid},
		{name: "nil option", tools: []gaictx.ToolSignature{staticTool{name: "search", description: "Searches", params: testParams("query")}}, options: []tooldefinitions.Option{nil}, wantErr: tooldefinitions.ErrToolInvalid},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := tooldefinitions.New(&gaictx.XMLRenderer{}, test.tools, nil, test.options...)
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("error = %v, want %v", err, test.wantErr)
			}
		})
	}

	sink := &captureSink{}
	source, err := tooldefinitions.New(&gaictx.XMLRenderer{}, []gaictx.ToolSignature{
		staticTool{name: "broken", description: "Broken definition.", params: invalidParams()},
	}, sink)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = source.Function(context.Background(), 100)
	if !errors.Is(err, tooldefinitions.ErrToolInvalid) {
		t.Fatalf("Function error = %v, want ErrToolInvalid", err)
	}
	events := sink.Events()
	if len(events) != 2 || events[1].Name != "tool_definitions_build_failed" || events[1].Err != nil || events[1].Fields["outcome"] != "error" {
		t.Fatalf("unexpected failure events: %#v", events)
	}
}

func TestSourceRejectsTypedNilToolSignature(t *testing.T) {
	t.Parallel()

	var tool *staticTool
	_, err := tooldefinitions.New(&gaictx.XMLRenderer{}, []gaictx.ToolSignature{tool}, nil)
	if !errors.Is(err, tooldefinitions.ErrToolInvalid) {
		t.Fatalf("New error = %v, want ErrToolInvalid", err)
	}
}

func TestSourceTracing(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	previousProvider := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		otel.SetTracerProvider(previousProvider)
		_ = provider.Shutdown(context.Background())
	})

	source, err := tooldefinitions.New(&gaictx.XMLRenderer{}, []gaictx.ToolSignature{
		staticTool{name: "broken", description: "Broken definition.", params: invalidParams()},
	}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, callErr := source.Function(context.Background(), 99)
	if callErr == nil {
		t.Fatal("expected source error")
	}

	var found bool
	for _, span := range recorder.Ended() {
		if span.Name() != "context.tool_definitions.build" {
			continue
		}
		found = true
		if span.Status().Code != codes.Error {
			t.Fatalf("span status = %v, want error", span.Status().Code)
		}
	}
	if !found {
		t.Fatal("context.tool_definitions.build span was not recorded")
	}
}

type staticTool struct {
	name        string
	description string
	params      ai.ToolParameters
}

type mutatingRenderer struct{}

var _ tooldefinitions.ToolSignatureRenderer = mutatingRenderer{}

func (mutatingRenderer) RenderToolSignatures(tools []gaictx.ToolSignature) (string, error) {
	tools[0] = staticTool{name: "replacement", description: "Replaces the stored tool.", params: testParams("query")}
	return "", nil
}

func (t staticTool) Name() string        { return t.name }
func (t staticTool) Description() string { return t.description }
func (t staticTool) Params() ai.ToolParameters {
	return t.params
}
// Function supplies an empty successful handler for fixtures that exercise tool-definition
// rendering.
func (t staticTool) Function(context.Context, ai.ToolCall) (string, error) {
	return "", nil
}

func testParams(name string) ai.ToolParameters {
	return ai.ToolParameters{
		Properties: []ai.ToolParameter{
			{Name: name, Type: ai.ToolParameterString},
		},
	}
}

func invalidParams() ai.ToolParameters {
	return ai.ToolParameters{
		Properties: []ai.ToolParameter{
			{Name: "value", Type: ai.ToolParameterType("unsupported")},
		},
	}
}

type captureSink struct {
	mu     sync.Mutex
	events []gai.Observation
}

func (s *captureSink) Emit(_ context.Context, event gai.Observation) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, event)
}

func (s *captureSink) Events() []gai.Observation {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]gai.Observation(nil), s.events...)
}
