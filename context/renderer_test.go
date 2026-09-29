package context_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/tooldefinitions"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

type renderTestPart struct {
	name string
	node gaictx.RenderNode
}

type rendererObservationSink struct {
	events []gai.Observation
}

func (s *rendererObservationSink) Emit(_ context.Context, event gai.Observation) {
	s.events = append(s.events, event)
}

func (p renderTestPart) Name() string {
	return p.name
}

func (p renderTestPart) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	return 0, nil
}

func (p renderTestPart) Render(ctx context.Context) (gaictx.RenderNode, error) {
	return p.node, nil
}

func TestRendererEmitsToOTelWithoutSink(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })

	ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{Prompt: gai.CaptureEnabled})
	ctx, span := provider.Tracer("renderer-test").Start(ctx, "render")
	_, err := (gaictx.SimpleRenderer{}).Render(ctx, []gaictx.Part{
		gaictx.NewMessagePart(gaictx.RoleUser, gaictx.NewTextContent("allowed prompt")),
	})
	span.End()
	if err != nil {
		t.Fatal(err)
	}

	for _, event := range recorder.Ended()[0].Events() {
		if event.Name != "observation.renderer_part_rendered" {
			continue
		}
		for _, attribute := range event.Attributes {
			if string(attribute.Key) == "observation.rendered" && attribute.Value.AsString() == "user: allowed prompt" {
				return
			}
		}
		t.Fatalf("renderer event omitted policy-enabled content: %#v", event.Attributes)
	}
	t.Fatalf("OTel-only renderer omitted part observation: %#v", recorder.Ended())
}

func TestXMLRendererRendersNestedNodesAndEscapesContent(t *testing.T) {
	t.Parallel()

	rendered, err := (gaictx.XMLRenderer{}).Render(context.Background(), []gaictx.Part{
		renderTestPart{
			name: "debug-name",
			node: gaictx.RenderNode{
				Type:   "message",
				Fields: []gaictx.RenderField{{Key: "role", Value: `user&"admin"`}},
				Children: []gaictx.RenderNode{
					{Type: "text", Value: "hello <world> & everyone"},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	want := "<message role=\"user&amp;&#34;admin&#34;\">\n  <text>\n    hello &lt;world&gt; &amp; everyone\n  </text>\n</message>\n"
	if rendered != want {
		t.Fatalf("unexpected render output:\nwant %q\n got %q", want, rendered)
	}
	if strings.Contains(rendered, "debug-name") {
		t.Fatalf("expected part name not to be used as xml tag: %q", rendered)
	}
}

func TestXMLRendererRejectsInvalidNames(t *testing.T) {
	t.Parallel()

	_, err := (gaictx.XMLRenderer{}).Render(context.Background(), []gaictx.Part{
		renderTestPart{
			name: "bad",
			node: gaictx.RenderNode{Type: "1bad"},
		},
	})
	if err == nil {
		t.Fatal("expected invalid tag error")
	}
	if !strings.Contains(err.Error(), `invalid xml tag name: "1bad"`) {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = (gaictx.XMLRenderer{}).Render(context.Background(), []gaictx.Part{
		renderTestPart{
			name: "bad",
			node: gaictx.RenderNode{
				Type:   "valid",
				Fields: []gaictx.RenderField{{Key: "bad attr", Value: "x"}},
			},
		},
	})
	if err == nil {
		t.Fatal("expected invalid attribute error")
	}
	if !strings.Contains(err.Error(), `invalid xml attribute name: "bad attr"`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestXMLRendererReturnsPartRenderError(t *testing.T) {
	t.Parallel()

	wantErr := errors.New("render failed")
	part := failingRenderPart{err: wantErr}
	_, err := (gaictx.XMLRenderer{}).Render(context.Background(), []gaictx.Part{part})
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected render error %v, got %v", wantErr, err)
	}
}

func TestSimpleRendererRendersInstructionsHistoryAndConversation(t *testing.T) {
	t.Parallel()

	rendered, err := (gaictx.SimpleRenderer{}).Render(context.Background(), []gaictx.Part{
		gaictx.NewSystemPart([]gaictx.Part{
			gaictx.NewTextPart("follow the instructions carefully"),
			renderTestPart{
				name: "system",
				node: gaictx.RenderNode{Type: "text", Value: "be precise <raw> & direct"},
			},
			renderTestPart{
				name: "tool",
				node: gaictx.RenderNode{Type: "text", Value: "always call search(\"x\") if needed"},
			},
		}),
		&historyPartAdapter{
			contents: []gaictx.Message{
				{Role: gaictx.RoleUser, Content: gaictx.NewTextContent("hi <there> & \"quoted\"")},
				{Role: gaictx.RoleAssistant, Content: gaictx.NewToolCallContent("search", `{"q":"lace<&>"}`)},
				{Role: gaictx.RoleTool, Content: gaictx.NewToolResultContent("search", `found <docs> & "notes"`, false, "")},
				{Role: gaictx.RoleAssistant, Content: gaictx.NewTextContent("done")},
			},
		},
		gaictx.NewMessagePart(gaictx.RoleUser, gaictx.NewTextContent("find docs")),
		gaictx.NewMessagePart(gaictx.RoleAssistant, gaictx.NewToolCallContent("search", `{"q":"lace"}`)),
		gaictx.NewMessagePart(gaictx.RoleTool, gaictx.NewToolResultContent("search", "found <docs>", false, "")),
	})
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	want := `<Instructions>

follow the instructions carefully

System:
be precise <raw> & direct

Tool:
always call search("x") if needed

</Instructions>

<history>
user: hi <there> & "quoted"
assistant: {"arguments":{"q":"lace\u003c\u0026\u003e"},"name":"search","type":"function"}
tool res: found <docs> & "notes"
assistant: done
</history>

user: find docs

assistant: {"arguments":{"q":"lace"},"name":"search","type":"function"}

tool res: found <docs>`
	if rendered != want {
		t.Fatalf("unexpected render output:\nwant %q\n got %q", want, rendered)
	}
	if strings.Contains(rendered, "&lt;") || strings.Contains(rendered, "&amp;") || strings.Contains(rendered, "&#34;") {
		t.Fatalf("expected raw characters to be preserved: %q", rendered)
	}
}

func TestSimpleRendererPreservesGenericNodeStructure(t *testing.T) {
	t.Parallel()

	rendered, err := (gaictx.SimpleRenderer{}).Render(context.Background(), []gaictx.Part{
		renderTestPart{
			name: "memory_profile",
			node: gaictx.RenderNode{
				Type:   "memory_profile",
				Fields: []gaictx.RenderField{{Key: "version", Value: "2"}},
				Value:  `{"preferred_name":"Sam"}`,
				Children: []gaictx.RenderNode{
					{Type: "source", Value: "persisted"},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	want := `<memory_profile version="2">
{"preferred_name":"Sam"}
<source>
persisted
</source>
</memory_profile>`
	if rendered != want {
		t.Fatalf("unexpected generic node render:\nwant %q\n got %q", want, rendered)
	}
}

func TestRenderersEmitDetailedTruncatedObservations(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		source   string
		renderer func(*rendererObservationSink) gaictx.Renderer
	}{
		{
			name:   "xml",
			source: "context:XMLRenderer",
			renderer: func(sink *rendererObservationSink) gaictx.Renderer {
				return &gaictx.XMLRenderer{ObservationSink: sink, ObservationPreviewChars: 5}
			},
		},
		{
			name:   "simple",
			source: "context:SimpleRenderer",
			renderer: func(sink *rendererObservationSink) gaictx.Renderer {
				return &gaictx.SimpleRenderer{ObservationSink: sink, ObservationPreviewChars: 5}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			sink := &rendererObservationSink{}
			ctx := gai.WithContentCapturePolicy(context.Background(), gai.ContentCapturePolicy{Prompt: gai.CaptureEnabled, Completion: gai.CaptureEnabled})
			_, err := tt.renderer(sink).Render(ctx, []gaictx.Part{
				gaictx.NewMessagePart(gaictx.RoleUser, gaictx.NewTextContent("first long message")),
				gaictx.NewMessagePart(gaictx.RoleAssistant, gaictx.NewTextContent("second long response")),
			})
			if err != nil {
				t.Fatalf("Render failed: %v", err)
			}

			wantNames := []string{
				"renderer_render_started",
				"renderer_part_rendered",
				"renderer_part_rendered",
				"renderer_render_finished",
			}
			if len(sink.events) != len(wantNames) {
				t.Fatalf("unexpected event count: got %d want %d", len(sink.events), len(wantNames))
			}
			for i, want := range wantNames {
				if got := sink.events[i].Name; got != want {
					t.Fatalf("event %d: got %q want %q", i, got, want)
				}
				if got := sink.events[i].Source; got != tt.source {
					t.Fatalf("event %d: source = %q, want %q", i, got, tt.source)
				}
			}

			partEvent := sink.events[1]
			if got := partEvent.Fields["part_index"]; got != 0 {
				t.Fatalf("unexpected part index: %v", got)
			}
			if got := partEvent.Fields["rendered_content_kind"]; got != "prompt" {
				t.Fatalf("expected policy-captured rendered content, got %v", got)
			}
			node, ok := partEvent.Fields["node"].(map[string]any)
			if !ok || node["type"] != "user" || node["value_content_kind"] != "prompt" {
				t.Fatalf("unexpected node structure: %#v", partEvent.Fields["node"])
			}
			assertTruncatedPreview(t, partEvent.Fields, "rendered", 5)
			assertTruncatedPreview(t, node, "value", 5)

			finalEvent := sink.events[3]
			if got := finalEvent.Fields["prompt_content_kind"]; got != "prompt" {
				t.Fatalf("expected policy-captured prompt, got %v", got)
			}
			assertTruncatedPreview(t, finalEvent.Fields, "prompt", 5)
			structure, ok := finalEvent.Fields["structure"].([]map[string]any)
			if !ok || len(structure) != 2 {
				t.Fatalf("unexpected final structure: %#v", finalEvent.Fields["structure"])
			}
		})
	}
}

func TestRendererObservationsSnapshotNestedFieldsAcrossEvents(t *testing.T) {
	sink := &rendererObservationSink{}
	ctx := gai.WithContentCapturePolicy(context.Background(), gai.ContentCapturePolicy{Prompt: gai.CaptureEnabled})
	_, err := (&gaictx.SimpleRenderer{ObservationSink: sink}).Render(ctx, []gaictx.Part{
		gaictx.NewMessagePart(gaictx.RoleUser, gaictx.NewTextContent("first message")),
	})
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	partNode, ok := sink.events[1].Fields["node"].(map[string]any)
	if !ok {
		t.Fatalf("unexpected part node: %#v", sink.events[1].Fields["node"])
	}
	partNode["type"] = "mutated"

	finished := sink.events[len(sink.events)-1]
	structure, ok := finished.Fields["structure"].([]map[string]any)
	if !ok || len(structure) != 1 {
		t.Fatalf("unexpected finished structure: %#v", finished.Fields["structure"])
	}
	finishedNode, ok := structure[0]["node"].(map[string]any)
	if !ok {
		t.Fatalf("unexpected finished node: %#v", structure[0]["node"])
	}
	if got := finishedNode["type"]; got != "user" {
		t.Fatalf("finished observation was mutated through part observation: type = %v, want user", got)
	}
}

func assertTruncatedPreview(t *testing.T, fields map[string]any, key string, wantChars int) {
	t.Helper()

	for _, suffix := range []string{"_head", "_tail"} {
		preview, ok := fields[key+suffix].(string)
		if !ok {
			t.Fatalf("missing %s preview: %#v", key+suffix, fields)
		}
		if got := len([]rune(preview)); got != wantChars {
			t.Fatalf("%s preview characters = %d, want %d", key+suffix, got, wantChars)
		}
	}
	if got := fields[key+"_mode"]; got != "truncated" {
		t.Fatalf("%s preview mode = %v, want truncated", key, got)
	}
}

func TestRendererDebugStructureOmitsContentForNonSensitiveSink(t *testing.T) {
	t.Parallel()

	sink := &rendererObservationSink{}
	_, err := (gaictx.SimpleRenderer{ObservationSink: sink}).Render(context.Background(), []gaictx.Part{
		gaictx.NewMessagePart(gaictx.RoleUser, gaictx.NewTextContent("secret prompt content")),
	})
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	partEvent := sink.events[1]
	if _, ok := partEvent.Fields["rendered"]; ok {
		t.Fatal("non-sensitive event contains rendered content")
	}
	node := partEvent.Fields["node"].(map[string]any)
	if _, ok := node["value"]; ok {
		t.Fatal("non-sensitive node contains value content")
	}
	if got := node["value_chars"]; got != len("secret prompt content") {
		t.Fatalf("unexpected value character count: %v", got)
	}

	finalEvent := sink.events[len(sink.events)-1]
	if _, ok := finalEvent.Fields["prompt"]; ok {
		t.Fatal("non-sensitive event contains prompt content")
	}
}

func TestRenderersClassifyStructuredContentIndependently(t *testing.T) {
	tests := []struct {
		name     string
		renderer func(*rendererObservationSink) gaictx.Renderer
	}{
		{name: "xml", renderer: func(sink *rendererObservationSink) gaictx.Renderer { return &gaictx.XMLRenderer{ObservationSink: sink} }},
		{name: "simple", renderer: func(sink *rendererObservationSink) gaictx.Renderer {
			return &gaictx.SimpleRenderer{ObservationSink: sink}
		}},
	}
	mixedHistory := renderTestPart{name: "history", node: gaictx.RenderNode{
		Type: "history",
		Children: []gaictx.RenderNode{
			{Type: string(gaictx.RoleUser), Value: "memory-only-value"},
			{
				Type:   gaictx.ContentTypeToolResult,
				Fields: []gaictx.RenderField{{Key: "name", Value: "search-tool"}},
				Children: []gaictx.RenderNode{
					{Type: "result", Value: "tool-output-value"},
				},
			},
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			policyTests := []struct {
				name            string
				policy          gai.ContentCapturePolicy
				wantValues      []string
				unwantedValues  []string
				wantFinalPrompt bool
			}{
				{
					name: "memory only", policy: gai.ContentCapturePolicy{Memory: gai.CaptureEnabled},
					wantValues: []string{"memory-only-value"}, unwantedValues: []string{"search-tool", "tool-output-value"},
				},
				{
					name: "tool output only", policy: gai.ContentCapturePolicy{ToolOutput: gai.CaptureEnabled},
					wantValues: []string{"search-tool", "tool-output-value"}, unwantedValues: []string{"memory-only-value"},
				},
				{
					name: "prompt only", policy: gai.ContentCapturePolicy{Prompt: gai.CaptureEnabled},
					unwantedValues: []string{"memory-only-value", "search-tool", "tool-output-value"}, wantFinalPrompt: true,
				},
			}
			for _, policyTest := range policyTests {
				t.Run(policyTest.name, func(t *testing.T) {
					sink := &rendererObservationSink{}
					ctx := gai.WithContentCapturePolicy(t.Context(), policyTest.policy)
					if _, err := tt.renderer(sink).Render(ctx, []gaictx.Part{mixedHistory}); err != nil {
						t.Fatalf("Render failed: %v", err)
					}
					partEvent := sink.events[1]
					if _, ok := partEvent.Fields["rendered"]; ok {
						t.Fatalf("mixed part exposed aggregate rendered content: %#v", partEvent.Fields)
					}
					values := rendererCapturedValues(partEvent.Fields["node"])
					for _, want := range policyTest.wantValues {
						if !values[want] {
							t.Errorf("missing captured value %q in %#v", want, values)
						}
					}
					for _, unwanted := range policyTest.unwantedValues {
						if values[unwanted] {
							t.Errorf("unexpected captured value %q in %#v", unwanted, values)
						}
					}
					finalEvent := sink.events[len(sink.events)-1]
					_, hasPrompt := finalEvent.Fields["prompt"]
					if hasPrompt != policyTest.wantFinalPrompt {
						t.Fatalf("final prompt presence = %v, want %v: %#v", hasPrompt, policyTest.wantFinalPrompt, finalEvent.Fields)
					}
				})
			}

			sink := &rendererObservationSink{}
			ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{ToolOutput: gai.CaptureEnabled})
			toolMessage := renderTestPart{name: "message", node: gaictx.RenderNode{Type: string(gaictx.RoleTool), Value: "direct-tool-output"}}
			if _, err := tt.renderer(sink).Render(ctx, []gaictx.Part{toolMessage}); err != nil {
				t.Fatalf("Render direct tool message: %v", err)
			}
			partEvent := sink.events[1]
			if partEvent.Fields["rendered_content_kind"] != string(gai.ContentKindToolOutput) {
				t.Fatalf("direct tool aggregate kind = %#v", partEvent.Fields)
			}
			if values := rendererCapturedValues(partEvent.Fields["node"]); !values["direct-tool-output"] {
				t.Fatalf("direct tool output was not captured: %#v", values)
			}

			for _, nodeType := range []string{"arguments", "result", "error", "summary"} {
				t.Run("generic "+nodeType+" remains prompt", func(t *testing.T) {
					part := renderTestPart{name: nodeType, node: gaictx.RenderNode{Type: nodeType, Value: "generic-prompt-value"}}

					promptSink := &rendererObservationSink{}
					promptCtx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{Prompt: gai.CaptureEnabled})
					if _, err := tt.renderer(promptSink).Render(promptCtx, []gaictx.Part{part}); err != nil {
						t.Fatalf("Render generic prompt node: %v", err)
					}
					promptPartEvent := promptSink.events[1]
					if promptPartEvent.Fields["rendered_content_kind"] != string(gai.ContentKindPrompt) {
						t.Fatalf("generic node aggregate kind = %#v", promptPartEvent.Fields)
					}
					if values := rendererCapturedValues(promptPartEvent.Fields["node"]); !values["generic-prompt-value"] {
						t.Fatalf("generic prompt node was not captured: %#v", values)
					}

					toolSink := &rendererObservationSink{}
					toolCtx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{ToolOutput: gai.CaptureEnabled})
					if _, err := tt.renderer(toolSink).Render(toolCtx, []gaictx.Part{part}); err != nil {
						t.Fatalf("Render generic node with tool policy: %v", err)
					}
					toolPartEvent := toolSink.events[1]
					if _, ok := toolPartEvent.Fields["rendered"]; ok {
						t.Fatalf("generic prompt aggregate captured as tool output: %#v", toolPartEvent.Fields)
					}
					if values := rendererCapturedValues(toolPartEvent.Fields["node"]); values["generic-prompt-value"] {
						t.Fatalf("generic prompt node captured as tool output: %#v", values)
					}
				})
			}
		})
	}
}

func rendererCapturedValues(value any) map[string]bool {
	values := map[string]bool{}
	var visit func(any)
	visit = func(value any) {
		switch value := value.(type) {
		case map[string]any:
			if content, ok := value["value"].(string); ok {
				values[content] = true
			}
			for _, nested := range value {
				visit(nested)
			}
		case []map[string]any:
			for _, nested := range value {
				visit(nested)
			}
		case []any:
			for _, nested := range value {
				visit(nested)
			}
		}
	}
	visit(value)
	return values
}

func TestRenderersNotifyRenderResultCallback(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		renderer interface {
			gaictx.Renderer
			SetRenderResultCallback(context.Context, gaictx.RenderResultCallback) error
		}
	}{
		{name: "xml", renderer: &gaictx.XMLRenderer{}},
		{name: "simple", renderer: &gaictx.SimpleRenderer{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			parts := []gaictx.Part{
				gaictx.NewMessagePart(gaictx.RoleUser, gaictx.NewTextContent("hello")),
			}
			returned := false
			calls := 0
			var callbackPrompt string

			err := tt.renderer.SetRenderResultCallback(context.Background(), func(receivedParts []gaictx.Part, prompt string) {
				if returned {
					t.Fatal("callback ran after Render returned")
				}
				calls++
				callbackPrompt = prompt
				if len(receivedParts) != len(parts) || receivedParts[0].Name() != parts[0].Name() {
					t.Fatalf("unexpected callback parts: %#v", receivedParts)
				}
			})
			if err != nil {
				t.Fatalf("SetRenderResultCallback failed: %v", err)
			}

			prompt, err := tt.renderer.Render(context.Background(), parts)
			returned = true
			if err != nil {
				t.Fatalf("Render failed: %v", err)
			}
			if calls != 1 {
				t.Fatalf("unexpected callback count: %d", calls)
			}
			if callbackPrompt != prompt {
				t.Fatalf("callback prompt differs from returned prompt: got %q want %q", callbackPrompt, prompt)
			}
		})
	}
}

func TestRenderersNotifyRenderResultCallbackForEmptyPrompt(t *testing.T) {
	t.Parallel()

	renderer := &gaictx.SimpleRenderer{}
	called := false
	if err := renderer.SetRenderResultCallback(context.Background(), func(parts []gaictx.Part, prompt string) {
		called = true
		if len(parts) != 0 || prompt != "" {
			t.Fatalf("unexpected empty render result: parts=%d prompt=%q", len(parts), prompt)
		}
	}); err != nil {
		t.Fatalf("SetRenderResultCallback failed: %v", err)
	}

	if _, err := renderer.Render(context.Background(), nil); err != nil {
		t.Fatalf("Render failed: %v", err)
	}
	if !called {
		t.Fatal("expected callback for empty successful render")
	}
}

func TestRenderToolSignatures(t *testing.T) {
	t.Parallel()

	tools := []toolSignatureTestTool{
		{name: "weather", description: "Gets weather."},
		{name: "search", description: "Searches docs."},
	}
	want := "\n<tool name=\"search\">\n<description>Searches docs.</description>\n<signature>{\"type\":\"object\"}</signature>\n</tool>\n<tool name=\"weather\">\n<description>Gets weather.</description>\n<signature>{\"type\":\"object\"}</signature>\n</tool>"

	renderers := []tooldefinitions.ToolSignatureRenderer{
		&gaictx.XMLRenderer{},
		&gaictx.SimpleRenderer{},
	}
	for _, renderer := range renderers {
		renderTools := make([]gaictx.ToolSignature, 0, len(tools))
		for _, tool := range tools {
			renderTools = append(renderTools, tool)
		}
		got, err := renderer.RenderToolSignatures(renderTools)
		if err != nil {
			t.Fatalf("%T.RenderToolSignatures() error = %v", renderer, err)
		}
		if got != want {
			t.Fatalf("%T.RenderToolSignatures() = %q, want %q", renderer, got, want)
		}
	}
}

type failingRenderPart struct {
	err error
}

func (p failingRenderPart) Name() string {
	return "failing"
}

func (p failingRenderPart) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	return 0, nil
}

func (p failingRenderPart) Render(ctx context.Context) (gaictx.RenderNode, error) {
	return gaictx.RenderNode{}, p.err
}

type historyPartAdapter struct {
	contents []gaictx.Message
}

func (p *historyPartAdapter) Name() string {
	return "history"
}

func (p *historyPartAdapter) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	return 0, nil
}

func (p *historyPartAdapter) Render(ctx context.Context) (gaictx.RenderNode, error) {
	node := gaictx.RenderNode{Type: "history"}
	for _, message := range p.contents {
		part := gaictx.NewMessagePart(message.Role, message.Content)
		child, err := part.Render(ctx)
		if err != nil {
			return gaictx.RenderNode{}, err
		}
		node.Children = append(node.Children, child)
	}
	return node, nil
}

type toolSignatureTestTool struct {
	name, description string
}

func (t toolSignatureTestTool) Name() string        { return t.name }
func (t toolSignatureTestTool) Description() string { return t.description }
func (t toolSignatureTestTool) Params() ai.ToolParameters {
	return ai.ToolParameters{}
}
