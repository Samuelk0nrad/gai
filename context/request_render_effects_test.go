package context_test

import (
	"context"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

type renderEffectCounter struct{}

func (renderEffectCounter) ID() string                      { return "test/render-effects-runes-v1" }
func (renderEffectCounter) Fidelity() ai.TokenCountFidelity { return ai.TokenCountExact }
func (renderEffectCounter) CountTokens(_ context.Context, text string) (int, error) {
	return utf8.RuneCountInString(text), nil
}

type renderEffectLegacySource struct {
	part  gaictx.Part
	calls int
}

func (*renderEffectLegacySource) Name() string { return "legacy-source" }
func (s *renderEffectLegacySource) Function(context.Context, int) (gaictx.Part, error) {
	s.calls++
	return s.part, nil
}

type renderEffectProjectionSource struct {
	accepted, rejected                 gaictx.Part
	calls, legacyCalls                 int
	acceptedCost, rejectedCost, budget int
}

func (*renderEffectProjectionSource) Name() string { return "projection-source" }
func (s *renderEffectProjectionSource) Function(context.Context, int) (gaictx.Part, error) {
	s.legacyCalls++
	return s.accepted, nil
}
func (s *renderEffectProjectionSource) FunctionWithBudget(ctx context.Context, budget int, project func(context.Context, gaictx.Part) (int, error)) (gaictx.Part, int, error) {
	s.calls++
	s.budget = budget
	var err error
	s.rejectedCost, err = project(ctx, s.rejected)
	if err != nil {
		return nil, 0, err
	}
	s.acceptedCost, err = project(ctx, s.accepted)
	if err != nil {
		return nil, 0, err
	}
	if s.acceptedCost > budget {
		return nil, 0, nil
	}
	return s.accepted, s.acceptedCost, nil
}

type renderEffectSurface interface {
	gaictx.Renderer
	SetRenderResultCallback(context.Context, gaictx.RenderResultCallback) error
}

func renderEffectNamedPart(t *testing.T, name, text string) gaictx.Part {
	t.Helper()
	part, err := gaictx.NewNamedPart(name, text)
	if err != nil {
		t.Fatal(err)
	}
	return part
}

func renderEffectSinkEvents(sink *rendererObservationSink) map[string]int {
	counts := make(map[string]int)
	for _, event := range sink.events {
		if strings.HasPrefix(event.Name, "renderer_") {
			counts[event.Name]++
		}
	}
	return counts
}

func renderEffectTraceEvents(recorder *tracetest.SpanRecorder) map[string]int {
	counts := make(map[string]int)
	for _, span := range recorder.Ended() {
		for _, event := range span.Events() {
			if strings.HasPrefix(event.Name, "observation.renderer_") {
				counts[strings.TrimPrefix(event.Name, "observation.")]++
			}
		}
	}
	return counts
}

func TestRequestAllocationDoesNotPublishRendererEffects(t *testing.T) {
	for _, rendererName := range []string{"XML", "simple"} {
		for _, destination := range []string{"sink", "trace without sink"} {
			t.Run(rendererName+"/"+destination, func(t *testing.T) {
				ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{Prompt: gai.CaptureEnabled})
				var sink *rendererObservationSink
				var recorder *tracetest.SpanRecorder
				var endRoot func()
				if destination == "sink" {
					sink = &rendererObservationSink{}
				} else {
					recorder = tracetest.NewSpanRecorder()
					provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
					previous := otel.GetTracerProvider()
					otel.SetTracerProvider(provider)
					t.Cleanup(func() { otel.SetTracerProvider(previous); _ = provider.Shutdown(context.Background()) })
					rootCtx, root := provider.Tracer("render-effects-test").Start(ctx, "request")
					ctx, endRoot = rootCtx, func() { root.End() }
				}
				var renderer renderEffectSurface
				var observationSink gai.ObservationSink
				if sink != nil {
					observationSink = sink
				}
				if rendererName == "XML" {
					renderer = &gaictx.XMLRenderer{ObservationSink: observationSink}
				} else {
					renderer = &gaictx.SimpleRenderer{ObservationSink: observationSink}
				}
				var callbacks []string
				if err := renderer.SetRenderResultCallback(ctx, func(parts []gaictx.Part, prompt string) {
					if len(parts) != 1 {
						t.Errorf("callback received %d renderer parts, want one emitted message", len(parts))
						return
					}
					if strings.Contains(prompt, "rejected-candidate-secret") {
						t.Error("callback exposed an unselected candidate")
					}
					callbacks = append(callbacks, parts[0].Name())
				}); err != nil {
					t.Fatal(err)
				}
				legacy := &renderEffectLegacySource{part: renderEffectNamedPart(t, "legacy_payload", "legacy <& content")}
				projection := &renderEffectProjectionSource{
					accepted: renderEffectNamedPart(t, "selected_payload", "selected <& content"),
					rejected: renderEffectNamedPart(t, "rejected_payload", strings.Repeat("rejected-candidate-secret <& ", 100)),
				}
				builder := gaictx.New(gaictx.Definition{
					Renderer:           renderer,
					TokenCounter:       renderEffectCounter{},
					SystemInstructions: []gaictx.Part{gaictx.NewTextPart("required system one"), gaictx.NewTextPart("required system two")},
					ContextSources:     []gaictx.ContextSource{legacy, projection},
					PromptInput: gaictx.PromptInput{
						User:    ai.TextParts("question"),
						Context: []gaictx.Part{renderEffectNamedPart(t, "fixed_payload", "fixed <& context"), gaictx.NewMessagePart(ai.TextMessage(ai.RoleAssistant, "canonical context"))},
					},
				})
				if err := builder.SetBudgetAllocation(1000, 0, 3); err != nil {
					t.Fatal(err)
				}
				if _, err := builder.BuildContext(ctx); err != nil {
					t.Fatal(err)
				}
				if projection.calls != 1 || projection.legacyCalls != 0 || legacy.calls != 1 || projection.rejectedCost <= projection.budget || projection.acceptedCost > projection.budget {
					t.Fatalf("fixture did not probe an accepted and rejected candidate once: projection=%+v legacy=%+v", projection, legacy)
				}
				if len(callbacks) != 0 {
					t.Errorf("allocation published %d render callbacks before request assembly: %v", len(callbacks), callbacks)
				}
				var previewEvents map[string]int
				if sink != nil {
					previewEvents = renderEffectSinkEvents(sink)
				} else {
					previewEvents = renderEffectTraceEvents(recorder)
				}
				if len(previewEvents) != 0 {
					t.Errorf("allocation published renderer observations: %v", previewEvents)
				}
				request, err := builder.BuildRequest(ctx, nil)
				if err != nil {
					t.Fatal(err)
				}
				if endRoot != nil {
					endRoot()
				}
				if len(request.Messages) != 6 {
					t.Fatalf("emitted messages=%d, want system, two selected sources, fixed, canonical, user", len(request.Messages))
				}
				want := []string{"system", "legacy_payload", "selected_payload", "fixed_payload"}
				if len(callbacks) != len(want) {
					t.Errorf("render callbacks=%v, want once for each %v", callbacks, want)
				} else {
					for i, name := range want {
						if callbacks[i] != name {
							t.Errorf("callback %d=%q, want %q", i, callbacks[i], name)
						}
					}
				}
				var finalEvents map[string]int
				if sink != nil {
					finalEvents = renderEffectSinkEvents(sink)
				} else {
					finalEvents = renderEffectTraceEvents(recorder)
				}
				for _, name := range []string{"renderer_render_started", "renderer_part_rendered", "renderer_render_finished"} {
					if finalEvents[name] != 4 {
						t.Errorf("%s=%d, want four emitted render operations", name, finalEvents[name])
					}
				}
				if finalEvents["renderer_part_failed"] != 0 {
					t.Errorf("unexpected renderer failures: %v", finalEvents)
				}
			})
		}
	}
}
