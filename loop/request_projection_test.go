package loop_test

import (
	"context"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

type requestProjectionCounter struct{}

func (requestProjectionCounter) ID() string                      { return "test/projection-runes-v1" }
func (requestProjectionCounter) Fidelity() ai.TokenCountFidelity { return ai.TokenCountExact }
func (requestProjectionCounter) CountTokens(_ context.Context, text string) (int, error) {
	return utf8.RuneCountInString(text), nil
}

type requestProjectionSource struct {
	name            string
	candidate       gaictx.Part
	legacyCalls     int
	projectionCalls int
	budget          int
	projected       int
}

func (s *requestProjectionSource) Name() string { return s.name }
func (s *requestProjectionSource) Function(_ context.Context, budget int) (gaictx.Part, error) {
	s.legacyCalls++
	s.budget = budget
	return s.candidate, nil
}
func (s *requestProjectionSource) FunctionWithBudget(ctx context.Context, budget int, project func(context.Context, gaictx.Part) (int, error)) (gaictx.Part, int, error) {
	s.projectionCalls++
	s.budget = budget
	count, err := project(ctx, s.candidate)
	if err != nil {
		return nil, 0, err
	}
	s.projected = count
	if count > budget {
		return nil, 0, nil
	}
	return s.candidate, count, nil
}

func projectionNamedPart(t *testing.T, name, value string) gaictx.Part {
	t.Helper()
	part, err := gaictx.NewNamedPart(name, value)
	if err != nil {
		t.Fatal(err)
	}
	return part
}

func projectionRenderedMessage(t *testing.T, renderer gaictx.Renderer, role ai.Role, parts ...gaictx.Part) ai.Message {
	t.Helper()
	text, err := renderer.Render(t.Context(), parts)
	if err != nil {
		t.Fatal(err)
	}
	return ai.TextMessage(role, text)
}

func TestRequestBudgetProjectsRenderedCandidatesAfterRequiredInput(t *testing.T) {
	for _, tc := range []struct {
		name     string
		renderer gaictx.Renderer
	}{
		{"XML escaping", &gaictx.XMLRenderer{}},
		{"simple labels", &gaictx.SimpleRenderer{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			counter := requestProjectionCounter{}
			candidate := projectionNamedPart(t, "chosen_payload", `<tag a="value"> & '雪'`)
			source := &requestProjectionSource{name: "unrelated_source_identity_much_longer_than_part_name", candidate: candidate}
			probe := &requestProjectionSource{name: "next_source", candidate: projectionNamedPart(t, "e", "x")}
			systemParts := []gaictx.Part{projectionNamedPart(t, "policy", "required <& policy"), gaictx.NewTextPart("second instruction")}
			requiredContext := projectionNamedPart(t, "required_context", "required <& context")
			canonicalContext := ai.TextMessage(ai.RoleAssistant, "canonical selected context")
			user := ai.Message{Role: ai.RoleUser, Parts: append(ai.TextParts("current "), ai.TextParts("<& question")...)}
			expected := ai.AIRequest{MaxTokens: 8, Messages: []ai.Message{
				projectionRenderedMessage(t, tc.renderer, ai.RoleSystem, gaictx.NewSystemPart(systemParts)),
				projectionRenderedMessage(t, tc.renderer, ai.RoleUser, candidate),
				projectionRenderedMessage(t, tc.renderer, ai.RoleUser, requiredContext),
				canonicalContext,
				user,
			}}
			if tc.name == "XML escaping" && (!strings.Contains(expected.Messages[1].Text(), "&lt;") || !strings.Contains(expected.Messages[1].Text(), "&amp;")) {
				t.Fatalf("fixture did not expand escaped XML: %q", expected.Messages[1].Text())
			}
			candidateCost, err := ai.EstimateMessageTokens(t.Context(), expected.Messages[1:2], counter)
			if err != nil {
				t.Fatal(err)
			}
			estimate, err := ai.EstimateRequestTokens(t.Context(), expected, counter)
			if err != nil {
				t.Fatal(err)
			}
			window := estimate.InputTokens + 8 + 5
			builder := gaictx.New(gaictx.Definition{
				Renderer:           tc.renderer,
				SystemInstructions: systemParts,
				ContextSources:     []gaictx.ContextSource{source, probe},
				PromptInput:        gaictx.PromptInput{User: user.Parts, Context: []gaictx.Part{requiredContext, gaictx.NewMessagePart(canonicalContext)}},
			})
			model := &scriptedStreamModel{sequences: [][]ai.Token{{requestFinalToken()}}}
			l := loop.New(model, nil, builder, nil)
			l.MaxTokens = 8
			l.TokenCounter = counter
			l.RequestBudget = &ai.RequestBudgetConfig{Limit: window, OutputReserve: 8, SafetyMargin: 5}
			events := collectLoopEvents(t, l, t.Context())
			if err := loopError(events); err != nil {
				t.Fatalf("exact-fit rendered candidate rejected: %v", err)
			}
			if source.projectionCalls != 1 || source.legacyCalls != 0 || source.budget != candidateCost || source.projected != candidateCost {
				t.Fatalf("projection did not reserve actual required rendering: source=%+v want candidate cost=%d", source, candidateCost)
			}
			if probe.projectionCalls != 1 || probe.legacyCalls != 0 || probe.budget != 0 || probe.projected <= 0 {
				t.Fatalf("selected rendered candidate did not spend its entire allocation: probe=%+v", probe)
			}
			requests := model.Requests()
			if len(requests) != 1 || !reflect.DeepEqual(requests[0], expected) {
				t.Fatalf("generated request differs from projected snapshot: actual=%+v expected=%+v", requests, expected)
			}
			budget := requestBudgetSnapshots(events)[0]
			if budget.InputTokens != estimate.InputTokens || budget.TotalTokens != window || budget.Limit != window || budget.OutputReserve != 8 || budget.SafetyMargin != 5 {
				t.Fatalf("allocation and final guard disagree: budget=%+v estimate=%+v", budget, estimate)
			}
		})
	}
}

func TestRequestBudgetProjectsCanonicalCandidateWithoutRenderingIt(t *testing.T) {
	message := ai.Message{Role: ai.RoleUser, Parts: []ai.ContentPart{{Kind: ai.ContentJSON, JSON: []byte(`{"selected":"<&雪"}`)}}}
	source := &requestProjectionSource{name: "structured_source", candidate: gaictx.NewMessagePart(message)}
	counter := requestProjectionCounter{}
	expected := ai.AIRequest{Messages: []ai.Message{message, ai.TextMessage(ai.RoleUser, "question")}}
	estimate, err := ai.EstimateRequestTokens(t.Context(), expected, counter)
	if err != nil {
		t.Fatal(err)
	}
	candidateCost, err := ai.EstimateMessageTokens(t.Context(), []ai.Message{message}, counter)
	if err != nil {
		t.Fatal(err)
	}
	builder := gaictx.New(gaictx.Definition{ContextSources: []gaictx.ContextSource{source}, PromptInput: gaictx.PromptInput{User: ai.TextParts("question")}})
	model := &scriptedStreamModel{sequences: [][]ai.Token{{requestFinalToken()}}}
	l := loop.New(model, nil, builder, nil)
	l.TokenCounter = counter
	l.RequestBudget = &ai.RequestBudgetConfig{Limit: estimate.InputTokens}
	events := collectLoopEvents(t, l, t.Context())
	if err := loopError(events); err != nil {
		t.Fatal(err)
	}
	if source.projectionCalls != 1 || source.legacyCalls != 0 || source.budget != candidateCost || source.projected != candidateCost {
		t.Fatalf("canonical projection=%+v", source)
	}
	if requests := model.Requests(); len(requests) != 1 || !reflect.DeepEqual(requests[0], expected) {
		t.Fatalf("canonical candidate changed: %+v", requests)
	}
}

func TestRequestBudgetProjectionIsDisabledWithExplicitZeroPolicy(t *testing.T) {
	source := &requestProjectionSource{name: "projection_source", candidate: projectionNamedPart(t, "payload", "selected text")}
	builder := gaictx.New(gaictx.Definition{TokenBudget: 1, ContextSources: []gaictx.ContextSource{source}, PromptInput: gaictx.PromptInput{User: ai.TextParts("question")}})
	model := &scriptedStreamModel{sequences: [][]ai.Token{{requestFinalToken()}}}
	l := loop.New(model, nil, builder, nil)
	l.TokenCounter = requestProjectionCounter{}
	l.RequestBudget = &ai.RequestBudgetConfig{}
	if err := loopError(collectLoopEvents(t, l, t.Context())); err != nil {
		t.Fatal(err)
	}
	if source.legacyCalls != 1 || source.projectionCalls != 0 || len(model.Requests()) != 1 {
		t.Fatalf("disabled allocation projected/rebuilt source: %+v", source)
	}
}
