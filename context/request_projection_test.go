package context

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/lace-ai/gai/ai"
)

type requestProjectionCounter struct {
	texts []string
	count func(context.Context, string) (int, error)
}

func (*requestProjectionCounter) ID() string                      { return "test/request-projection-bytes" }
func (*requestProjectionCounter) Fidelity() ai.TokenCountFidelity { return ai.TokenCountEstimated }
func (c *requestProjectionCounter) CountTokens(ctx context.Context, text string) (int, error) {
	c.texts = append(c.texts, text)
	if c.count != nil {
		return c.count(ctx, text)
	}
	return len(text), ctx.Err()
}

type requestProjectionRenderer func(context.Context, []Part) (string, error)

func (r requestProjectionRenderer) Render(ctx context.Context, parts []Part) (string, error) {
	return r(ctx, parts)
}

type requestProjectionLegacySource struct {
	part    Part
	budgets []int
}

func (*requestProjectionLegacySource) Name() string { return "legacy-projection-source" }
func (s *requestProjectionLegacySource) Function(_ context.Context, budget int) (Part, error) {
	s.budgets = append(s.budgets, budget)
	return s.part, nil
}

type requestProjectionSource struct {
	part        Part
	legacyCalls int
	budgets     []int
	costs       []int
	selectPart  func(context.Context, int, func(context.Context, Part) (int, error)) (Part, int, error)
}

var _ ContextSourceWithBudgetProjection = (*requestProjectionSource)(nil)

func (*requestProjectionSource) Name() string { return "projected-source" }
func (s *requestProjectionSource) Function(context.Context, int) (Part, error) {
	s.legacyCalls++
	return s.part, nil
}
func (s *requestProjectionSource) FunctionWithBudget(ctx context.Context, budget int, project func(context.Context, Part) (int, error)) (Part, int, error) {
	s.budgets = append(s.budgets, budget)
	if s.selectPart != nil {
		return s.selectPart(ctx, budget, project)
	}
	cost, err := project(ctx, s.part)
	s.costs = append(s.costs, cost)
	if err != nil {
		return nil, 0, err
	}
	if cost > budget {
		return nil, 0, nil
	}
	return s.part, cost, nil
}

func TestRequestProjectionDebitsActualFixedAndLegacySourceMessages(t *testing.T) {
	for _, tc := range []struct {
		name     string
		renderer Renderer
		first    int
		next     int
	}{
		{"XML", XMLRenderer{}, 49, 26},
		{"simple", SimpleRenderer{}, 89, 84},
		{"custom render-only", requestProjectionRenderer(func(ctx context.Context, parts []Part) (string, error) {
			text, err := (SimpleRenderer{}).Render(ctx, parts)
			return "prefix:" + text, err
		}), 75, 63},
	} {
		t.Run(tc.name, func(t *testing.T) {
			counter := &requestProjectionCounter{}
			first := &requestProjectionLegacySource{part: NewTextPart("x")}
			next := &requestProjectionLegacySource{}
			builder := New(Definition{
				Renderer: tc.renderer, TokenCounter: counter, TokenBudget: 200,
				SystemInstructions: []Part{NewTextPart("s")},
				ContextSources:     []ContextSource{first, next},
				PromptInput:        PromptInput{User: ai.TextParts("u"), Context: []Part{NewTextPart("c")}},
			})
			// The loop's extra allowance includes the 3-token response prefix.
			if err := builder.SetBudgetAllocation(150, 10, 3); err != nil {
				t.Fatal(err)
			}
			parts, err := builder.BuildContext(t.Context())
			if err != nil || len(parts) != 2 || len(first.budgets) != 1 || first.budgets[0] != tc.first || len(next.budgets) != 1 || next.budgets[0] != tc.next {
				t.Fatalf("BuildContext = %v, %v; source allowances = %v then %v, want %d then %d", parts, err, first.budgets, next.budgets, tc.first, tc.next)
			}
			request, err := builder.BuildRequest(t.Context(), nil)
			if err != nil {
				t.Fatal(err)
			}
			final, err := ai.EstimateRequestTokens(t.Context(), request, counter)
			if err != nil || final.InputTokens+10 != 150-tc.next {
				t.Fatalf("allocation differs from emitted request: %+v, %v, remaining %d", final, err, tc.next)
			}
		})
	}
}

type requestProjectionCanonicalPart struct{ messages []ai.Message }

func (*requestProjectionCanonicalPart) Name() string { return "canonical-projection-part" }
func (*requestProjectionCanonicalPart) Tokens(context.Context, ai.TokenCounter) (int, error) {
	panic("complete source total must not be recounted through Part.Tokens")
}
func (*requestProjectionCanonicalPart) Render(context.Context) (RenderNode, error) {
	panic("canonical content must bypass the renderer")
}
func (p *requestProjectionCanonicalPart) ConversationMessages() []ai.Message {
	return ai.CloneMessages(p.messages)
}

func TestRequestProjectionConsumesCanonicalSourceTotalOnce(t *testing.T) {
	counter := &requestProjectionCounter{}
	part := &requestProjectionCanonicalPart{messages: []ai.Message{ai.TextMessage(ai.RoleAssistant, "h")}}
	source := &requestProjectionSource{part: part}
	next := &requestProjectionLegacySource{}
	builder := New(Definition{TokenCounter: counter, ContextSources: []ContextSource{source, next}, PromptInput: PromptInput{User: ai.TextParts("u")}})
	if err := builder.SetBudgetAllocation(100, 0, 3); err != nil {
		t.Fatal(err)
	}
	parts, err := builder.BuildContext(t.Context())
	if err != nil || len(parts) != 1 || source.legacyCalls != 0 || len(source.budgets) != 1 || source.budgets[0] != 92 || len(source.costs) != 1 || source.costs[0] != 5 || len(next.budgets) != 1 || next.budgets[0] != 87 {
		t.Fatalf("canonical handoff = %v, %v; source %+v, next allowances %v", parts, err, source, next.budgets)
	}
	if len(counter.texts) != 2 || counter.texts[0] != "u" || counter.texts[1] != "h" {
		t.Fatalf("selected canonical content was recounted: %q", counter.texts)
	}
	request, err := builder.BuildRequest(t.Context(), nil)
	if err != nil || len(request.Messages) != 2 || request.Messages[0].Role != ai.RoleAssistant || request.Messages[0].Text() != "h" || len(counter.texts) != 2 {
		t.Fatalf("canonical request = %+v, %v; counted text %q", request, err, counter.texts)
	}
}

func TestRequestProjectionProjectsLegacyCalculatedMessageCost(t *testing.T) {
	counter := &requestProjectionCounter{}
	// The legacy capability supplies Part.Tokens (1), which excludes the four
	// framing tokens. Request allocation must debit the emitted message cost (5).
	source := &countedTestSource{part: NewMessagePart(ai.TextMessage(ai.RoleAssistant, "h")), tokens: 1}
	next := &requestProjectionLegacySource{}
	builder := New(Definition{TokenCounter: counter, ContextSources: []ContextSource{source, next}, PromptInput: PromptInput{User: ai.TextParts("u")}})
	if err := builder.SetBudgetAllocation(100, 0, 3); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.BuildContext(t.Context()); err != nil || source.countedCalls != 1 || source.legacyCalls != 0 || source.budget != 92 || len(next.budgets) != 1 || next.budgets[0] != 87 {
		t.Fatalf("legacy message handoff: source %+v, next allowances %v, error %v", source, next.budgets, err)
	}
	if len(counter.texts) != 2 || counter.texts[0] != "u" || counter.texts[1] != "h" {
		t.Fatalf("legacy message projection = %q", counter.texts)
	}
}

func TestRequestProjectionCapabilityOnlyRunsWithRequestAllocation(t *testing.T) {
	for _, tc := range []struct {
		name      string
		override  bool
		limit     int
		noCounter bool
		projected bool
		allowance int
	}{
		{name: "direct builder preserves legacy", limit: 100, allowance: 95},
		{name: "request allocation", override: true, limit: 100, projected: true, allowance: 92},
		{name: "budget disabled", override: true, allowance: 1000000},
		{name: "counter missing", override: true, limit: 100, noCounter: true, allowance: 97},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source := &requestProjectionSource{part: NewTextPart("x")}
			builder := New(Definition{TokenBudget: tc.limit, TokenCounter: &requestProjectionCounter{}, ContextSources: []ContextSource{source}, PromptInput: PromptInput{User: ai.TextParts("u")}})
			if tc.override {
				if err := builder.SetBudgetAllocation(tc.limit, 0, 3); err != nil {
					t.Fatal(err)
				}
			}
			if tc.noCounter {
				builder.counter = nil
			}
			parts, err := builder.BuildContext(t.Context())
			if err != nil || len(parts) != 1 {
				t.Fatalf("BuildContext = %v, %v", parts, err)
			}
			if tc.projected {
				if source.legacyCalls != 0 || len(source.budgets) != 1 || source.budgets[0] != tc.allowance || len(source.costs) != 1 || source.costs[0] != 23 {
					t.Fatalf("projection invocation = %+v", source)
				}
			} else if source.legacyCalls != 1 || len(source.budgets) != 0 || len(source.costs) != 0 {
				t.Fatalf("disabled capability ran: %+v", source)
			}
		})
	}
}

func TestRequestProjectionCapabilityReevaluatesEachBuild(t *testing.T) {
	source := &requestProjectionSource{part: NewTextPart("x")}
	builder := New(Definition{TokenCounter: &requestProjectionCounter{}, ContextSources: []ContextSource{source}, PromptInput: PromptInput{User: ai.TextParts("u")}})
	if err := builder.SetBudgetAllocation(100, 0, 3); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.BuildContext(t.Context()); err != nil {
		t.Fatal(err)
	}
	builder.SetInput(PromptInput{User: ai.TextParts("uu")})
	if _, err := builder.BuildContext(t.Context()); err != nil || source.legacyCalls != 0 || len(source.budgets) != 2 || source.budgets[0] != 92 || source.budgets[1] != 91 || len(source.costs) != 2 || source.costs[0] != 23 || source.costs[1] != 23 {
		t.Fatalf("per-build projection = %+v, %v", source, err)
	}
}

func TestRequestProjectionValidatesReturnedCompleteCost(t *testing.T) {
	for _, absent := range []bool{false, true} {
		t.Run(map[bool]string{false: "negative selected cost", true: "absent part ignores cost"}[absent], func(t *testing.T) {
			source := &requestProjectionSource{selectPart: func(context.Context, int, func(context.Context, Part) (int, error)) (Part, int, error) {
				if absent {
					return nil, -1, nil
				}
				return NewTextPart("x"), -1, nil
			}}
			next := &requestProjectionLegacySource{}
			builder := New(Definition{ContextSources: []ContextSource{source, next}})
			if err := builder.SetBudgetAllocation(100, 0, 3); err != nil {
				t.Fatal(err)
			}
			_, err := builder.BuildContext(t.Context())
			if absent {
				if err != nil || len(next.budgets) != 1 || next.budgets[0] != 97 {
					t.Fatalf("absent source changed allocation: %v, next %v", err, next.budgets)
				}
			} else if !errors.Is(err, ErrInvalidTokenCount) || len(next.budgets) != 0 {
				t.Fatalf("invalid count = %v, next allowances %v", err, next.budgets)
			}
			if source.legacyCalls != 0 || len(source.budgets) != 1 {
				t.Fatalf("projection source was retried or fell back: %+v", source)
			}
		})
	}
}

func TestRequestProjectionPropagatesFailuresAndCancellation(t *testing.T) {
	failure := errors.New("request projection failed")
	for _, origin := range []string{"system", "input context", "legacy source", "projected source"} {
		for _, operation := range []string{"render failure", "count failure", "render cancellation", "count cancellation"} {
			t.Run(origin+"/"+operation, func(t *testing.T) {
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				want := failure
				if strings.HasSuffix(operation, "cancellation") {
					want = context.Canceled
				}
				counter := &requestProjectionCounter{count: func(ctx context.Context, text string) (int, error) {
					if strings.Contains(text, "candidate") {
						switch operation {
						case "count failure":
							return 0, failure
						case "count cancellation":
							cancel()
						}
					}
					return len(text), nil
				}}
				renderer := requestProjectionRenderer(func(ctx context.Context, parts []Part) (string, error) {
					text, err := (XMLRenderer{}).Render(ctx, parts)
					if strings.Contains(text, "candidate") {
						switch operation {
						case "render failure":
							return "", failure
						case "render cancellation":
							cancel()
						}
					}
					return text, err
				})
				legacy := &requestProjectionLegacySource{part: NewTextPart("candidate")}
				projected := &requestProjectionSource{part: NewTextPart("candidate")}
				next := &requestProjectionLegacySource{}
				definition := Definition{Renderer: renderer, TokenCounter: counter, PromptInput: PromptInput{User: ai.TextParts("u")}, ContextSources: []ContextSource{next}}
				switch origin {
				case "system":
					definition.SystemInstructions = []Part{NewTextPart("candidate")}
				case "input context":
					definition.PromptInput.Context = []Part{NewTextPart("candidate")}
				case "legacy source":
					definition.ContextSources = []ContextSource{legacy, next}
				case "projected source":
					definition.ContextSources = []ContextSource{projected, next}
				}
				builder := New(definition)
				if err := builder.SetBudgetAllocation(1000, 0, 3); err != nil {
					t.Fatal(err)
				}
				if _, err := builder.BuildContext(ctx); !errors.Is(err, want) || len(next.budgets) != 0 {
					t.Fatalf("projection error = %v, want %v; later source allowances %v", err, want, next.budgets)
				}
				if origin == "legacy source" && len(legacy.budgets) != 1 {
					t.Fatalf("legacy source invocations = %v", legacy.budgets)
				}
				if origin == "projected source" && (projected.legacyCalls != 0 || len(projected.budgets) != 1) {
					t.Fatalf("projection source retried or fell back: %+v", projected)
				}
			})
		}
	}
}
