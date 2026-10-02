package context

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/lace-ai/gai/ai"
)

type noRecountPart struct{ TextPart }

func (noRecountPart) Tokens(context.Context, ai.TokenCounter) (int, error) {
	panic("builder recounted a source's already counted part")
}

type countedTestSource struct {
	part         Part
	tokens       int
	err          error
	counter      ai.TokenCounter
	budget       int
	countedCalls int
	legacyCalls  int
	cancel       context.CancelFunc
}

func (*countedTestSource) Name() string { return "counted" }
func (s *countedTestSource) SetTokenCounter(counter ai.TokenCounter) {
	s.counter = counter
}
func (s *countedTestSource) Function(context.Context, int) (Part, error) {
	s.legacyCalls++
	return s.part, s.err
}
func (s *countedTestSource) FunctionWithTokens(_ context.Context, budget int) (Part, int, error) {
	s.countedCalls++
	s.budget = budget
	if s.cancel != nil {
		s.cancel()
	}
	return s.part, s.tokens, s.err
}

func TestBuildContextUsesSourceCountWithInjectedCounter(t *testing.T) {
	t.Parallel()
	source := &countedTestSource{part: noRecountPart{NewTextPart("one two three")}, tokens: 3}
	next := &testContextSource{name: "next"}
	sink := &debugEventSink{}
	counter := debugTestTokenCounter{}
	builder := New(Definition{
		TokenBudget:        10,
		OutputTokenReserve: 2,
		SystemInstructions: []Part{NewTextPart("system")},
		ContextSources:     []ContextSource{source, next},
		TokenCounter:       counter,
		ObservationSink:    sink,
	})
	parts, err := builder.BuildContext(t.Context())
	if err != nil || len(parts) != 2 {
		t.Fatalf("BuildContext = %v, %v", parts, err)
	}
	if source.counter != counter || source.budget != 7 || next.budget != 4 || source.countedCalls != 1 || source.legacyCalls != 0 {
		t.Fatalf("wrong counter or budget handoff: source=%+v, next=%+v", source, next)
	}
	if _, err := builder.BuildRequest(t.Context(), nil); err != nil {
		t.Fatal(err)
	}
	for _, event := range sink.events {
		if event.Name == "prompt_builder_source_included" && event.Fields["source"] == source.Name() {
			if event.Fields["tokens"] != 3 || event.Fields["tokens_counted"] != true || event.Fields["remaining_tokens"] != 4 {
				t.Fatalf("wrong source accounting observation: %v", event.Fields)
			}
			return
		}
	}
	t.Fatal("missing source accounting observation")
}

func TestBuildContextRejectsInvalidSourceCount(t *testing.T) {
	t.Parallel()
	source := &countedTestSource{part: noRecountPart{NewTextPart("text")}, tokens: -1}
	sink := &debugEventSink{}
	builder := New(Definition{TokenBudget: 10, ContextSources: []ContextSource{source}, ObservationSink: sink})
	if _, err := builder.BuildContext(t.Context()); !errors.Is(err, ErrInvalidTokenCount) {
		t.Fatalf("negative source count error = %v", err)
	}
	for _, event := range sink.events {
		if event.Name == "prompt_builder_token_count_failed" && event.Fields["source"] == source.Name() {
			return
		}
	}
	t.Fatal("missing invalid count observation")
}

func TestBuildContextPreservesCountedSourceFailures(t *testing.T) {
	t.Parallel()
	failure := errors.New("source count failed")
	for _, test := range []struct {
		name   string
		err    error
		cancel bool
		want   error
	}{
		{name: "source error", err: failure, want: failure},
		{name: "cancellation after source returns", cancel: true, want: context.Canceled},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			source := &countedTestSource{part: noRecountPart{NewTextPart("text")}, tokens: 1, err: test.err}
			if test.cancel {
				source.cancel = cancel
			}
			next := &countedTestSource{}
			builder := New(Definition{TokenBudget: 10, ContextSources: []ContextSource{source, next}})
			if _, err := builder.BuildContext(ctx); !errors.Is(err, test.want) {
				t.Fatalf("BuildContext error = %v, want %v", err, test.want)
			}
			if source.legacyCalls != 0 || next.countedCalls != 0 || next.legacyCalls != 0 {
				t.Fatal("failed source fell back or continued building")
			}
		})
	}
}

func TestBuildContextUsesLegacySourceWhenCountingDisabled(t *testing.T) {
	t.Parallel()
	for _, budget := range []int{0, -1, 10} {
		t.Run(fmt.Sprint(budget), func(t *testing.T) {
			source := &countedTestSource{part: noRecountPart{NewTextPart("text")}, tokens: -1}
			builder := New(Definition{TokenBudget: budget, ContextSources: []ContextSource{source}})
			if budget > 0 {
				builder.counter = nil
			}
			parts, err := builder.BuildContext(t.Context())
			if err != nil || len(parts) != 1 || source.countedCalls != 0 || source.legacyCalls != 1 {
				t.Fatalf("uncounted BuildContext = %v, %v; source=%+v", parts, err, source)
			}
		})
	}
}

func TestBuildContextIgnoresCountForAbsentPart(t *testing.T) {
	t.Parallel()
	source := &countedTestSource{tokens: -1}
	next := &testContextSource{name: "next"}
	builder := New(Definition{TokenBudget: 10, ContextSources: []ContextSource{source, next}})
	if _, err := builder.BuildContext(t.Context()); err != nil || next.budget != 10 {
		t.Fatalf("absent part changed remaining budget: %d, %v", next.budget, err)
	}
}
