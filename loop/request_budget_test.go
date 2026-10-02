package loop_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

// Each local text/projection call costs one token. Plain-message framing
// remains visible separately, so integration tests can detect double counting.
type requestUnitCounter struct{}

func (requestUnitCounter) ID() string                                       { return "test/request-unit-v1" }
func (requestUnitCounter) Fidelity() ai.TokenCountFidelity                  { return ai.TokenCountExact }
func (requestUnitCounter) CountTokens(context.Context, string) (int, error) { return 1, nil }

type requestFailingCounter struct{ err error }

func (requestFailingCounter) ID() string                                         { return "test/failing-counter-v1" }
func (requestFailingCounter) Fidelity() ai.TokenCountFidelity                    { return ai.TokenCountEstimated }
func (c requestFailingCounter) CountTokens(context.Context, string) (int, error) { return 0, c.err }

type requestSelectionCounter struct {
	id    string
	count int
}

func (c *requestSelectionCounter) ID() string                    { return c.id }
func (*requestSelectionCounter) Fidelity() ai.TokenCountFidelity { return ai.TokenCountEstimated }
func (c *requestSelectionCounter) CountTokens(context.Context, string) (int, error) {
	return c.count, nil
}

type requestCounterModel struct {
	*scriptedStreamModel
	counter ai.TokenCounter
}

func (m *requestCounterModel) TokenCounter() ai.TokenCounter { return m.counter }

type requestBudgetSource struct{ calls atomic.Int32 }

func (*requestBudgetSource) Name() string { return "request-budget-source" }
func (s *requestBudgetSource) Function(context.Context, int) (gaictx.Part, error) {
	s.calls.Add(1)
	return gaictx.NewTextPart("selected context"), nil
}

type requestAllocationSource struct{ budgets []int }

func (*requestAllocationSource) Name() string { return "allocation-recorder" }
func (s *requestAllocationSource) Function(_ context.Context, budget int) (gaictx.Part, error) {
	s.budgets = append(s.budgets, budget)
	return gaictx.NewTextPart("selected context"), nil
}

type requestPreflightModel struct {
	*scriptedStreamModel
	count   int
	err     error
	cancel  context.CancelFunc
	mutate  bool
	counted []ai.AIRequest
}

func (m *requestPreflightModel) CountInputTokens(ctx context.Context, request ai.AIRequest) (int, error) {
	m.counted = append(m.counted, request.Copy())
	if m.cancel != nil {
		m.cancel()
		return 0, ctx.Err()
	}
	if m.mutate {
		request.Messages[0].Parts[0].Text = "counter argument was mutated"
		if len(request.Tools) > 0 {
			request.Tools[0].Parameters[0] = '!'
		}
		if len(request.ToolChoice.Names) > 0 {
			request.ToolChoice.Names[0] = "counter-mutated-tool"
		}
		if len(request.ResponseFormat.Schema) > 0 {
			request.ResponseFormat.Schema[0] = '!'
		}
	}
	return m.count, m.err
}

func requestToolToken(id string) ai.Token {
	return ai.Token{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{
		ID: id, Type: "function", Name: "echo", Args: json.RawMessage(`{"text":"payload"}`),
	}}}
}

func requestReportedUsage(input int) ai.Token {
	return ai.Token{Completion: &ai.Completion{UsageReported: true, Usage: ai.Usage{InputTokens: input, OutputTokens: 2}}}
}

func requestFinalToken() ai.Token {
	return ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "done"}}
}

func requestBudgetSnapshots(events []loop.Event) []*ai.RequestBudgetResult {
	var budgets []*ai.RequestBudgetResult
	for _, event := range events {
		if event.Type == loop.EventIterationDone && event.Iteration != nil {
			budgets = append(budgets, event.Iteration.RequestBudget)
		}
	}
	return budgets
}

func TestRequestBudgetRejectsCompleteInitialInputBeforeGeneration(t *testing.T) {
	// Raw text fits comfortably; the finalized canonical envelopes do not.
	for _, input := range []struct {
		name       string
		definition gaictx.Definition
	}{
		{"system", gaictx.Definition{SystemInstructions: []gaictx.Part{gaictx.NewTextPart("required system")}, PromptInput: gaictx.PromptInput{User: ai.TextParts("question")}}},
		{"user", gaictx.Definition{PromptInput: gaictx.PromptInput{User: ai.TextParts("required question")}}},
		{"context", gaictx.Definition{PromptInput: gaictx.PromptInput{User: ai.TextParts("question"), Context: []gaictx.Part{gaictx.NewTextPart("required context")}}}},
	} {
		t.Run(input.name, func(t *testing.T) {
			model := &scriptedStreamModel{sequences: [][]ai.Token{{requestFinalToken()}}}
			builder := gaictx.New(input.definition)
			l := loop.New(model, nil, builder, nil)
			l.TokenCounter = requestUnitCounter{}
			l.RequestBudget = &ai.RequestBudgetConfig{Limit: 3}
			events := collectLoopEvents(t, l, t.Context())
			var budgetErr *ai.RequestBudgetExceededError
			if err := loopError(events); !errors.As(err, &budgetErr) || !errors.Is(err, ai.ErrRequestBudgetExceeded) {
				t.Fatalf("error=%v, want typed budget rejection", err)
			}
			if len(model.Requests()) != 0 || len(l.Iterations) != 0 {
				t.Fatal("over-budget required input reached generation or persistence")
			}
			if budgetErr.Budget.Fidelity != ai.TokenCountEstimated || budgetErr.Budget.CounterFidelity != ai.TokenCountExact || budgetErr.Budget.InputTokens <= 3 {
				t.Fatalf("budget diagnostics=%+v", budgetErr.Budget)
			}
		})
	}
}

func TestRequestBudgetCountsNativeSchemasOptionsAndOutputCapacityOnce(t *testing.T) {
	for _, tc := range []struct {
		name      string
		budget    ai.RequestBudgetConfig
		wantError bool
	}{
		{"exact total", ai.RequestBudgetConfig{Limit: 21, OutputReserve: 4, SafetyMargin: 2}, false},
		{"total too small", ai.RequestBudgetConfig{Limit: 20, OutputReserve: 4, SafetyMargin: 2}, true},
		{"exact input", ai.RequestBudgetConfig{InputLimit: 12, OutputReserve: 4, SafetyMargin: 2}, false},
		{"input too small", ai.RequestBudgetConfig{InputLimit: 11, OutputReserve: 4, SafetyMargin: 2}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &scriptedStreamModel{sequences: [][]ai.Token{{requestFinalToken()}}}
			l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
			l.TokenCounter = requestUnitCounter{}
			l.RequestBudget = &tc.budget
			l.MaxTokens = 9
			l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceAuto}
			l.ResponseFormat = ai.ResponseFormat{Type: ai.ResponseFormatJSONSchema, Name: "answer", Schema: json.RawMessage(`{"type":"object"}`)}
			l.Reasoning = ai.ReasoningConfig{Effort: ai.ReasoningEffortLow}
			events := collectLoopEvents(t, l, t.Context())
			if err := loopError(events); errors.Is(err, ai.ErrRequestBudgetExceeded) != tc.wantError || (!tc.wantError && err != nil) {
				t.Fatalf("error=%v, want rejection=%v", err, tc.wantError)
			}
			var result *ai.RequestBudgetResult
			if tc.wantError {
				var exceeded *ai.RequestBudgetExceededError
				if !errors.As(loopError(events), &exceeded) {
					t.Fatal("missing typed diagnostics")
				}
				result = &exceeded.Budget
				if len(model.Requests()) != 0 {
					t.Fatal("rejected request generated")
				}
			} else {
				result = requestBudgetSnapshots(events)[0]
			}
			// One text count + four message framing tokens + schemas/options + reply prefix.
			if result.MessageTokens != 1 || result.ToolTokens != 1 || result.OptionTokens != 1 || result.FramingTokens != 7 || result.InputTokens != 10 || result.OutputReserve != 9 || result.SafetyMargin != 2 || result.TotalTokens != 21 {
				t.Fatalf("budget=%+v, want 10 input + 9 output + 2 margin", result)
			}
		})
	}
}

func TestRequestBudgetTextToolProtocolHasNoNativeSchemaCost(t *testing.T) {
	model := &scriptedStreamModel{sequences: [][]ai.Token{{requestFinalToken()}}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, &stubPromptBuilder{userPrompt: "question", contextText: "rendered echo schema"}, nil)
	l.ToolTransport = loop.ToolTransportText
	l.TokenCounter = requestUnitCounter{}
	l.RequestBudget = &ai.RequestBudgetConfig{Limit: 12}
	events := collectLoopEvents(t, l, t.Context())
	if err := loopError(events); err != nil {
		t.Fatal(err)
	}
	request := model.Requests()[0]
	budget := requestBudgetSnapshots(events)[0]
	if len(request.Tools) != 0 || budget.ToolTokens != 0 || budget.OptionTokens != 0 || budget.InputTokens != 8 {
		t.Fatalf("text transport request=%+v budget=%+v", request, budget)
	}
}

func TestRequestBudgetRejectsToolConversationGrowthBeforeNextGeneration(t *testing.T) {
	model := &scriptedStreamModel{sequences: [][]ai.Token{{requestToolToken("call-1")}, {requestFinalToken()}}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.TokenCounter = requestUnitCounter{}
	l.RequestBudget = &ai.RequestBudgetConfig{Limit: 9}
	events := collectLoopEvents(t, l, t.Context())
	var exceeded *ai.RequestBudgetExceededError
	if err := loopError(events); !errors.As(err, &exceeded) {
		t.Fatalf("error=%v", err)
	}
	if len(model.Requests()) != 1 || len(l.Iterations) != 1 || exceeded.Budget.InputTokens != 11 || exceeded.Budget.MessageTokens != 3 {
		t.Fatalf("requests=%d iterations=%d budget=%+v", len(model.Requests()), len(l.Iterations), exceeded.Budget)
	}
}

func TestRequestBudgetUsageCheckpointReplacesEstimateAndNextReport(t *testing.T) {
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{requestToolToken("call-1"), requestReportedUsage(40)},
		{requestToolToken("call-2"), requestReportedUsage(7)},
		{requestFinalToken()},
	}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.TokenCounter = requestUnitCounter{}
	l.RequestBudget = &ai.RequestBudgetConfig{Limit: 100}
	events := collectLoopEvents(t, l, t.Context())
	if err := loopError(events); err != nil {
		t.Fatal(err)
	}
	budgets := requestBudgetSnapshots(events)
	if len(budgets) != 3 || budgets[0].Method != "local_estimate" || budgets[0].InputTokens != 9 || budgets[1].Method != "usage_checkpoint" || budgets[1].CheckpointTokens != 40 || budgets[1].MessageTokens != 2 || budgets[1].InputTokens != 42 || budgets[2].CheckpointTokens != 7 || budgets[2].MessageTokens != 2 || budgets[2].InputTokens != 9 {
		t.Fatalf("budgets=%+v", budgets)
	}
	if budgets[1].ToolTokens != 0 || budgets[1].OptionTokens != 0 || budgets[1].FramingTokens != 0 || budgets[1].Fidelity != ai.TokenCountEstimated {
		t.Fatalf("checkpoint double-counted existing overhead: %+v", budgets[1])
	}
}

func TestRequestBudgetDistinguishesReportedZeroAbsentAndLaterMetadata(t *testing.T) {
	for _, tc := range []struct {
		name        string
		completions []ai.Token
		limit       int
		wantInput   int
		wantMethod  string
		wantError   bool
	}{
		{"reported zero", []ai.Token{requestReportedUsage(0)}, 9, 2, "usage_checkpoint", false},
		{"absent usage", []ai.Token{{Completion: &ai.Completion{FinishReason: "tool_calls"}}}, 9, 11, "local_estimate", true},
		{"later unreported metadata", []ai.Token{requestReportedUsage(40), {Completion: &ai.Completion{FinishReason: "tool_calls"}}}, 100, 42, "usage_checkpoint", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			first := append([]ai.Token{requestToolToken("call-1")}, tc.completions...)
			model := &scriptedStreamModel{sequences: [][]ai.Token{first, {requestFinalToken()}}}
			l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
			l.TokenCounter = requestUnitCounter{}
			l.RequestBudget = &ai.RequestBudgetConfig{Limit: tc.limit}
			events := collectLoopEvents(t, l, t.Context())
			if err := loopError(events); errors.Is(err, ai.ErrRequestBudgetExceeded) != tc.wantError || (!tc.wantError && err != nil) {
				t.Fatalf("error=%v", err)
			}
			var budget *ai.RequestBudgetResult
			if tc.wantError {
				var exceeded *ai.RequestBudgetExceededError
				if !errors.As(loopError(events), &exceeded) {
					t.Fatal("missing typed rejection")
				}
				budget = &exceeded.Budget
				if len(model.Requests()) != 1 {
					t.Fatal("absent usage incorrectly permitted generation")
				}
			} else {
				budget = requestBudgetSnapshots(events)[1]
			}
			if budget.InputTokens != tc.wantInput || budget.Method != tc.wantMethod {
				t.Fatalf("budget=%+v", budget)
			}
		})
	}
}

func TestRequestBudgetRejectedAttemptUsageCannotReplaceAcceptedCheckpoint(t *testing.T) {
	for _, rejected := range []string{"retry", "discard"} {
		t.Run(rejected, func(t *testing.T) {
			first := []ai.Token{requestFinalToken(), requestReportedUsage(999)}
			if rejected == "retry" {
				first = append(first, ai.Token{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("retry this attempt")}})
			}
			model := &scriptedStreamModel{sequences: [][]ai.Token{first, {requestToolToken("call-1"), requestReportedUsage(40)}, {requestFinalToken()}}}
			l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
			l.TokenCounter = requestUnitCounter{}
			l.RequestBudget = &ai.RequestBudgetConfig{Limit: 100}
			if rejected == "retry" {
				l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}
			} else {
				l.ToolTransport = loop.ToolTransportText
				l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired}
			}
			events := collectLoopEvents(t, l, t.Context())
			if err := loopError(events); err != nil {
				t.Fatal(err)
			}
			if len(model.Requests()) != 3 || len(l.Iterations) != 2 {
				t.Fatalf("requests=%d accepted=%d", len(model.Requests()), len(l.Iterations))
			}
			budgets := requestBudgetSnapshots(events)
			if budgets[0].Method != "local_estimate" || budgets[1].Method != "usage_checkpoint" || budgets[1].CheckpointTokens != 40 || budgets[1].InputTokens != 42 {
				t.Fatalf("accepted budgets=%+v", budgets)
			}
			for _, event := range events {
				if event.Type == loop.EventRetry || event.Type == loop.EventDiscard {
					if event.Iteration == nil || event.Iteration.Usage.InputTokens != 999 {
						t.Fatalf("billed attempt metadata lost: %+v", event)
					}
				}
			}
		})
	}
}

func TestRequestBudgetRequiredNativeChoiceInvalidatesCheckpoint(t *testing.T) {
	model := &scriptedStreamModel{sequences: [][]ai.Token{{requestToolToken("call-1"), requestReportedUsage(40)}, {requestFinalToken()}}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired}
	l.TokenCounter = requestUnitCounter{}
	l.RequestBudget = &ai.RequestBudgetConfig{Limit: 12}
	events := collectLoopEvents(t, l, t.Context())
	if err := loopError(events); err != nil {
		t.Fatal(err)
	}
	budgets := requestBudgetSnapshots(events)
	if budgets[1].Method != "local_estimate" || budgets[1].CheckpointTokens != 0 || budgets[1].InputTokens != 12 {
		t.Fatalf("required-to-auto did not invalidate checkpoint: %+v", budgets[1])
	}
}

func TestRequestBudgetKeepsPriorCheckpointAcrossLaterRejectedAttempts(t *testing.T) {
	for _, rejected := range []string{"retry", "discard"} {
		t.Run(rejected, func(t *testing.T) {
			failed := []ai.Token{requestFinalToken(), requestReportedUsage(999), {Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("temporary")}}}
			if rejected == "discard" {
				wrongTool := requestToolToken("rejected-call")
				wrongTool.Part.ToolCall.Name = "unavailable"
				failed = []ai.Token{wrongTool, requestReportedUsage(999)}
			}
			model := &scriptedStreamModel{sequences: [][]ai.Token{
				{requestToolToken("call-1"), requestReportedUsage(40)},
				failed,
				{requestToolToken("call-2")},
				{requestFinalToken()},
			}}
			l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
			l.TokenCounter = requestUnitCounter{}
			l.RequestBudget = &ai.RequestBudgetConfig{Limit: 44}
			if rejected == "retry" {
				l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}
			} else {
				l.ToolTransport = loop.ToolTransportText
				l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"echo"}}
			}
			events := collectLoopEvents(t, l, t.Context())
			if err := loopError(events); err != nil {
				t.Fatal(err)
			}
			budgets := requestBudgetSnapshots(events)
			if len(model.Requests()) != 4 || len(budgets) != 3 || budgets[1].Method != "usage_checkpoint" || budgets[1].InputTokens != 42 || budgets[2].Method != "usage_checkpoint" || budgets[2].CheckpointTokens != 40 || budgets[2].InputTokens != 44 || budgets[2].MessageTokens != 4 {
				t.Fatalf("requests=%d budgets=%+v", len(model.Requests()), budgets)
			}
			requests := model.Requests()
			if !reflect.DeepEqual(requests[1], requests[2]) {
				t.Fatal("rejected attempt changed next request")
			}
		})
	}
}

func TestRequestBudgetRetriesDoNotRebuildSourcesOrDuplicateUserInput(t *testing.T) {
	source := &requestBudgetSource{}
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("temporary")}}},
		{requestToolToken("call-1"), requestReportedUsage(40)},
		{requestFinalToken()},
	}}
	builder := gaictx.New(gaictx.Definition{ContextSources: []gaictx.ContextSource{source}, PromptInput: gaictx.PromptInput{User: ai.TextParts("unique user input")}})
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, builder, nil)
	l.TokenCounter = requestUnitCounter{}
	l.RequestBudget = &ai.RequestBudgetConfig{Limit: 100}
	l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}
	events := collectLoopEvents(t, l, t.Context())
	if err := loopError(events); err != nil {
		t.Fatal(err)
	}
	if source.calls.Load() != 1 {
		t.Fatalf("source builds=%d", source.calls.Load())
	}
	requests := model.Requests()
	if len(requests) != 3 || !reflect.DeepEqual(requests[0], requests[1]) {
		t.Fatal("retry changed prepared input")
	}
	for _, request := range requests {
		users := 0
		for _, message := range request.Messages {
			if message.Text() == "unique user input" {
				users++
			}
		}
		if users != 1 {
			t.Fatalf("user input appears %d times: %+v", users, request.Messages)
		}
	}
}

func TestRequestBudgetAccuratePreflightUsesFinalRequestAndDetachedArgument(t *testing.T) {
	model := &requestPreflightModel{scriptedStreamModel: &scriptedStreamModel{sequences: [][]ai.Token{{requestFinalToken()}}}, count: 11, mutate: true}
	builder := &stubPromptBuilder{systemPrompt: "system", contextText: "selected context", userPrompt: "question"}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, builder, nil)
	l.TokenCounter = requestUnitCounter{}
	l.RequestBudget = &ai.RequestBudgetConfig{Limit: 19, OutputReserve: 2, SafetyMargin: 1, Mode: ai.RequestCountAccurate}
	l.MaxTokens = 7
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceAuto}
	l.ResponseFormat = ai.ResponseFormat{Type: ai.ResponseFormatJSONSchema, Name: "answer", Schema: json.RawMessage(`{"type":"object"}`)}
	l.Reasoning = ai.ReasoningConfig{Effort: ai.ReasoningEffortHigh}
	events := collectLoopEvents(t, l, t.Context())
	if err := loopError(events); err != nil {
		t.Fatal(err)
	}
	requests := model.Requests()
	if len(model.counted) != 1 || len(requests) != 1 || !reflect.DeepEqual(model.counted[0], requests[0]) {
		t.Fatalf("counted=%+v generated=%+v", model.counted, requests)
	}
	if requests[0].MaxTokens != 7 || requests[0].ResponseFormat.Type != ai.ResponseFormatJSONSchema || requests[0].Reasoning.Effort != ai.ReasoningEffortHigh || len(requests[0].Tools) != 1 {
		t.Fatalf("preflight missed execution settings: %+v", requests[0])
	}
	budget := requestBudgetSnapshots(events)[0]
	if budget.Method != "provider_preflight" || budget.InputTokens != 11 || budget.TotalTokens != 19 || budget.FramingTokens != 0 || budget.ToolTokens != 0 || budget.OptionTokens != 0 {
		t.Fatalf("preflight double-counted overhead: %+v", budget)
	}
}

func TestRequestBudgetDefaultNeverCallsAccurateCapability(t *testing.T) {
	model := &requestPreflightModel{scriptedStreamModel: &scriptedStreamModel{sequences: [][]ai.Token{{requestFinalToken()}}}, err: errors.New("must not be called")}
	l := loop.New(model, nil, testPromptBuilder(), nil)
	l.TokenCounter = requestUnitCounter{}
	l.RequestBudget = &ai.RequestBudgetConfig{Limit: 100}
	if err := loopError(collectLoopEvents(t, l, t.Context())); err != nil {
		t.Fatal(err)
	}
	if len(model.counted) != 0 || len(model.Requests()) != 1 {
		t.Fatal("default mode invoked accurate preflight")
	}
}

func TestRequestBudgetAccurateRetriesCountEachFinalRequestWithoutSourceRebuild(t *testing.T) {
	source := &requestBudgetSource{}
	model := &requestPreflightModel{scriptedStreamModel: &scriptedStreamModel{sequences: [][]ai.Token{
		{{Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("temporary")}}},
		{requestToolToken("call-1"), requestReportedUsage(40)},
		{requestFinalToken(), requestReportedUsage(7)},
	}}, count: 5}
	builder := gaictx.New(gaictx.Definition{ContextSources: []gaictx.ContextSource{source}, PromptInput: gaictx.PromptInput{User: ai.TextParts("question")}})
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, builder, nil)
	l.TokenCounter = requestUnitCounter{}
	l.RequestBudget = &ai.RequestBudgetConfig{Limit: 5, Mode: ai.RequestCountAccurate}
	l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 1}
	events := collectLoopEvents(t, l, t.Context())
	if err := loopError(events); err != nil {
		t.Fatal(err)
	}
	if source.calls.Load() != 1 || len(model.counted) != 3 || !reflect.DeepEqual(model.counted, model.Requests()) {
		t.Fatalf("source builds=%d counted=%+v generated=%+v", source.calls.Load(), model.counted, model.Requests())
	}
	for _, budget := range requestBudgetSnapshots(events) {
		if budget.Method != "provider_preflight" || budget.InputTokens != 5 || budget.TotalTokens != 5 {
			t.Fatalf("accurate mode reused estimated checkpoint: %+v", budget)
		}
	}
	if l.Iterations[0].Usage.InputTokens != 40 || l.Iterations[1].Usage.InputTokens != 7 {
		t.Fatal("reported usage failed to replace preflight accounting")
	}
}

func TestRequestBudgetLocalCountFailureStopsGeneration(t *testing.T) {
	countErr := errors.New("local encoding count failed")
	model := &scriptedStreamModel{sequences: [][]ai.Token{{requestFinalToken()}}}
	l := loop.New(model, nil, testPromptBuilder(), nil)
	l.TokenCounter = requestFailingCounter{err: countErr}
	l.RequestBudget = &ai.RequestBudgetConfig{Limit: 100}
	err := loopError(collectLoopEvents(t, l, t.Context()))
	if !errors.Is(err, ai.ErrRequestCountFailed) || !errors.Is(err, countErr) {
		t.Fatalf("count error=%v", err)
	}
	if len(model.Requests()) != 0 {
		t.Fatal("failed local count reached generation")
	}
}

func TestRequestBudgetAccuratePreflightFailuresStopGeneration(t *testing.T) {
	providerErr := errors.New("provider count unavailable")
	for _, tc := range []struct {
		name                string
		count               int
		err                 error
		unsupported, cancel bool
		want                error
	}{
		{"unsupported", 0, nil, true, false, ai.ErrInputTokenCountUnsupported},
		{"provider failure", 0, providerErr, false, false, ai.ErrRequestCountFailed},
		{"negative result", -1, nil, false, false, ai.ErrRequestCountFailed},
		{"over budget", 101, nil, false, false, ai.ErrRequestBudgetExceeded},
		{"canceled", 0, nil, false, true, context.Canceled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			stream := &scriptedStreamModel{sequences: [][]ai.Token{{requestFinalToken()}}}
			model := &requestPreflightModel{scriptedStreamModel: stream, count: tc.count, err: tc.err}
			if tc.cancel {
				model.cancel = cancel
			}
			var selected ai.Model = model
			if tc.unsupported {
				selected = stream
			}
			l := loop.New(selected, nil, testPromptBuilder(), nil)
			l.TokenCounter = requestUnitCounter{}
			l.RequestBudget = &ai.RequestBudgetConfig{Limit: 100, Mode: ai.RequestCountAccurate}
			l.RetryPolicy = &loop.RetryPolicy{MaxRetries: 2}
			events := collectLoopEvents(t, l, ctx)
			var terminal loop.Event
			for _, event := range events {
				if event.Type == loop.EventError || event.Type == loop.EventCanceled {
					terminal = event
				}
			}
			if !errors.Is(terminal.Err, tc.want) {
				t.Fatalf("terminal=%+v, want %v", terminal, tc.want)
			}
			if tc.err != nil && !errors.Is(terminal.Err, tc.err) {
				t.Fatal("underlying provider cause lost")
			}
			if tc.cancel && terminal.Type != loop.EventCanceled {
				t.Fatalf("terminal type=%s", terminal.Type)
			}
			if len(stream.Requests()) != 0 || len(l.Iterations) != 0 || (!tc.unsupported && len(model.counted) != 1) {
				t.Fatal("failed preflight retried, generated, or persisted")
			}
			for _, event := range events {
				if event.Type == loop.EventRetry || event.Type == loop.EventIterationDone || event.Type == loop.EventToken {
					t.Fatalf("unexpected event after preflight failure: %s", event.Type)
				}
			}
		})
	}
}

func TestRequestBudgetEventSnapshotsCannotMutateAccounting(t *testing.T) {
	model := &scriptedStreamModel{sequences: [][]ai.Token{{requestToolToken("call-1"), requestReportedUsage(40)}, {requestFinalToken()}}}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.TokenCounter = requestUnitCounter{}
	l.RequestBudget = &ai.RequestBudgetConfig{Limit: 100}
	var second *ai.RequestBudgetResult
	for event := range l.Run(t.Context()) {
		if event.Type == loop.EventError {
			t.Fatal(event.Err)
		}
		if event.Type != loop.EventIterationDone {
			continue
		}
		if event.IterationCount == 1 {
			event.Iteration.RequestBudget.InputTokens = 999
			event.Iteration.RequestBudget.Limit = 1
			event.Iteration.Usage.InputTokens = 999
			event.Iteration.Conversation[0].Parts[0].Text = "mutated event"
		} else {
			copy := *event.Iteration.RequestBudget
			second = &copy
		}
	}
	if second == nil || second.Method != "usage_checkpoint" || second.InputTokens != 42 || second.CheckpointTokens != 40 || l.Iterations[0].RequestBudget.InputTokens != 9 || l.Iterations[0].RequestBudget.Limit != 100 || l.Iterations[0].Usage.InputTokens != 40 {
		t.Fatalf("second=%+v persisted=%+v", second, l.Iterations)
	}
}

func TestRequestBudgetSequentialRunsKeepConfiguredBuilderDefaults(t *testing.T) {
	for _, tc := range []struct {
		name            string
		firstPolicy     *ai.RequestBudgetConfig
		firstAllocation int
	}{
		{"inherited window", nil, 52},
		{"explicit disabled window", &ai.RequestBudgetConfig{}, -1},
		{"temporary input-only window", &ai.RequestBudgetConfig{InputLimit: 20}, 12},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source := &requestAllocationSource{}
			builder := gaictx.New(gaictx.Definition{
				TokenBudget:    100,
				ContextSources: []gaictx.ContextSource{source},
				PromptInput:    gaictx.PromptInput{User: ai.TextParts("question")},
			})
			configured := ai.RequestBudgetConfig{Limit: 100}
			model := &scriptedStreamModel{sequences: [][]ai.Token{{requestFinalToken()}, {requestFinalToken()}}}
			l := loop.New(model, nil, builder, nil)
			l.TokenCounter = requestUnitCounter{}
			l.RequestBudget = tc.firstPolicy
			l.MaxTokens = 40
			first := collectLoopEvents(t, l, t.Context())
			if err := loopError(first); err != nil {
				t.Fatal(err)
			}
			if got := builder.RequestBudget(); got != configured {
				t.Fatalf("run allocation overwrote builder defaults: got=%+v want=%+v", got, configured)
			}
			if len(source.budgets) != 1 {
				t.Fatalf("source evaluations=%d", len(source.budgets))
			}
			if tc.firstAllocation >= 0 {
				if source.budgets[0] != tc.firstAllocation {
					t.Fatalf("first source budget=%d, want %d", source.budgets[0], tc.firstAllocation)
				}
				if requestBudgetSnapshots(first)[0].OutputReserve != 40 {
					t.Fatal("first request omitted its current output capacity")
				}
			} else if source.budgets[0] <= configured.Limit {
				t.Fatalf("explicit zero policy retained inherited source limit: %d", source.budgets[0])
			}

			// Reusing a Loop explicitly clears its accepted conversation. The same
			// builder should inherit its configured window afresh on the next run.
			l.Iterations = nil
			l.RequestBudget = nil
			l.MaxTokens = 10
			second := collectLoopEvents(t, l, t.Context())
			if err := loopError(second); err != nil {
				t.Fatal(err)
			}
			if got := builder.RequestBudget(); got != configured {
				t.Fatalf("second run changed builder defaults: %+v", got)
			}
			if len(source.budgets) != 2 || source.budgets[1] != 82 {
				t.Fatalf("source allocations=%v, want final allocation82", source.budgets)
			}
			budget := requestBudgetSnapshots(second)[0]
			if budget.Limit != 100 || budget.InputLimit != 0 || budget.OutputReserve != 10 {
				t.Fatalf("prior reserve/disabled/synthetic window leaked into next run: %+v", budget)
			}
			if len(model.Requests()) != 2 || model.Requests()[1].MaxTokens != 10 {
				t.Fatal("second generation did not use current output configuration")
			}
		})
	}
}

func TestRequestBudgetDirectLoopSelectsConfiguredCounterBeforeProviderFallback(t *testing.T) {
	for _, selected := range []string{"provider", "builder definition", "builder setter", "builder clear", "loop override"} {
		t.Run(selected, func(t *testing.T) {
			providerCounter := &requestSelectionCounter{id: "test/provider-v1", count: 7}
			definitionCounter := &requestSelectionCounter{id: "test/builder-definition-v1", count: 11}
			setterCounter := &requestSelectionCounter{id: "test/builder-setter-v1", count: 13}
			loopCounter := &requestSelectionCounter{id: "test/loop-override-v1", count: 17}
			definition := gaictx.Definition{PromptInput: gaictx.PromptInput{User: ai.TextParts("question")}}
			want := providerCounter
			if selected != "provider" {
				definition.TokenCounter = definitionCounter
				want = definitionCounter
			}
			builder := gaictx.New(definition)
			if selected == "builder setter" || selected == "loop override" {
				builder.SetTokenCounter(setterCounter)
				want = setterCounter
			}
			if selected == "builder clear" {
				builder.SetTokenCounter(nil)
				want = providerCounter
			}
			model := &requestCounterModel{scriptedStreamModel: &scriptedStreamModel{sequences: [][]ai.Token{{requestFinalToken()}}}, counter: providerCounter}
			l := loop.New(model, nil, builder, nil)
			if selected == "loop override" {
				l.TokenCounter = loopCounter
				want = loopCounter
			}
			l.RequestBudget = &ai.RequestBudgetConfig{Limit: 1000}
			events := collectLoopEvents(t, l, t.Context())
			if err := loopError(events); err != nil {
				t.Fatal(err)
			}
			budget := requestBudgetSnapshots(events)[0]
			if budget.CounterID != ai.RequestEstimateID+":"+want.ID() || budget.InputTokens != want.count+7 {
				t.Fatalf("selected budget=%+v, want counter=%s and input=%d", budget, want.ID(), want.count+7)
			}
		})
	}
}

func TestRequestBudgetSequentialDirectLoopsReselectModelCounter(t *testing.T) {
	builder := gaictx.New(gaictx.Definition{PromptInput: gaictx.PromptInput{User: ai.TextParts("question")}})
	firstCounter := &requestSelectionCounter{id: "test/provider-v1", count: 7}
	secondCounter := &requestSelectionCounter{id: "test/provider-v2", count: 19}
	firstModel := &requestCounterModel{scriptedStreamModel: &scriptedStreamModel{sequences: [][]ai.Token{{requestFinalToken()}}}, counter: firstCounter}
	l := loop.New(firstModel, nil, builder, nil)
	l.RequestBudget = &ai.RequestBudgetConfig{Limit: 1000}
	first := collectLoopEvents(t, l, t.Context())
	if err := loopError(first); err != nil {
		t.Fatal(err)
	}
	if budget := requestBudgetSnapshots(first)[0]; budget.CounterID != ai.RequestEstimateID+":"+firstCounter.ID() || budget.InputTokens != 14 {
		t.Fatalf("first budget=%+v", budget)
	}
	l.Iterations = nil
	l.Model = &requestCounterModel{scriptedStreamModel: &scriptedStreamModel{sequences: [][]ai.Token{{requestFinalToken()}}}, counter: secondCounter}
	second := collectLoopEvents(t, l, t.Context())
	if err := loopError(second); err != nil {
		t.Fatal(err)
	}
	if budget := requestBudgetSnapshots(second)[0]; budget.CounterID != ai.RequestEstimateID+":"+secondCounter.ID() || budget.InputTokens != 26 {
		t.Fatalf("previous runtime counter injection leaked into next run: %+v", budget)
	}
}
