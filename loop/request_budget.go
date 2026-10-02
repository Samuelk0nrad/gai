package loop

import (
	"context"
	"fmt"
	"reflect"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

// requestCheckpoint belongs only to a run. Its copied prefix is never part of
// persisted history, and failed/discarded attempts cannot replace it.
type requestCheckpoint struct {
	request   ai.AIRequest
	model     ai.Model
	modelName string
	counterID string
	input     int
}

type requestBudgetProvider interface{ RequestBudget() ai.RequestBudgetConfig }
type budgetAllocationSetter interface {
	SetBudgetAllocation(limit, outputReserve, additionalReserve int) error
}
type localCounterProvider interface{ TokenCounter() ai.TokenCounter }
type configuredCounterProvider interface{ ConfiguredTokenCounter() ai.TokenCounter }
type budgetCounterSetter interface{ SetBudgetTokenCounter(ai.TokenCounter) }

func (r *runExecution) prepareRequestBudget() error {
	if r.owner.RequestBudget != nil {
		r.budget = *r.owner.RequestBudget
	} else if provider, ok := r.owner.PromptBuilder.(requestBudgetProvider); ok {
		r.budget = provider.RequestBudget()
	}
	if err := r.budget.Validate(); err != nil {
		return err
	}
	r.counter = r.owner.TokenCounter
	if r.counter == nil {
		if provider, ok := r.owner.PromptBuilder.(configuredCounterProvider); ok {
			r.counter = provider.ConfiguredTokenCounter()
		} else if provider, ok := r.owner.PromptBuilder.(localCounterProvider); ok {
			r.counter = provider.TokenCounter()
		}
	}
	if r.counter == nil {
		if provider, ok := r.owner.Model.(ai.TokenCounterProvider); ok {
			r.counter = provider.TokenCounter()
		}
	}
	if r.counter == nil {
		r.counter = ai.TextTokenEstimator{}
	}
	if nilRuntimeDependency(r.counter) {
		return &ai.RequestCountError{Cause: fmt.Errorf("local counter is a typed nil")}
	}
	if setter, ok := r.owner.PromptBuilder.(budgetCounterSetter); ok {
		setter.SetBudgetTokenCounter(r.counter)
	} else if setter, ok := r.owner.PromptBuilder.(gaictx.TokenCounterSetter); ok {
		setter.SetTokenCounter(r.counter)
	}
	setter, configurable := r.owner.PromptBuilder.(budgetAllocationSetter)
	if !configurable {
		return nil
	}
	reserve := max(r.budget.OutputReserve, r.owner.MaxTokens)
	limit := r.budget.Limit
	if r.budget.InputLimit > 0 {
		inputWindow, err := ai.AddTokenCounts(r.budget.InputLimit, reserve)
		if err != nil {
			return &ai.RequestCountError{Cause: err}
		}
		if limit == 0 || inputWindow < limit {
			limit = inputWindow
		}
	}
	if limit == 0 {
		return setter.SetBudgetAllocation(0, reserve, 0)
	}
	choice := r.owner.ToolChoice
	if r.owner.ToolTransport == ToolTransportText || len(r.toolDefinitions) == 0 {
		choice = ai.ToolChoice{}
	}
	// Only native schemas/options and request-wide framing are extra allocation
	// costs. Fixed message content is reserved by the builder itself.
	overhead, err := ai.EstimateRequestTokens(r.ctx, ai.AIRequest{
		Tools: r.toolDefinitions, ToolChoice: choice,
		ResponseFormat: r.owner.ResponseFormat, Reasoning: r.owner.Reasoning,
	}, r.counter)
	if err != nil {
		return err
	}
	extra, err := ai.AddTokenCounts(overhead.InputTokens, r.budget.SafetyMargin)
	if err != nil {
		return &ai.RequestCountError{Cause: err}
	}
	return setter.SetBudgetAllocation(limit, reserve, extra)
}

func (r *runExecution) checkRequestBudget(ctx context.Context, request ai.AIRequest) (ai.RequestBudgetResult, error) {
	result := ai.RequestBudgetResult{}
	if r.budget.Limit == 0 && r.budget.InputLimit == 0 && r.budget.Mode == ai.RequestCountEstimate {
		return result, nil
	}
	var err error
	if r.budget.Mode == ai.RequestCountAccurate {
		result.Method, result.Fidelity = "provider_preflight", ai.TokenCountEstimated
		counter, ok := r.owner.Model.(ai.InputTokenCounter)
		if !ok {
			err = &ai.InputTokenCountUnsupportedError{Model: ai.ModelName(r.owner.Model), Reason: "model has no complete-request counter"}
		} else {
			result.InputTokens, err = counter.CountInputTokens(ctx, request.Copy())
			if err == nil {
				_, err = ai.AddTokenCounts(result.InputTokens)
			}
			if err == nil {
				err = ctx.Err()
			}
			if err != nil {
				err = &ai.RequestCountError{Cause: err}
			}
		}
	} else if r.checkpoint.matches(request, r.owner.Model, r.counter) {
		result = ai.RequestBudgetResult{
			Method: "usage_checkpoint", Fidelity: ai.TokenCountEstimated,
			CounterID: ai.RequestEstimateID + ":" + r.counter.ID(), CounterFidelity: r.counter.Fidelity(),
			CheckpointTokens: r.checkpoint.input,
		}
		result.MessageTokens, err = ai.EstimateMessageTokens(ctx, request.Messages[len(r.checkpoint.request.Messages):], r.counter)
		if err == nil {
			result.InputTokens, err = ai.AddTokenCounts(result.CheckpointTokens, result.MessageTokens)
			if err != nil {
				err = &ai.RequestCountError{Cause: err}
			}
		}
	} else {
		result, err = ai.EstimateRequestTokens(ctx, request, r.counter)
	}
	result.Limit, result.InputLimit = r.budget.Limit, r.budget.InputLimit
	result.OutputReserve, result.SafetyMargin = max(r.budget.OutputReserve, request.MaxTokens), r.budget.SafetyMargin
	result.OutputLimitUnknown = request.MaxTokens == 0
	if err == nil {
		result.TotalTokens, err = ai.AddTokenCounts(result.InputTokens, result.OutputReserve, result.SafetyMargin)
		if err != nil {
			err = &ai.RequestCountError{Cause: err}
		}
	}
	if err == nil && ((result.Limit > 0 && result.TotalTokens > result.Limit) ||
		(result.InputLimit > 0 && result.InputTokens+result.SafetyMargin > result.InputLimit)) {
		err = &ai.RequestBudgetExceededError{Budget: result}
	}
	if err == nil {
		err = ctx.Err()
	}
	gai.EmitObservation(ctx, r.owner.ObservationSink, gai.Observation{
		Name: "loop_request_budget", Source: "loop:request_budget", Err: err,
		Fields: map[string]any{
			"input_tokens": result.InputTokens, "total_tokens": result.TotalTokens,
			"method": result.Method, "fidelity": result.Fidelity, "counter_id": result.CounterID,
			"limit": result.Limit, "input_limit": result.InputLimit,
			"output_reserve": result.OutputReserve, "safety_margin": result.SafetyMargin,
			"message_tokens": result.MessageTokens, "tool_tokens": result.ToolTokens,
			"option_tokens": result.OptionTokens, "framing_tokens": result.FramingTokens,
			"checkpoint_tokens": result.CheckpointTokens, "output_limit_unknown": result.OutputLimitUnknown,
		},
	})
	return result, err
}

func (c *requestCheckpoint) matches(request ai.AIRequest, model ai.Model, counter ai.TokenCounter) bool {
	if c == nil || counter == nil || c.counterID != ai.RequestEstimateID+":"+counter.ID() ||
		c.modelName != ai.ModelName(model) || !sameModel(c.model, model) || len(request.Messages) < len(c.request.Messages) {
		return false
	}
	if !reflect.DeepEqual(c.request.Messages, request.Messages[:len(c.request.Messages)]) {
		return false
	}
	base, current := c.request, request
	base.Messages, current.Messages = nil, nil
	return reflect.DeepEqual(base, current)
}

func sameModel(a, b ai.Model) bool {
	if a == nil || b == nil || reflect.TypeOf(a) != reflect.TypeOf(b) || !reflect.TypeOf(a).Comparable() {
		return false
	}
	return a == b
}

func nilRuntimeDependency(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	}
	return false
}
