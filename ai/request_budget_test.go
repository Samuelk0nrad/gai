package ai

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

type requestTestCounter struct {
	texts []string
	count func(context.Context, string) (int, error)
}

func (*requestTestCounter) ID() string                   { return "test/request-v1" }
func (*requestTestCounter) Fidelity() TokenCountFidelity { return TokenCountExact }
func (c *requestTestCounter) CountTokens(ctx context.Context, text string) (int, error) {
	c.texts = append(c.texts, text)
	if c.count != nil {
		return c.count(ctx, text)
	}
	return len(text), nil
}

func TestRequestEstimatePlainPartsAndFraming(t *testing.T) {
	counter := &requestTestCounter{}
	req := AIRequest{Messages: []Message{
		{Role: RoleUser, Parts: append(TextParts("first"), TextParts("second")...)},
		TextMessage(RoleAssistant, ""),
	}, MaxTokens: 999}
	result, err := EstimateRequestTokens(t.Context(), req, counter)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(counter.texts, "|") != "firstsecond|" || result.MessageTokens != 11 || result.FramingTokens != 11 || result.InputTokens != 22 {
		t.Fatalf("counted %q: %+v", counter.texts, result)
	}
	if result.CounterFidelity != TokenCountExact || result.Fidelity != TokenCountEstimated || result.CounterID != RequestEstimateID+":"+counter.ID() {
		t.Fatalf("text and request fidelity conflated: %+v", result)
	}
	addition, err := EstimateMessageTokens(t.Context(), req.Messages, &requestTestCounter{})
	if err != nil || addition != 19 {
		t.Fatalf("addition = %d, %v", addition, err)
	}
}

func TestRequestEstimateOpaqueAndStructuredMessagesUseCanonicalEnvelope(t *testing.T) {
	for _, message := range []Message{
		{Role: RoleUser},
		{Role: RoleUser, Parts: []ContentPart{{Kind: ContentJSON, JSON: json.RawMessage(`{"value":3}`)}}},
		{Role: RoleUser, Parts: TextParts("visible"), Extensions: []Extension{{Namespace: "test", Type: "state", Data: json.RawMessage(`{"opaque":true}`)}}},
		{Role: RoleUser, Parts: []ContentPart{{Kind: ContentText, Text: "visible", Extensions: []Extension{{Namespace: "test", Type: "state", Data: json.RawMessage(`{"opaque":true}`)}}}}},
	} {
		counter := &requestTestCounter{}
		result, err := EstimateRequestTokens(t.Context(), AIRequest{Messages: []Message{message}}, counter)
		if err != nil {
			t.Fatal(err)
		}
		payload, err := json.Marshal(message)
		if err != nil {
			t.Fatal(err)
		}
		if len(counter.texts) != 1 || counter.texts[0] != string(payload) || result.FramingTokens != 3 || result.InputTokens != len(payload)+3 {
			t.Fatalf("projection %q: %+v", counter.texts, result)
		}
	}
}

func TestRequestEstimateRejectsUnavailableFailedAndOverflowingCounts(t *testing.T) {
	failure := errors.New("counter failed")
	var missing *requestTestCounter
	for _, counter := range []TokenCounter{
		nil, missing,
		&requestTestCounter{count: func(context.Context, string) (int, error) { return 0, failure }},
		&requestTestCounter{count: func(context.Context, string) (int, error) { return -1, nil }},
		&requestTestCounter{count: func(context.Context, string) (int, error) { return int(^uint(0) >> 1), nil }},
	} {
		_, err := EstimateRequestTokens(t.Context(), AIRequest{Messages: []Message{TextMessage(RoleUser, "input")}}, counter)
		if !errors.Is(err, ErrRequestCountFailed) {
			t.Fatalf("error = %v", err)
		}
	}
	_, err := EstimateRequestTokens(t.Context(), AIRequest{Messages: []Message{TextMessage(RoleUser, "input")}}, &requestTestCounter{count: func(context.Context, string) (int, error) { return 0, failure }})
	if !errors.Is(err, failure) {
		t.Fatalf("lost cause: %v", err)
	}
	if _, err := EstimateMessageTokens(t.Context(), nil, nil); !errors.Is(err, ErrRequestCountFailed) {
		t.Fatalf("missing addition counter: %v", err)
	}
}

func TestRequestEstimateCancellationBeforeAndDuringCounting(t *testing.T) {
	for _, during := range []bool{false, true} {
		ctx, cancel := context.WithCancel(t.Context())
		counter := &requestTestCounter{count: func(context.Context, string) (int, error) { cancel(); return 1, nil }}
		if !during {
			cancel()
		}
		_, err := EstimateRequestTokens(ctx, AIRequest{Messages: []Message{TextMessage(RoleUser, "input")}}, counter)
		if !errors.Is(err, context.Canceled) || !errors.Is(err, ErrRequestCountFailed) || (!during && len(counter.texts) != 0) {
			t.Fatalf("during=%v calls=%d error=%v", during, len(counter.texts), err)
		}
	}
}

func TestRequestBudgetConfigurationAndArithmetic(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, config := range []RequestBudgetConfig{
		{Limit: -1}, {InputLimit: -1}, {OutputReserve: -1}, {SafetyMargin: -1}, {Mode: 255},
		{OutputReserve: maxInt, SafetyMargin: 1},
	} {
		if !errors.Is(config.Validate(), ErrInvalidRequestBudget) {
			t.Fatalf("accepted %+v", config)
		}
	}
	if err := (RequestBudgetConfig{}).Validate(); err != nil {
		t.Fatal(err)
	}
	if sum, err := AddTokenCounts(maxInt-1, 1); err != nil || sum != maxInt {
		t.Fatalf("boundary = %d, %v", sum, err)
	}
	if _, err := AddTokenCounts(maxInt, 1); err == nil {
		t.Fatal("accepted overflow")
	}
}
