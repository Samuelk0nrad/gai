package mocks

import (
	"context"
	"errors"
	"strings"

	"github.com/lace-ai/gai/ai"
)

type MockModelResponse struct {
	Res ai.AIResponse
	Err error
}

type MockModel struct {
	ModelName         string
	Count             int
	Responses         []MockModelResponse
	TokenCounterValue ai.TokenCounter
}

func (m *MockModel) Name() string {
	return m.ModelName
}

var ErrNoMockResponses = errors.New("mock model: no scripted responses left")

func (m *MockModel) Generate(ctx context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
	if m.Count >= len(m.Responses) {
		return nil, ErrNoMockResponses
	}
	r := m.Responses[m.Count]
	m.Count++
	return &r.Res, r.Err
}

func (m *MockModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)

	go func() {
		defer close(out)

		res, err := m.Generate(ctx, req)
		if err != nil {
			out <- ai.Token{Type: ai.TokenTypeErr, Err: err}
			return
		}
		if res == nil || (res.Text == "" && res.Reasoning == "") {
			return
		}

		if res.Reasoning != "" {
			out <- ai.Token{Type: ai.TokenTypeThought, Text: res.Reasoning, Data: []byte(res.Reasoning), TokenUsage: res.ReasoningTokens}
		}
		if res.Text != "" {
			out <- ai.Token{Type: ai.TokenTypeText, Data: []byte(res.Text)}
		}
	}()

	return out
}

func (m *MockModel) Close() error {
	return nil
}

func (m *MockModel) TokenCounter() ai.TokenCounter {
	if m.TokenCounterValue != nil {
		return m.TokenCounterValue
	}
	return &MockTokenCounter{}
}

type MockTokenCounter struct {
	IDValue    string
	Count      int
	CountCalls int
	Err        error
}

func (*MockTokenCounter) Fidelity() ai.TokenCountFidelity { return ai.TokenCountEstimated }

func (t *MockTokenCounter) ID() string {
	if t.IDValue != "" {
		return t.IDValue
	}
	return "mock.counter"
}

func (t *MockTokenCounter) Tokenize(ctx context.Context, text string) ([]string, error) {
	if t.Err != nil {
		return nil, t.Err
	}
	return strings.Fields(text), nil
}

func (t *MockTokenCounter) CountTokens(ctx context.Context, text string) (int, error) {
	t.CountCalls++
	if t.Err != nil {
		return 0, t.Err
	}
	if t.Count > 0 {
		return t.Count, nil
	}
	tokens, err := t.Tokenize(ctx, text)
	if err != nil {
		return 0, err
	}
	return len(tokens), nil
}
