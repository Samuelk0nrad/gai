package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/lace-ai/gai/ai"
)

const (
	maxErrorBodySize    = 1 << 20
	maxStreamRecordSize = 16 << 20
)

// Model is an immutable, concurrency-safe Ollama chat model.
type Model struct {
	name        string
	provider    *Provider
	options     Options
	toolSupport ai.FeatureSupport
}

var _ ai.Model = (*Model)(nil)
var _ ai.ModelDescriber = (*Model)(nil)

func (m *Model) Name() string { return m.name }

func (m *Model) Descriptor() ai.ModelDescriptor {
	return ai.ModelDescriptor{
		Model:            m.name,
		NativeMessages:   ai.FeatureSupportSupported,
		NativeTools:      m.toolSupport,
		ToolChoiceModes:  []ai.ToolChoiceMode{ai.ToolChoiceAuto},
		Usage:            ai.FeatureSupportSupported,
		FinishReason:     ai.FeatureSupportSupported,
		StreamingUsage:   ai.FeatureSupportUnsupported,
		JSONOutput:       ai.FeatureSupportUnsupported,
		JSONSchemaOutput: ai.FeatureSupportUnsupported,
		Reasoning:        ai.FeatureSupportUnsupported,
		ReasoningEffort:  ai.FeatureSupportUnsupported,
	}
}

type rawChatResponse struct {
	Model   string `json:"model"`
	Message struct {
		Role      string `json:"role"`
		Content   string `json:"content"`
		Thinking  string `json:"thinking"`
		ToolCalls []struct {
			ID       string `json:"id"`
			Function struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	} `json:"message"`
	Done            bool   `json:"done"`
	DoneReason      string `json:"done_reason"`
	PromptEvalCount *int   `json:"prompt_eval_count"`
	EvalCount       *int   `json:"eval_count"`
	Error           string `json:"error"`
}

func (m *Model) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)
	req = req.Copy()
	go func() {
		defer close(out)
		ctx, observation := ai.StartGenerationObservation(ctx, req, ai.GenerationConfig{Provider: "ollama", Model: m.name, Streaming: true, Sink: m.provider.debug})
		result := ai.GenerationResult{}
		emit := func(token ai.Token) bool {
			observation.ObserveToken(token)
			if token.Err != nil {
				result.Err = token.Err
				var providerErr *ai.ProviderError
				if errors.As(token.Err, &providerErr) {
					result.HTTPStatus = providerErr.StatusCode
				}
			}
			if call := token.ToolCall(); call != nil {
				result.ToolCallCount++
			}
			if token.Completion != nil {
				result.ResponseModel = token.Completion.Model
				result.RequestID = token.Completion.RequestID
				result.FinishReason = token.Completion.FinishReason
				if token.Completion.UsageReported {
					usage := token.Completion.Usage
					result.Usage = &usage
				}
			}
			if ai.SendToken(ctx, out, token) {
				return true
			}
			if result.Err == nil {
				result.Err = ctx.Err()
			}
			return false
		}
		m.generateStream(ctx, req, emit)
		observation.Finish(result)
	}()
	return out
}

func (m *Model) generateStream(ctx context.Context, req ai.AIRequest, emit func(ai.Token) bool) {
	payload, err := m.chatRequest(req)
	if err != nil {
		emit(ai.Token{Err: err})
		return
	}
	response, err := m.provider.doChat(ctx, payload)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			emit(ai.Token{Err: ctxErr})
			return
		}
		emit(ai.Token{Err: ai.ClassifyProviderError(err, 0, "", "", nil)})
		return
	}
	defer response.Body.Close()

	requestID := response.Header.Get("X-Request-Id")
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		body, readErr := io.ReadAll(io.LimitReader(response.Body, maxErrorBodySize+1))
		if ctxErr := ctx.Err(); ctxErr != nil {
			emit(ai.Token{Err: ctxErr})
			return
		}
		if readErr != nil {
			body = nil
		}
		if len(body) > maxErrorBodySize {
			body = body[:maxErrorBodySize]
		}
		httpErr := newHTTPError(response.StatusCode, body)
		emit(ai.Token{Err: ai.ClassifyProviderError(httpErr, response.StatusCode, ollamaErrorCode(body), requestID, response.Header)})
		return
	}

	completion := ai.Completion{Provider: "ollama", RequestID: requestID}
	var pendingCalls []ai.ToolCall
	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, 64*1024), maxStreamRecordSize)
	for scanner.Scan() {
		record := append(json.RawMessage(nil), scanner.Bytes()...)
		if len(bytes.TrimSpace(record)) == 0 {
			continue
		}
		var chunk rawChatResponse
		if err := json.Unmarshal(record, &chunk); err != nil {
			emit(ai.Token{Err: errors.New("decode Ollama stream: malformed response record")})
			return
		}
		if chunk.Error != "" {
			emit(ai.Token{Err: ai.ClassifyProviderError(errors.New("ollama stream error"), 0, chunk.Error, requestID, response.Header)})
			return
		}
		if chunk.Model != "" {
			completion.Model = chunk.Model
		}
		if chunk.Message.Role != "" && chunk.Message.Role != "assistant" {
			emit(ai.Token{Err: fmt.Errorf("invalid Ollama stream role %q", chunk.Message.Role)})
			return
		}
		if chunk.Message.Thinking != "" {
			emit(ai.Token{Err: fmt.Errorf("%w: Ollama reasoning output", ai.ErrUnsupportedCapability)})
			return
		}
		if chunk.Message.Content != "" && !emit(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: chunk.Message.Content}}) {
			return
		}
		for _, wireCall := range chunk.Message.ToolCalls {
			name := strings.TrimSpace(wireCall.Function.Name)
			if name == "" || !jsonObject(wireCall.Function.Arguments) {
				emit(ai.Token{Err: fmt.Errorf("%w: malformed Ollama tool call", ai.ErrInvalidToolCall)})
				return
			}
			id := strings.TrimSpace(wireCall.ID)
			if id == "" {
				id = ai.GenerateToolCallID(name)
			}
			pendingCalls = append(pendingCalls, ai.ToolCall{
				ID:   id,
				Type: "function",
				Name: name,
				Args: append(json.RawMessage(nil), wireCall.Function.Arguments...),
			})
		}
		if !chunk.Done {
			continue
		}
		// Ollama stores content and tool calls in separate fields, so emit all
		// streamed text before completing calls for canonical replay.
		for index := range pendingCalls {
			call := pendingCalls[index].Clone()
			if !emit(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &call}}) {
				return
			}
		}
		completion.FinishReason = chunk.DoneReason
		if chunk.PromptEvalCount != nil || chunk.EvalCount != nil {
			completion.UsageReported = true
			if chunk.PromptEvalCount != nil {
				completion.Usage.InputTokens = *chunk.PromptEvalCount
			}
			if chunk.EvalCount != nil {
				completion.Usage.OutputTokens = *chunk.EvalCount
			}
		}
		completion.Raw = record
		snapshot := completion
		snapshot.Raw = append(json.RawMessage(nil), completion.Raw...)
		emit(ai.Token{Completion: &snapshot})
		return
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		emit(ai.Token{Err: ctxErr})
		return
	}
	if err := scanner.Err(); err != nil {
		emit(ai.Token{Err: ai.ClassifyProviderError(err, 0, "", requestID, response.Header)})
		return
	}
	emit(ai.Token{Err: io.ErrUnexpectedEOF})
}
