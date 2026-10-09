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
	maxStreamRecordSize = 8 << 20
	maxErrorBodySize    = 1 << 20
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

type chatResponse struct {
	Model           string      `json:"model"`
	Message         chatMessage `json:"message"`
	Done            bool        `json:"done"`
	Reason          string      `json:"done_reason"`
	Error           string      `json:"error"`
	PromptEvalCount *int        `json:"prompt_eval_count"`
	EvalCount       *int        `json:"eval_count"`
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
	body, err := json.Marshal(payload)
	if err != nil {
		emit(ai.Token{Err: fmt.Errorf("marshal Ollama request: %w", err)})
		return
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(m.provider.baseURL, "/")+"/api/chat", bytes.NewReader(body))
	if err != nil {
		emit(ai.Token{Err: fmt.Errorf("create Ollama request: %w", err)})
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/x-ndjson")
	if m.provider.bearerToken != "" {
		httpReq.Header.Set("Authorization", "Bearer "+m.provider.bearerToken)
	}
	client, err := m.provider.requestClient()
	if err != nil {
		emit(ai.Token{Err: fmt.Errorf("configure Ollama HTTP client: %w", err)})
		return
	}
	response, err := client.Do(httpReq)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			emit(ai.Token{Err: ctxErr})
			return
		}
		emit(ai.Token{Err: ai.ClassifyProviderError(err, 0, "", "", nil)})
		return
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		limited := io.LimitReader(response.Body, maxErrorBodySize+1)
		errorBody, readErr := io.ReadAll(limited)
		if readErr != nil {
			errorBody = nil
		}
		if len(errorBody) > maxErrorBodySize {
			errorBody = errorBody[:maxErrorBodySize]
		}
		httpErr := newHTTPError(response.StatusCode, errorBody)
		requestID := response.Header.Get("X-Request-Id")
		emit(ai.Token{Err: ai.ClassifyProviderError(httpErr, response.StatusCode, ollamaErrorCode(errorBody), requestID, response.Header)})
		return
	}

	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, 64<<10), maxStreamRecordSize)
	completion := ai.Completion{Provider: "ollama", RequestID: response.Header.Get("X-Request-Id")}
	var pendingCalls []ai.ToolCall
	for scanner.Scan() {
		record := bytes.TrimSpace(scanner.Bytes())
		if len(record) == 0 {
			continue
		}
		var chunk chatResponse
		if err := json.Unmarshal(record, &chunk); err != nil {
			emit(ai.Token{Err: fmt.Errorf("decode Ollama stream: %w", err)})
			return
		}
		if chunk.Error != "" {
			emit(ai.Token{Err: ai.ClassifyProviderError(errors.New("Ollama stream error: "+chunk.Error), 0, chunk.Error, "", response.Header)})
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
		if chunk.Message.Content != "" {
			if !emit(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: chunk.Message.Content}}) {
				return
			}
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
			pendingCalls = append(pendingCalls, ai.ToolCall{ID: id, Type: "function", Name: name, Args: append(json.RawMessage(nil), wireCall.Function.Arguments...)})
		}
		if !chunk.Done {
			continue
		}
		// Ollama's wire message stores content and tool calls in separate
		// fields, so it cannot represent their interleaving. Emit all streamed
		// text first and complete calls at the terminal record. The resulting
		// canonical assistant message can therefore be replayed losslessly in
		// the next tool-result request.
		for i := range pendingCalls {
			call := pendingCalls[i].Clone()
			if !emit(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &call}}) {
				return
			}
		}
		completion.FinishReason = chunk.Reason
		if chunk.PromptEvalCount != nil || chunk.EvalCount != nil {
			completion.UsageReported = true
			if chunk.PromptEvalCount != nil {
				completion.Usage.InputTokens = *chunk.PromptEvalCount
			}
			if chunk.EvalCount != nil {
				completion.Usage.OutputTokens = *chunk.EvalCount
			}
		}
		completion.Raw = append(json.RawMessage(nil), record...)
		snapshot := completion
		snapshot.Raw = append(json.RawMessage(nil), completion.Raw...)
		emit(ai.Token{Completion: &snapshot})
		return
	}
	if err := scanner.Err(); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			emit(ai.Token{Err: ctxErr})
			return
		}
		emit(ai.Token{Err: fmt.Errorf("read Ollama stream: %w", err)})
		return
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		emit(ai.Token{Err: ctxErr})
		return
	}
	emit(ai.Token{Err: io.ErrUnexpectedEOF})
}
