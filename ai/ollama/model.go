package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/lace-ai/gai/ai"
	ollamaapi "github.com/ollama/ollama/api"
)

const maxErrorBodySize = 1 << 20

var (
	errSDKStreamDone      = errors.New("ollama SDK stream complete")
	errSDKCallbackHandled = errors.New("ollama SDK callback error handled")
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
	Message struct {
		Role      string `json:"role"`
		Thinking  string `json:"thinking"`
		ToolCalls []struct {
			ID       string `json:"id"`
			Function struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	} `json:"message"`
	PromptEvalCount *int `json:"prompt_eval_count"`
	EvalCount       *int `json:"eval_count"`
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
	schemas := make([]json.RawMessage, len(req.Tools))
	for index := range req.Tools {
		schemas[index] = req.Tools[index].Parameters
	}
	client, capture, err := m.provider.sdkClient(schemas)
	if err != nil {
		emit(ai.Token{Err: fmt.Errorf("configure Ollama SDK client: %w", err)})
		return
	}

	completion := ai.Completion{Provider: "ollama"}
	var pendingCalls []ai.ToolCall
	err = client.Chat(ctx, payload, func(chunk ollamaapi.ChatResponse) error {
		record := capture.popRecord()
		if len(record) == 0 {
			emit(ai.Token{Err: fmt.Errorf("decode Ollama SDK stream: response record unavailable")})
			return errSDKCallbackHandled
		}
		var raw rawChatResponse
		if err := json.Unmarshal(record, &raw); err != nil {
			emit(ai.Token{Err: fmt.Errorf("decode Ollama SDK stream metadata: %w", err)})
			return errSDKCallbackHandled
		}
		_, headers, _ := capture.metadata()
		completion.RequestID = headers.Get("X-Request-Id")
		if chunk.Model != "" {
			completion.Model = chunk.Model
		}
		if raw.Message.Role != "" && raw.Message.Role != "assistant" {
			emit(ai.Token{Err: fmt.Errorf("invalid Ollama stream role %q", raw.Message.Role)})
			return errSDKCallbackHandled
		}
		if raw.Message.Thinking != "" {
			emit(ai.Token{Err: fmt.Errorf("%w: Ollama reasoning output", ai.ErrUnsupportedCapability)})
			return errSDKCallbackHandled
		}
		if chunk.Message.Content != "" && !emit(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: chunk.Message.Content}}) {
			return errSDKCallbackHandled
		}
		if len(raw.Message.ToolCalls) != len(chunk.Message.ToolCalls) {
			emit(ai.Token{Err: fmt.Errorf("%w: malformed Ollama tool calls", ai.ErrInvalidToolCall)})
			return errSDKCallbackHandled
		}
		for index, sdkCall := range chunk.Message.ToolCalls {
			rawCall := raw.Message.ToolCalls[index]
			name := strings.TrimSpace(sdkCall.Function.Name)
			if name == "" || !jsonObject(rawCall.Function.Arguments) {
				emit(ai.Token{Err: fmt.Errorf("%w: malformed Ollama tool call", ai.ErrInvalidToolCall)})
				return errSDKCallbackHandled
			}
			id := strings.TrimSpace(sdkCall.ID)
			if id == "" {
				id = ai.GenerateToolCallID(name)
			}
			pendingCalls = append(pendingCalls, ai.ToolCall{
				ID:   id,
				Type: "function",
				Name: name,
				Args: append(json.RawMessage(nil), rawCall.Function.Arguments...),
			})
		}
		if !chunk.Done {
			return nil
		}
		// Ollama stores content and tool calls in separate fields, so emit all
		// streamed text before completing calls for canonical replay.
		for index := range pendingCalls {
			call := pendingCalls[index].Clone()
			if !emit(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &call}}) {
				return errSDKCallbackHandled
			}
		}
		completion.FinishReason = chunk.DoneReason
		if raw.PromptEvalCount != nil || raw.EvalCount != nil {
			completion.UsageReported = true
			if raw.PromptEvalCount != nil {
				completion.Usage.InputTokens = *raw.PromptEvalCount
			}
			if raw.EvalCount != nil {
				completion.Usage.OutputTokens = *raw.EvalCount
			}
		}
		completion.Raw = append(json.RawMessage(nil), record...)
		snapshot := completion
		snapshot.Raw = append(json.RawMessage(nil), completion.Raw...)
		if !emit(ai.Token{Completion: &snapshot}) {
			return errSDKCallbackHandled
		}
		return errSDKStreamDone
	})

	if errors.Is(err, errSDKStreamDone) || errors.Is(err, errSDKCallbackHandled) {
		return
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		emit(ai.Token{Err: ctxErr})
		return
	}
	status, headers, errorBody := capture.metadata()
	if status == 0 {
		if err != nil {
			emit(ai.Token{Err: ai.ClassifyProviderError(err, 0, "", "", nil)})
			return
		}
		emit(ai.Token{Err: io.ErrUnexpectedEOF})
		return
	}
	if status < http.StatusOK || status >= http.StatusMultipleChoices {
		httpErr := newHTTPError(status, errorBody)
		emit(ai.Token{Err: ai.ClassifyProviderError(httpErr, status, ollamaErrorCode(errorBody), headers.Get("X-Request-Id"), headers)})
		return
	}
	if err != nil {
		record := capture.unconsumedRecord()
		if len(record) == 0 {
			emit(ai.Token{Err: ai.ClassifyProviderError(err, 0, "", headers.Get("X-Request-Id"), headers)})
			return
		}
		var inBand struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(record, &inBand) == nil && inBand.Error != "" {
			emit(ai.Token{Err: ai.ClassifyProviderError(errors.New("ollama stream error"), 0, inBand.Error, headers.Get("X-Request-Id"), headers)})
			return
		}
		// The SDK includes malformed response records verbatim in its error.
		// Do not expose model output through errors or observation telemetry.
		emit(ai.Token{Err: errors.New("decode Ollama SDK stream: malformed response record")})
		return
	}
	emit(ai.Token{Err: io.ErrUnexpectedEOF})
}
