package mistral

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

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"go.opentelemetry.io/otel/attribute"
)

const mistralTracerName = "github.com/lace-ai/gai/ai/mistral"

type Model struct {
	name   string
	client *Provider
	debug  gai.ObservationSink
}

var _ ai.Model = (*Model)(nil)
var _ ai.ModelDescriber = (*Model)(nil)

func (m *Model) Name() string {
	return m.name
}

func (m *Model) NativeTools() bool { return true }

// TokenCounter uses a local estimate for automatic budgeting. Tokenizer is
// still available explicitly, but may perform network I/O.
func (m *Model) TokenCounter() ai.TokenCounter { return ai.TextTokenEstimator{} }

func (m *Model) Tokenizer() ai.Tokenizer {
	return &Tokenizer{
		modelName: m.name,
		client:    m.client,
		debug:     m.debug,
	}
}

func (m *Model) Descriptor() ai.ModelDescriptor {
	if facts, ok := m.client.catalog.Lookup(m.name); ok {
		return effectiveMistralDescriptor(m.name, facts)
	}
	return mistralAdapterDescriptor(m.name)
}

func mistralAdapterDescriptor(model string) ai.ModelDescriptor {
	return ai.ModelDescriptor{Model: model, NativeMessages: ai.FeatureSupportSupported, NativeTools: ai.FeatureSupportSupported,
		ToolChoiceModes: []ai.ToolChoiceMode{ai.ToolChoiceAuto, ai.ToolChoiceNone, ai.ToolChoiceRequired},
		Multimodal:      ai.FeatureSupportUnsupported,
		Usage:           ai.FeatureSupportSupported, FinishReason: ai.FeatureSupportSupported, StreamingUsage: ai.FeatureSupportSupported,
		ToolCalling: ai.FeatureSupportSupported,
		JSONOutput:  ai.FeatureSupportSupported, JSONSchemaOutput: ai.FeatureSupportSupported,
		Reasoning: ai.FeatureSupportUnsupported, ReasoningEffort: ai.FeatureSupportUnsupported,
		Tokenizer: ai.TokenizerDescriptor{Available: ai.FeatureSupportSupported, Fidelity: ai.TokenizerFidelityEstimated}}
}

func (m *Model) Close() error {
	return nil
}

type chatCompletionRequest struct {
	Model          string               `json:"model"`
	Messages       []chatMessageRequest `json:"messages"`
	MaxTokens      *int                 `json:"max_tokens,omitempty"`
	Stream         bool                 `json:"stream,omitempty"`
	Tools          []chatToolRequest    `json:"tools,omitempty"`
	ToolChoice     any                  `json:"tool_choice,omitempty"`
	ResponseFormat *chatResponseFormat  `json:"response_format,omitempty"`
	StreamOptions  *chatStreamOptions   `json:"stream_options,omitempty"`
}

type chatStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatMessageRequest struct {
	Role       string                `json:"role"`
	Content    string                `json:"content"`
	ToolCalls  []chatMessageToolCall `json:"tool_calls,omitempty"`
	ToolCallID string                `json:"tool_call_id,omitempty"`
}
type chatMessageToolCall struct {
	ID       string               `json:"id"`
	Type     string               `json:"type"`
	Function chatToolCallFunction `json:"function"`
}
type chatToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatToolRequest struct {
	Type     string                  `json:"type"`
	Function chatToolFunctionRequest `json:"function"`
}

type chatToolFunctionRequest struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type chatResponseFormat struct {
	Type       string                  `json:"type"`
	JSONSchema *chatResponseJSONSchema `json:"json_schema,omitempty"`
}

type chatResponseJSONSchema struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
}

type chatCompletionResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Content   string          `json:"content"`
			ToolCalls json.RawMessage `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

func buildChatCompletionRequest(req ai.AIRequest, modelName string, stream bool) (chatCompletionRequest, error) {
	req, err := req.Normalized()
	if err != nil {
		return chatCompletionRequest{}, err
	}
	if err := req.ResponseFormat.Validate(); err != nil {
		return chatCompletionRequest{}, err
	}
	payload := chatCompletionRequest{
		Model: modelName,
		Messages: []chatMessageRequest{
			{
				Role:    "user",
				Content: req.Prompt,
			},
		},
		Stream: stream,
	}
	if stream {
		payload.StreamOptions = &chatStreamOptions{IncludeUsage: true}
	}
	if len(req.Messages) > 0 {
		messages, err := mapNativeMessages(req.Messages)
		if err != nil {
			return chatCompletionRequest{}, err
		}
		payload.Messages = messages
	}
	if req.MaxTokens > 0 {
		payload.MaxTokens = &req.MaxTokens
	}
	if len(req.Tools) > 0 {
		tools, err := mapChatTools(req.Tools)
		if err != nil {
			return chatCompletionRequest{}, err
		}
		payload.Tools = tools
		toolChoice, err := mapChatToolChoice(req.ToolChoice)
		if err != nil {
			return chatCompletionRequest{}, err
		}
		payload.ToolChoice = toolChoice
	}
	responseFormat, err := mapChatResponseFormat(req.ResponseFormat)
	if err != nil {
		return chatCompletionRequest{}, err
	}
	payload.ResponseFormat = responseFormat
	return payload, nil
}
func mapNativeMessages(messages []ai.Message) ([]chatMessageRequest, error) {
	out := make([]chatMessageRequest, 0, len(messages))
	for _, m := range messages {
		if err := m.Validate(); err != nil {
			return nil, err
		}
		if err := rejectRequiredExtensions(m.Extensions); err != nil {
			return nil, err
		}
		message := chatMessageRequest{Role: string(m.Role)}
		for _, part := range m.Parts {
			if err := rejectRequiredExtensions(part.Extensions); err != nil {
				return nil, err
			}
			switch part.Kind {
			case ai.ContentText, ai.ContentJSON:
				if len(message.ToolCalls) > 0 {
					return nil, fmt.Errorf("%w: Mistral text after tool calls", ai.ErrUnsupportedCapability)
				}
				if part.Kind == ai.ContentText {
					message.Content += part.Text
				} else {
					message.Content += string(part.JSON)
				}
			case ai.ContentToolCall:
				c := part.ToolCall
				if err := rejectRequiredExtensions(c.Extensions); err != nil {
					return nil, err
				}
				if len(c.ThoughtSignature) > 0 {
					return nil, fmt.Errorf("%w: Mistral tool signature", ai.ErrUnsupportedCapability)
				}
				message.ToolCalls = append(message.ToolCalls, chatMessageToolCall{ID: c.ID, Type: "function", Function: chatToolCallFunction{Name: c.Name, Arguments: string(c.Args)}})
			case ai.ContentToolResult:
				content, err := canonicalResultText(part.ToolResult)
				if err != nil {
					return nil, err
				}
				out = append(out, chatMessageRequest{Role: "tool", Content: content, ToolCallID: part.ToolResult.ToolCallID})
			case ai.ContentExtension:
			default:
				return nil, fmt.Errorf("%w: Mistral content %q", ai.ErrUnsupportedCapability, part.Kind)
			}
		}
		if m.Role != ai.RoleTool {
			out = append(out, message)
		}
	}
	return out, nil
}

func mapChatTools(definitions []ai.ToolDefinition) ([]chatToolRequest, error) {
	tools := make([]chatToolRequest, 0, len(definitions))
	for _, definition := range definitions {
		if err := definition.Validate(); err != nil {
			return nil, err
		}
		tools = append(tools, chatToolRequest{
			Type: "function",
			Function: chatToolFunctionRequest{
				Name:        definition.Name,
				Description: definition.Description,
				Parameters:  append(json.RawMessage(nil), definition.Parameters...),
			},
		})
	}
	return tools, nil
}

func mapChatToolChoice(choice ai.ToolChoice) (any, error) {
	switch choice.Mode {
	case ai.ToolChoiceNone:
		return "none", nil
	case ai.ToolChoiceRequired:
		if len(choice.Names) == 1 {
			return map[string]any{
				"type": "function",
				"function": map[string]string{
					"name": choice.Names[0],
				},
			}, nil
		}
		return "required", nil
	case ai.ToolChoiceAuto, "":
		return "auto", nil
	default:
		return nil, fmt.Errorf("unsupported mistral tool choice mode %q", choice.Mode)
	}
}

func mapChatResponseFormat(format ai.ResponseFormat) (*chatResponseFormat, error) {
	if err := format.Validate(); err != nil {
		return nil, err
	}
	switch format.Type {
	case "", ai.ResponseFormatText:
		return nil, nil
	case ai.ResponseFormatJSONObject:
		return &chatResponseFormat{Type: "json_object"}, nil
	case ai.ResponseFormatJSONSchema:
		return &chatResponseFormat{
			Type: "json_schema",
			JSONSchema: &chatResponseJSONSchema{
				Name:   format.Name,
				Schema: append(json.RawMessage(nil), format.Schema...),
			},
		}, nil
	default:
		return nil, fmt.Errorf("%w: %s", ai.ErrInvalidResponseFormat, format.Type)
	}
}

type Tokenizer struct {
	modelName string
	client    *Provider
	debug     gai.ObservationSink
}

func (t *Tokenizer) ID() string {
	return "mistral." + t.modelName
}

func (t *Tokenizer) Tokenize(ctx context.Context, text string) (tokens []string, err error) {
	_, span := gai.StartOperationSpan(ctx, mistralTracerName, "ai.mistral", "ai.operation", "tokenizer.tokenize",
		attribute.String("ai.provider", "mistral"),
		attribute.String("ai.model", t.modelName),
		attribute.String("ai.tokenizer", t.ID()),
		attribute.Int("ai.input_length", len(text)),
	)
	err = ai.ErrTokenizerUnsupported
	defer func() { gai.EndSpan(span, err) }()
	return nil, err
}

func (t *Tokenizer) CountTokens(ctx context.Context, text string) (tokens int, err error) {
	ctx, span := gai.StartOperationSpan(ctx, mistralTracerName, "ai.mistral", "ai.operation", "tokenizer.count_tokens",
		attribute.String("ai.provider", "mistral"),
		attribute.String("ai.model", t.modelName),
		attribute.String("ai.tokenizer", t.ID()),
		attribute.Int("ai.input_length", len(text)),
	)
	defer func() {
		span.SetAttributes(attribute.Int("ai.input_tokens", tokens))
		gai.EndSpan(span, err)
	}()

	const maxTokensForTokenCount = 1
	payload := chatCompletionRequest{
		Model: t.modelName,
		Messages: []chatMessageRequest{
			{
				Role:    "user",
				Content: text,
			},
		},
		MaxTokens: intPtr(maxTokensForTokenCount),
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return 0, err
	}

	httpReq, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		t.client.baseURL+"/v1/chat/completions",
		bytes.NewReader(body),
	)
	if err != nil {
		return 0, err
	}
	httpReq.Header.Set("Authorization", "Bearer "+t.client.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := t.client.httpClient.Do(httpReq)
	if err != nil {
		if gai.ObservationEnabled(ctx, t.debug) {
			gai.EmitObservation(ctx, t.debug, gai.Observation{
				Name:   "mistral_token_count_request_failed",
				Source: "ai:mistral.Tokenizer.CountTokens",
				Fields: map[string]any{
					"error": err.Error(),
				},
				Err: err,
			})
		}
		return 0, err
	}
	defer res.Body.Close()

	const maxResponseBody = 1 << 20 // 1MB
	resBody, err := io.ReadAll(io.LimitReader(res.Body, maxResponseBody))
	if err != nil {
		return 0, err
	}

	if res.StatusCode >= http.StatusMultipleChoices {
		span.SetAttributes(attribute.Int("http.response.status_code", res.StatusCode))
		if gai.ObservationEnabled(ctx, t.debug) {
			fields := map[string]any{
				"status_code": res.StatusCode,
			}

			gai.EmitObservation(ctx, t.debug, gai.Observation{
				Name:   "mistral_token_count_request_failed",
				Source: "ai:mistral.Tokenizer.CountTokens",
				Fields: fields,
			})
		}
		return 0, newHTTPError("token count", res.StatusCode, resBody)
	}
	span.SetAttributes(attribute.Int("http.response.status_code", res.StatusCode))

	var parsed chatCompletionResponse
	if err := json.Unmarshal(resBody, &parsed); err != nil {
		return 0, err
	}
	if parsed.Usage == nil {
		return 0, nil
	}
	return parsed.Usage.PromptTokens, nil
}

type chatCompletionStreamResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Delta struct {
			Content   json.RawMessage `json:"content"`
			ToolCalls json.RawMessage `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

type mistralStreamFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type mistralStreamToolCallEntry struct {
	Index    int                       `json:"index"`
	ID       string                    `json:"id"`
	Type     string                    `json:"type"`
	Function mistralStreamFunctionCall `json:"function"`
}

type mistralToolCallAccumulator struct {
	entries map[int]*mistralToolCallState
	order   []int
}

type mistralToolCallState struct {
	id        strings.Builder
	callType  string
	name      strings.Builder
	arguments strings.Builder
	emitted   bool
}

func (a *mistralToolCallAccumulator) add(raw json.RawMessage, final bool) ([]ai.ToolCall, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return a.ready(final), nil
	}

	var calls []mistralStreamToolCallEntry
	if err := json.Unmarshal(raw, &calls); err != nil {
		return nil, fmt.Errorf("decode tool_calls: %w", err)
	}

	for _, c := range calls {
		state := a.state(c.Index)
		if strings.TrimSpace(c.ID) != "" {
			state.id.WriteString(strings.TrimSpace(c.ID))
		}
		if strings.TrimSpace(c.Type) != "" {
			state.callType = strings.TrimSpace(c.Type)
		}
		if strings.TrimSpace(c.Function.Name) != "" {
			state.name.WriteString(strings.TrimSpace(c.Function.Name))
		}
		if c.Function.Arguments != "" {
			state.arguments.WriteString(c.Function.Arguments)
		}
	}

	return a.ready(final), nil
}

func (a *mistralToolCallAccumulator) state(index int) *mistralToolCallState {
	if a.entries == nil {
		a.entries = make(map[int]*mistralToolCallState)
	}
	state := a.entries[index]
	if state == nil {
		state = &mistralToolCallState{}
		a.entries[index] = state
		a.order = append(a.order, index)
	}
	return state
}

func (a *mistralToolCallAccumulator) ready(final bool) []ai.ToolCall {
	var result []ai.ToolCall
	for _, index := range a.order {
		state := a.entries[index]
		if state == nil || state.emitted {
			continue
		}
		toolName := strings.TrimSpace(state.name.String())
		if toolName == "" {
			continue
		}

		args := json.RawMessage(state.arguments.String())
		if len(args) == 0 {
			if !final {
				continue
			}
			args = json.RawMessage("{}")
		}
		if !json.Valid(args) {
			continue
		}

		callType := state.callType
		if callType == "" {
			callType = "function"
		}
		toolCallID := strings.TrimSpace(state.id.String())
		if toolCallID == "" {
			toolCallID = ai.GenerateToolCallID(toolName)
		}
		result = append(result, ai.ToolCall{
			ID:   toolCallID,
			Type: callType,
			Name: toolName,
			Args: args,
		})
		state.emitted = true
	}
	return result
}

func (m *Model) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	raw := make(chan ai.Token, 1)

	go func() {
		ctx := ctx
		var streamErr error
		var observation *ai.GenerationObservation
		generationResult := ai.GenerationResult{}
		defer close(raw)
		defer func() {
			generationResult.Err = streamErr
			observation.Finish(generationResult)
		}()
		emit := func(token ai.Token) bool {
			observation.ObserveToken(token)
			if ai.SendToken(ctx, raw, token) {
				return true
			}
			if streamErr == nil {
				streamErr = ctx.Err()
			}
			return false
		}
		normalized, normalizeErr := req.Normalized()
		if normalizeErr != nil {
			streamErr = normalizeErr
			emit(ai.Token{Type: ai.TokenTypeErr, Err: normalizeErr, Text: normalizeErr.Error()})
			return
		}
		req = normalized
		if err := ai.ValidateModelRequest(m, req); err != nil {
			streamErr = err
			emit(ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		}

		payload, err := buildChatCompletionRequest(req, m.name, true)
		if err != nil {
			streamErr = err
			if !emit(ai.Token{Err: err, Type: ai.TokenTypeErr}) {
				return
			}
			return
		}

		body, err := json.Marshal(payload)
		if err != nil {
			streamErr = err
			if gai.ObservationEnabled(ctx, m.debug) {
				fields := map[string]any{
					"error": err.Error(),
				}

				gai.EmitObservation(ctx, m.debug, gai.Observation{
					Name:   "mistral_stream_request_payload",
					Source: "ai:mistral.Model.GenerateStream",
					Fields: fields,
					Err:    err,
				})
			}
			if !emit(ai.Token{Err: err, Type: ai.TokenTypeErr}) {
				return
			}
			return
		}

		httpReq, err := http.NewRequestWithContext(
			ctx,
			http.MethodPost,
			m.client.baseURL+"/v1/chat/completions",
			bytes.NewReader(body),
		)
		if err != nil {
			streamErr = err
			if gai.ObservationEnabled(ctx, m.debug) {
				gai.EmitObservation(ctx, m.debug, gai.Observation{
					Name:   "mistral_stream_request_creation_failed",
					Source: "ai:mistral.Model.GenerateStream",
					Fields: map[string]any{
						"error": err.Error(),
					},
					Err: err,
				})
			}
			if !emit(ai.Token{Err: err, Type: ai.TokenTypeErr}) {
				return
			}
			return
		}
		httpReq.Header.Set("Authorization", "Bearer "+m.client.apiKey)
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")
		generationCtx, startedObservation := ai.StartGenerationObservation(ctx, req, ai.GenerationConfig{Provider: "mistral", Model: m.name, Streaming: true, Sink: m.debug})
		observation = startedObservation
		ctx = generationCtx
		if gai.ObservationEnabled(ctx, m.debug) {
			fields := map[string]any{"max_tokens": req.MaxTokens}
			gai.AddObservationContent(ctx, m.debug, fields, "prompt", gai.ContentKindPrompt, observationPrompt(req))
			gai.EmitObservation(ctx, m.debug, gai.Observation{
				Name:   "mistral_stream_request",
				Source: "ai:mistral.Model.GenerateStream",
				Fields: fields,
			})
		}
		httpReq = httpReq.WithContext(generationCtx)

		res, err := m.client.httpClient.Do(httpReq)
		if err != nil {
			streamErr = err
			if gai.ObservationEnabled(ctx, m.debug) {
				gai.EmitObservation(ctx, m.debug, gai.Observation{
					Name:   "mistral_stream_request_failed",
					Source: "ai:mistral.Model.GenerateStream",
					Fields: map[string]any{
						"error": err.Error(),
					},
					Err: err,
				})
			}
			if !emit(ai.Token{Err: err, Type: ai.TokenTypeErr}) {
				return
			}
			return
		}
		defer res.Body.Close()
		generationResult.HTTPStatus = res.StatusCode

		if res.StatusCode >= http.StatusMultipleChoices {
			const maxResponseBody = 1 << 20 // 1MB
			resBody, readErr := io.ReadAll(io.LimitReader(res.Body, maxResponseBody))
			if readErr != nil {
				if gai.ObservationEnabled(ctx, m.debug) {
					gai.EmitObservation(ctx, m.debug, gai.Observation{
						Name:   "mistral_stream_request_failed_with_unreadable_body",
						Source: "ai:mistral.Model.GenerateStream",
						Fields: map[string]any{
							"status_code": res.StatusCode,
							"error":       readErr.Error(),
						},
						Err: readErr,
					})
				}
				streamErr = fmt.Errorf("mistral chat stream failed (status %d): %w", res.StatusCode, readErr)
				if !emit(ai.Token{
					Err:  streamErr,
					Type: ai.TokenTypeErr,
				}) {
					return
				}
				return
			}
			if gai.ObservationEnabled(ctx, m.debug) {
				fields := map[string]any{
					"status_code": res.StatusCode,
				}

				gai.EmitObservation(ctx, m.debug, gai.Observation{
					Name:   "mistral_stream_request_failed",
					Source: "ai:mistral.Model.GenerateStream",
					Fields: fields,
				})
			}
			streamErr = ai.ClassifyProviderError(newHTTPError("chat stream", res.StatusCode, resBody), res.StatusCode, mistralErrorCode(resBody), res.Header.Get("x-request-id"), res.Header)
			if !emit(ai.Token{
				Err:  streamErr,
				Type: ai.TokenTypeErr,
			}) {
				return
			}
			return
		}

		reader := bufio.NewReader(res.Body)
		var eventData strings.Builder
		var toolCallAccumulator mistralToolCallAccumulator
		completion := ai.Completion{Provider: "mistral"}
		hasCompletion := false
		emitCompletion := func() error {
			if !hasCompletion {
				return nil
			}
			snapshot := completion
			snapshot.Raw = append(json.RawMessage(nil), completion.Raw...)
			if !emit(ai.Token{Type: ai.TokenTypeCompletion, Completion: &snapshot}) {
				return ctx.Err()
			}
			hasCompletion = false
			return nil
		}
		defer func() {
			previousErr := streamErr
			if emitErr := emitCompletion(); emitErr != nil {
				if previousErr != nil {
					streamErr = previousErr
				} else {
					streamErr = emitErr
				}
			}
		}()

		flushEvent := func() error {
			event := strings.TrimSpace(eventData.String())
			eventData.Reset()
			if event == "" {
				return nil
			}
			if event == "[DONE]" {
				for _, tc := range toolCallAccumulator.ready(true) {
					tcCopy := tc
					if !emit(ai.Token{
						Type:     ai.TokenTypeToolCall,
						ToolCall: &tcCopy,
					}) {
						return ctx.Err()
					}
				}
				if err := emitCompletion(); err != nil {
					return err
				}
				return io.EOF
			}

			var chunk chatCompletionStreamResponse
			if err := json.Unmarshal([]byte(event), &chunk); err != nil {
				return fmt.Errorf("decode stream chunk: %w", err)
			}
			if chunk.ID != "" {
				completion.RequestID = chunk.ID
				hasCompletion = true
			}
			if chunk.Model != "" {
				completion.Model = chunk.Model
				hasCompletion = true
			}
			if chunk.Usage != nil {
				completion.UsageReported = true
				completion.Usage = ai.Usage{InputTokens: chunk.Usage.PromptTokens, OutputTokens: chunk.Usage.CompletionTokens}
				hasCompletion = true
			}
			if len(chunk.Choices) == 0 {
				if hasCompletion {
					completion.Raw = append(completion.Raw[:0], []byte(event)...)
				}
				return nil
			}
			if chunk.Choices[0].FinishReason != "" {
				completion.FinishReason = chunk.Choices[0].FinishReason
				hasCompletion = true
			}
			if hasCompletion {
				completion.Raw = append(completion.Raw[:0], []byte(event)...)
			}

			text, err := extractStreamText(chunk.Choices[0].Delta.Content)
			if err != nil {
				return err
			}
			if text != "" {
				if !emit(ai.Token{Type: ai.TokenTypeText, Text: text, Data: []byte(text)}) {
					return ctx.Err()
				}
			}

			finalToolCalls := chunk.Choices[0].FinishReason == "tool_calls"
			toolCalls := strings.TrimSpace(string(chunk.Choices[0].Delta.ToolCalls))
			if (toolCalls != "" && toolCalls != "null") || finalToolCalls {
				calls, mapErr := toolCallAccumulator.add(chunk.Choices[0].Delta.ToolCalls, finalToolCalls)
				if mapErr != nil {
					if gai.ObservationEnabled(ctx, m.debug) {
						fields := map[string]any{
							"error": mapErr.Error(),
						}
						gai.AddObservationContent(ctx, m.debug, fields, "tool_calls", gai.ContentKindToolInput, chunk.Choices[0].Delta.ToolCalls)
						gai.EmitObservation(ctx, m.debug, gai.Observation{
							Name:   "mistral_stream_tool_calls_mapping_failed",
							Source: "ai:mistral.Model.GenerateStream",
							Fields: fields,
							Err:    mapErr,
						})
					}
					streamErr = fmt.Errorf("map tool_calls: %w", mapErr)
					return streamErr
				}
				if gai.ObservationEnabled(ctx, m.debug) {
					fields := map[string]any{}
					gai.AddObservationContent(ctx, m.debug, fields, "tool_calls", gai.ContentKindToolInput, calls)
					gai.EmitObservation(ctx, m.debug, gai.Observation{
						Name:   "mistral_stream_tool_calls_mapped",
						Source: "ai:mistral.Model.GenerateStream",
						Fields: fields,
					})
				}
				for _, tc := range calls {
					tcCopy := tc
					if !emit(ai.Token{
						Type:     ai.TokenTypeToolCall,
						Data:     append([]byte(nil), tc.Args...),
						ToolCall: &tcCopy,
					}) {
						return ctx.Err()
					}
				}
			}

			return nil
		}

		for {
			line, err := reader.ReadString('\n')
			if err != nil && !errors.Is(err, io.EOF) {
				if gai.ObservationEnabled(ctx, m.debug) {
					gai.EmitObservation(ctx, m.debug, gai.Observation{
						Name:   "mistral_stream_read_failed",
						Source: "ai:mistral.Model.GenerateStream",
						Fields: map[string]any{
							"error": err.Error(),
						},
						Err: err,
					})
				}
				streamErr = fmt.Errorf("read stream: %w", err)
				if !emit(ai.Token{Err: streamErr, Type: ai.TokenTypeErr}) {
					return
				}
				return
			}

			line = strings.TrimRight(line, "\r\n")
			if line == "" {
				if flushErr := flushEvent(); flushErr != nil {
					if errors.Is(flushErr, io.EOF) {
						if gai.ObservationEnabled(ctx, m.debug) {
							gai.EmitObservation(ctx, m.debug, gai.Observation{
								Name:   "mistral_stream_read_eof",
								Source: "ai:mistral.Model.GenerateStream",
								Fields: map[string]any{},
							})
						}
						return
					}
					if gai.ObservationEnabled(ctx, m.debug) {
						gai.EmitObservation(ctx, m.debug, gai.Observation{
							Name:   "mistral_stream_chunk_processing_failed",
							Source: "ai:mistral.Model.GenerateStream",
							Fields: map[string]any{
								"error": flushErr.Error(),
							},
							Err: flushErr,
						})
					}
					streamErr = flushErr
					if !emit(ai.Token{Err: flushErr, Type: ai.TokenTypeErr}) {
						return
					}
					return
				}
			} else if strings.HasPrefix(line, "data:") {
				if eventData.Len() > 0 {
					eventData.WriteByte('\n')
				}
				eventData.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			}

			if errors.Is(err, io.EOF) {
				if gai.ObservationEnabled(ctx, m.debug) {
					gai.EmitObservation(ctx, m.debug, gai.Observation{
						Name:   "mistral_stream_read_eof",
						Source: "ai:mistral.Model.GenerateStream",
						Fields: map[string]any{},
					})
				}
				if flushErr := flushEvent(); flushErr != nil && !errors.Is(flushErr, io.EOF) {
					if gai.ObservationEnabled(ctx, m.debug) {
						gai.EmitObservation(ctx, m.debug, gai.Observation{
							Name:   "mistral_stream_final_flush_failed",
							Source: "ai:mistral.Model.GenerateStream",
							Fields: map[string]any{
								"error": flushErr.Error(),
							},
							Err: flushErr,
						})
					}
					streamErr = flushErr
					if !emit(ai.Token{Err: flushErr, Type: ai.TokenTypeErr}) {
						return
					}
				}
				return
			}
		}
	}()

	return ai.DetectToolCallsInStream(ctx, raw, m.debug)
}

func extractStreamText(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil
	}

	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var builder strings.Builder
		for _, p := range parts {
			if p.Text == "" {
				continue
			}
			if p.Type == "" || p.Type == "text" {
				builder.WriteString(p.Text)
			}
		}
		return builder.String(), nil
	}

	return "", fmt.Errorf("unsupported stream content payload: %s", string(raw))
}

func intPtr(v int) *int {
	return &v
}

func (m *Model) Generate(ctx context.Context, req ai.AIRequest) (response *ai.AIResponse, err error) {
	normalized, normalizeErr := req.Normalized()
	if normalizeErr != nil {
		return nil, normalizeErr
	}
	req = normalized
	if err := ai.ValidateModelRequest(m, req); err != nil {
		return nil, err
	}
	payload, err := buildChatCompletionRequest(req, m.name, false)
	if err != nil {
		return nil, err
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		m.client.baseURL+"/v1/chat/completions",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Authorization", "Bearer "+m.client.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	ctx, observation := ai.StartGenerationObservation(ctx, req, ai.GenerationConfig{Provider: "mistral", Model: m.name, Sink: m.debug})
	if gai.ObservationEnabled(ctx, m.debug) {
		fields := map[string]any{"max_tokens": req.MaxTokens}
		gai.AddObservationContent(ctx, m.debug, fields, "prompt", gai.ContentKindPrompt, observationPrompt(req))
		gai.EmitObservation(ctx, m.debug, gai.Observation{
			Name:   "mistral_generate_request",
			Source: "ai:mistral.Model.Generate",
			Fields: fields,
		})
	}
	httpReq = httpReq.WithContext(ctx)
	generationResult := ai.GenerationResult{}
	defer func() {
		generationResult.Err = err
		observation.Finish(generationResult)
	}()

	res, err := m.client.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	generationResult.HTTPStatus = res.StatusCode

	const maxResponseBody = 1 << 20 // 10MB
	resBody, err := io.ReadAll(io.LimitReader(res.Body, maxResponseBody))
	if err != nil {
		return nil, err
	}

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, newHTTPError("chat completion", res.StatusCode, resBody)
	}

	var parsed chatCompletionResponse
	if err := json.Unmarshal(resBody, &parsed); err != nil {
		return nil, err
	}
	if len(parsed.Choices) == 0 {
		return nil, ErrNoChoices
	}
	toolCalls, err := mapChatResponseToolCalls(parsed.Choices[0].Message.ToolCalls)
	if err != nil {
		return nil, err
	}
	usage := ai.Usage{}
	if parsed.Usage != nil {
		usage = ai.Usage{InputTokens: parsed.Usage.PromptTokens, OutputTokens: parsed.Usage.CompletionTokens}
		generationResult.Usage = &usage
	}
	generationResult.ResponseModel = parsed.Model
	generationResult.RequestID = parsed.ID
	generationResult.FinishReason = parsed.Choices[0].FinishReason
	generationResult.ToolCallCount = len(toolCalls)
	if gai.ObservationEnabled(ctx, m.debug) {
		fields := map[string]any{
			"input_tokens":  usage.InputTokens,
			"output_tokens": usage.OutputTokens,
		}

		gai.AddObservationContent(ctx, m.debug, fields, "response_text", gai.ContentKindCompletion, parsed.Choices[0].Message.Content)
		gai.EmitObservation(ctx, m.debug, gai.Observation{
			Name:   "mistral_generate_response",
			Source: "ai:mistral.Model.Generate",
			Fields: fields,
		})
	}

	result := &ai.AIResponse{
		Text:         parsed.Choices[0].Message.Content,
		ToolCalls:    toolCalls,
		Raw:          append([]byte(nil), resBody...),
		FinishReason: parsed.Choices[0].FinishReason,
		InputTokens:  usage.InputTokens,
		OutputTokens: usage.OutputTokens,
	}
	semantic := ai.TextMessage(ai.RoleAssistant, result.Text)
	for _, call := range toolCalls {
		c := call
		semantic.Parts = append(semantic.Parts, ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &c})
	}
	result.SetMessage(semantic)
	return result, nil
}

func mapChatResponseToolCalls(raw json.RawMessage) ([]ai.ToolCall, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}

	var entries []mistralStreamToolCallEntry
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, fmt.Errorf("decode tool_calls: %w", err)
	}

	result := make([]ai.ToolCall, 0, len(entries))
	for i, entry := range entries {
		toolName := strings.TrimSpace(entry.Function.Name)
		if toolName == "" {
			return nil, fmt.Errorf("map tool_calls[%d]: missing tool name", i)
		}
		args := json.RawMessage(strings.TrimSpace(entry.Function.Arguments))
		if len(args) == 0 {
			args = json.RawMessage("{}")
		}
		if !json.Valid(args) {
			return nil, fmt.Errorf("map tool_calls[%d]: invalid JSON arguments for tool %q", i, toolName)
		}
		callType := strings.TrimSpace(entry.Type)
		if callType == "" {
			callType = "function"
		}
		callID := strings.TrimSpace(entry.ID)
		if callID == "" {
			callID = ai.GenerateToolCallID(toolName)
		}
		result = append(result, ai.ToolCall{
			ID:   callID,
			Type: callType,
			Name: toolName,
			Args: args,
		})
	}
	return result, nil
}

// observationPrompt respects the prompt capture category without mixing in
// tool inputs/results, reasoning, or opaque provider state.
func observationPrompt(req ai.AIRequest) string {
	var parts []string
	for _, message := range req.Messages {
		if message.Role == ai.RoleSystem || message.Role == ai.RoleUser {
			parts = append(parts, message.Text())
		}
	}
	return strings.Join(parts, "\n")
}
