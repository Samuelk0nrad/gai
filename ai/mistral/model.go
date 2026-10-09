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
)

type Model struct {
	name        string
	client      *Provider
	debug       gai.ObservationSink
	chatOptions ChatCompletionOptions
}

var _ ai.Model = (*Model)(nil)
var _ ai.ModelDescriber = (*Model)(nil)

func (m *Model) Name() string {
	return m.name
}

// TokenCounter supplies the generic local estimator without provider I/O.
func (m *Model) TokenCounter() ai.TokenCounter { return ai.TextTokenEstimator{} }

func (m *Model) Descriptor() ai.ModelDescriptor {
	if facts, ok := m.client.catalog.Lookup(m.name); ok {
		return effectiveMistralDescriptor(m.name, facts)
	}
	return mistralAdapterDescriptor(m.name)
}

// mistralImplementationDescriptor reports what this adapter can map for any
// model. It is an upper bound: per-model facts and the provider catalog can
// only narrow it, so a catalog flag never enables unimplemented behavior.
func mistralImplementationDescriptor(model string) ai.ModelDescriptor {
	return ai.ModelDescriptor{Model: model, NativeMessages: ai.FeatureSupportSupported, NativeTools: ai.FeatureSupportSupported,
		ToolChoiceModes: []ai.ToolChoiceMode{ai.ToolChoiceAuto, ai.ToolChoiceNone, ai.ToolChoiceRequired},
		Usage:           ai.FeatureSupportSupported, FinishReason: ai.FeatureSupportSupported, StreamingUsage: ai.FeatureSupportSupported,
		JSONOutput: ai.FeatureSupportSupported, JSONSchemaOutput: ai.FeatureSupportSupported,
		Reasoning: ai.FeatureSupportSupported, ReasoningEffort: ai.FeatureSupportSupported,
	}
}

// adjustableReasoningEfforts are the reasoning_effort values Mistral documents
// for adjustable-reasoning models. The API enum is wider, but other values are
// not documented per model, so they are not advertised.
var adjustableReasoningEfforts = []ai.ReasoningEffort{ai.ReasoningEffortNone, ai.ReasoningEffortHigh}

// mistralAdapterDescriptor reports the static facts GAI knows for model.
// Reasoning is Unknown for models without documented adjustable reasoning:
// requests pass local preflight and the provider decides.
func mistralAdapterDescriptor(model string) ai.ModelDescriptor {
	d := mistralImplementationDescriptor(model)
	if isAdjustableReasoningModel(model) {
		d.ReasoningEfforts = append([]ai.ReasoningEffort(nil), adjustableReasoningEfforts...)
		return d
	}
	d.Reasoning, d.ReasoningEffort = ai.FeatureSupportUnknown, ai.FeatureSupportUnknown
	return d
}

func isAdjustableReasoningModel(model string) bool {
	switch strings.TrimSpace(model) {
	case MistralSmallLatest, MistralMedium35:
		return true
	}
	return false
}

// mapReasoning converts the portable reasoning configuration into
// reasoning_effort. Omitted configuration stays omitted, explicit "none" is
// sent, and enabling reasoning without an effort selects "high", the only
// documented setting that returns thinking. Mistral always returns thinking
// when reasoning is on, so IncludeThoughts cannot hide it.
func mapReasoning(config ai.ReasoningConfig) (*string, error) {
	if config.BudgetTokens > 0 {
		return nil, fmt.Errorf("%w: Mistral has no reasoning token budget; set Reasoning.Effort instead", ai.ErrUnsupportedCapability)
	}
	enabled := config.Enabled || config.IncludeThoughts
	var effort string
	switch config.Effort {
	case "":
		if !enabled {
			return nil, nil
		}
		effort = string(ai.ReasoningEffortHigh)
	case ai.ReasoningEffortNone:
		if enabled {
			return nil, fmt.Errorf("%w: Mistral reasoning effort %q cannot be combined with enabled reasoning or included thoughts", ai.ErrUnsupportedCapability, config.Effort)
		}
		effort = string(config.Effort)
	case ai.ReasoningEffortMinimal, ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh, ai.ReasoningEffortXHigh:
		effort = string(config.Effort)
	default:
		return nil, fmt.Errorf("%w: Mistral reasoning effort %q", ai.ErrUnsupportedCapability, config.Effort)
	}
	return &effort, nil
}

// preflight applies descriptor validation plus Mistral input checks that the
// shared descriptor cannot express. It never performs discovery.
func (m *Model) preflight(req ai.AIRequest) error {
	if err := ai.ValidateModelRequest(m, req); err != nil {
		return err
	}
	if containsImageInput(req.Messages) {
		if vision, known := m.client.visionSupport(m.name); known && !vision {
			return &ai.UnsupportedCapabilityError{Model: m.name, Capability: "image input"}
		}
	}
	return nil
}

func (m *Model) Close() error {
	return nil
}

type chatCompletionRequest struct {
	ChatCompletionOptions
	Model          string               `json:"model"`
	Messages       []chatMessageRequest `json:"messages"`
	MaxTokens      *int                 `json:"max_tokens,omitempty"`
	Stream         bool                 `json:"stream,omitempty"`
	Tools          []chatToolRequest    `json:"tools,omitempty"`
	ToolChoice     any                  `json:"tool_choice,omitempty"`
	ResponseFormat *chatResponseFormat  `json:"response_format,omitempty"`
	StreamOptions  *chatStreamOptions   `json:"stream_options,omitempty"`
	// ReasoningEffort is owned by the portable request's Reasoning field.
	ReasoningEffort *string `json:"reasoning_effort,omitempty"`
}

type chatStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatMessageRequest struct {
	Role       string                `json:"role"`
	Content    chatContent           `json:"content"`
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
			// Content is a string or an ordered chunk array.
			Content   json.RawMessage `json:"content"`
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
	req = req.Copy()
	err := req.Validate()
	if err != nil {
		return chatCompletionRequest{}, err
	}
	if err := req.ResponseFormat.Validate(); err != nil {
		return chatCompletionRequest{}, err
	}
	payload := chatCompletionRequest{
		Model:  modelName,
		Stream: stream,
	}
	if stream {
		payload.StreamOptions = &chatStreamOptions{IncludeUsage: true}
	}
	messages, err := mapNativeMessages(req.Messages)
	if err != nil {
		return chatCompletionRequest{}, err
	}
	payload.Messages = messages
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
	effort, err := mapReasoning(req.Reasoning)
	if err != nil {
		return chatCompletionRequest{}, err
	}
	payload.ReasoningEffort = effort
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
			case ai.ContentText, ai.ContentJSON, ai.ContentReasoning, ai.ContentMedia:
				// Mistral carries content and tool calls in separate fields, so
				// content after a call cannot keep its canonical position.
				if len(message.ToolCalls) > 0 {
					return nil, fmt.Errorf("%w: Mistral %s content after tool calls", ai.ErrUnsupportedCapability, part.Kind)
				}
			}
			switch part.Kind {
			case ai.ContentText:
				message.Content.appendText(part.Text)
			case ai.ContentJSON:
				message.Content.appendText(string(part.JSON))
			case ai.ContentReasoning:
				chunk, err := reasoningChunk(part)
				if err != nil {
					return nil, err
				}
				message.Content = append(message.Content, chunk)
			case ai.ContentMedia:
				if m.Role != ai.RoleUser {
					return nil, &ai.UnsupportedContentError{Provider: "mistral", Kind: ai.ContentMedia, Detail: "images are only accepted in user messages"}
				}
				chunk, err := imageChunk(part.Media)
				if err != nil {
					return nil, err
				}
				message.Content = append(message.Content, chunk)
			case ai.ContentToolCall:
				c := part.ToolCall
				if err := rejectRequiredExtensions(c.Extensions); err != nil {
					return nil, err
				}
				message.ToolCalls = append(message.ToolCalls, chatMessageToolCall{ID: c.ID, Type: "function", Function: chatToolCallFunction{Name: c.Name, Arguments: string(c.Args)}})
			case ai.ContentToolResult:
				content, err := canonicalResultText(part.ToolResult)
				if err != nil {
					return nil, err
				}
				out = append(out, chatMessageRequest{Role: "tool", Content: textContent(content), ToolCallID: part.ToolResult.ToolCallID})
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
		req = req.Copy()
		if err := m.preflight(req); err != nil {
			streamErr = err
			emit(ai.Token{Err: err})
			return
		}

		payload, err := m.chatCompletionRequest(req, true)
		if err != nil {
			streamErr = err
			if !emit(ai.Token{Err: err}) {
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
			if !emit(ai.Token{Err: err}) {
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
			if !emit(ai.Token{Err: err}) {
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
			if !emit(ai.Token{Err: err}) {
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
					Err: streamErr,
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
				Err: streamErr,
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
			if !emit(ai.Token{Completion: &snapshot}) {
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
			if streamErr == nil {
				if terminalErr := mistralTerminalError(completion.FinishReason); terminalErr != nil {
					streamErr = terminalErr
					emit(ai.Token{Err: terminalErr})
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
					if !emit(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &tcCopy}}) {
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

			parts, err := parseResponseContent(chunk.Choices[0].Delta.Content)
			if err != nil {
				return fmt.Errorf("decode stream content: %w", err)
			}
			for _, part := range streamContentParts(parts) {
				p := part
				if !emit(ai.Token{Part: &p}) {
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
					if !emit(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &tcCopy}}) {
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
				if !emit(ai.Token{Err: streamErr}) {
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
					if !emit(ai.Token{Err: flushErr}) {
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
					if !emit(ai.Token{Err: flushErr}) {
						return
					}
				}
				return
			}
		}
	}()

	return ai.DetectToolCallsInStream(ctx, raw, m.debug)
}

// streamContentParts prepares delta parts for ai.Message.AppendToken. Empty
// deltas are skipped. A thinking signature is emitted after its text as an
// empty reasoning part, so AppendToken attaches it to the reasoning part
// accumulated from earlier deltas instead of splitting the thinking block.
func streamContentParts(parts []ai.ContentPart) []ai.ContentPart {
	out := make([]ai.ContentPart, 0, len(parts))
	for _, part := range parts {
		if part.Text == "" && len(part.Extensions) == 0 {
			continue
		}
		if part.Kind == ai.ContentReasoning && part.Text != "" && len(part.Extensions) > 0 {
			out = append(out, ai.ContentPart{Kind: part.Kind, Text: part.Text})
			part.Text = ""
		}
		out = append(out, part)
	}
	return out
}

func (m *Model) Generate(ctx context.Context, req ai.AIRequest) (response *ai.AIResponse, err error) {
	req = req.Copy()
	if err := m.preflight(req); err != nil {
		return nil, err
	}
	payload, err := m.chatCompletionRequest(req, false)
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
	contentParts, err := parseResponseContent(parsed.Choices[0].Message.Content)
	if err != nil {
		return nil, err
	}
	toolCalls, err := mapChatResponseToolCalls(parsed.Choices[0].Message.ToolCalls)
	if err != nil {
		return nil, err
	}
	// Mistral returns content before tool calls; keep that order. An absent or
	// string response keeps the leading text part, even when it is empty.
	if len(contentParts) == 0 {
		contentParts = ai.TextParts("")
	}
	semantic := ai.Message{Role: ai.RoleAssistant, Parts: contentParts}
	for _, call := range toolCalls {
		c := call
		semantic.Parts = append(semantic.Parts, ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &c})
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
	if err := mistralTerminalError(generationResult.FinishReason); err != nil {
		return nil, err
	}
	if gai.ObservationEnabled(ctx, m.debug) {
		fields := map[string]any{
			"input_tokens":  usage.InputTokens,
			"output_tokens": usage.OutputTokens,
		}

		gai.AddObservationContent(ctx, m.debug, fields, "response_text", gai.ContentKindCompletion, semantic.Text())
		gai.EmitObservation(ctx, m.debug, gai.Observation{
			Name:   "mistral_generate_response",
			Source: "ai:mistral.Model.Generate",
			Fields: fields,
		})
	}

	result := &ai.AIResponse{
		Raw:          append([]byte(nil), resBody...),
		FinishReason: parsed.Choices[0].FinishReason,
		InputTokens:  usage.InputTokens,
		OutputTokens: usage.OutputTokens,
	}
	result.SetMessage(semantic)
	return result, nil
}

func mistralTerminalError(reason string) error {
	switch reason {
	case "", "stop", "tool_calls", "function_call":
		return nil
	default:
		return &ai.TerminalError{Provider: "mistral", Reason: reason}
	}
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
