package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"go.opentelemetry.io/otel/attribute"
	"google.golang.org/genai"
	genaitokenizer "google.golang.org/genai/tokenizer"
)

const geminiTracerName = "github.com/lace-ai/gai/ai/gemini"

type Model struct {
	name   string
	client *Provider
	debug  gai.ObservationSink
	mu     sync.Mutex
	api    *genai.Client
}

var _ ai.Model = (*Model)(nil)
var _ ai.ModelDescriber = (*Model)(nil)

func (m *Model) Name() string {
	return m.name
}

func (m *Model) NativeTools() bool { return true }

// TokenCounter uses a local estimate for automatic budgeting. Tokenizer is
// still available explicitly, but may perform network I/O to load tokenizer data.
func (m *Model) TokenCounter() ai.TokenCounter { return ai.TextTokenEstimator{} }

func (m *Model) Tokenizer() ai.Tokenizer {
	return &Tokenizer{
		modelName: m.name,
	}
}

func (m *Model) Descriptor() ai.ModelDescriptor {
	if facts, ok := m.client.catalog.Lookup(m.name); ok {
		return effectiveGeminiDescriptor(m.name, facts)
	}
	return effectiveGeminiDescriptor(m.name, ai.ModelDescriptor{Model: m.name})
}

func geminiAdapterDescriptor(model string) ai.ModelDescriptor {
	return ai.ModelDescriptor{Model: model, NativeMessages: ai.FeatureSupportSupported, NativeTools: ai.FeatureSupportSupported,
		ToolChoiceModes: []ai.ToolChoiceMode{ai.ToolChoiceAuto, ai.ToolChoiceNone, ai.ToolChoiceRequired},
		Multimodal:      ai.FeatureSupportUnsupported,
		Usage:           ai.FeatureSupportSupported, FinishReason: ai.FeatureSupportSupported, StreamingUsage: ai.FeatureSupportSupported,
		ToolCalling: ai.FeatureSupportSupported,
		JSONOutput:  ai.FeatureSupportSupported, JSONSchemaOutput: ai.FeatureSupportSupported,
		Reasoning: ai.FeatureSupportSupported, ReasoningEffort: ai.FeatureSupportSupported, ReasoningEfforts: []ai.ReasoningEffort{ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh},
		Tokenizer: ai.TokenizerDescriptor{Available: ai.FeatureSupportSupported, Fidelity: ai.TokenizerFidelityExact}}
}

func geminiHTTPStatus(err error) int {
	var apiErr *genai.APIError
	if errors.As(err, &apiErr) && apiErr != nil {
		return apiErr.Code
	}
	var apiErrValue genai.APIError
	if errors.As(err, &apiErrValue) {
		return apiErrValue.Code
	}
	return 0
}

func classifyProviderError(err error) error {
	var apiErr *genai.APIError
	if errors.As(err, &apiErr) && apiErr != nil {
		return ai.ClassifyProviderError(err, apiErr.Code, "", "", nil)
	}
	var apiErrValue genai.APIError
	if errors.As(err, &apiErrValue) {
		return ai.ClassifyProviderError(err, apiErrValue.Code, "", "", nil)
	}
	return err
}

func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.api = nil
	return nil
}

func (m *Model) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)

	go func() {
		ctx := ctx
		var streamErr error
		var observation *ai.GenerationObservation
		generationResult := ai.GenerationResult{}
		completion := ai.Completion{Provider: "gemini"}
		hasCompletion := false
		defer close(out)
		defer func() {
			if hasCompletion {
				generationResult.ResponseModel = completion.Model
				generationResult.RequestID = completion.RequestID
				generationResult.FinishReason = completion.FinishReason
				if completion.UsageReported {
					usage := completion.Usage
					generationResult.Usage = &usage
				}
			}
			generationResult.HTTPStatus = geminiHTTPStatus(streamErr)
			generationResult.Err = streamErr
			observation.Finish(generationResult)
		}()
		emit := func(token ai.Token) bool {
			observation.ObserveToken(token)
			if ai.SendToken(ctx, out, token) {
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
			emit(ai.Token{Err: err, Type: ai.TokenTypeErr, Text: err.Error()})
			return
		}

		client, err := m.getClient(ctx)
		if err != nil {
			streamErr = err
			if gai.ObservationEnabled(ctx, m.debug) {
				gai.EmitObservation(ctx, m.debug, gai.Observation{
					Name:   "gemini_get_client_failed",
					Source: "ai:gemini.Model.GenerateStream",
					Fields: map[string]any{
						"error": err.Error(),
					},
					Err: err,
				})
			}
			emit(ai.Token{Err: err, Type: ai.TokenTypeErr, Text: err.Error()})
			return
		}

		config, err := buildGenerateContentConfig(req)
		if err != nil {
			streamErr = err
			emit(ai.Token{Err: err, Type: ai.TokenTypeErr, Text: err.Error()})
			return
		}

		contents, err := nativeContents(req)
		if err != nil {
			streamErr = err
			emit(ai.Token{Err: err, Type: ai.TokenTypeErr, Text: err.Error()})
			return
		}

		generationCtx, startedObservation := ai.StartGenerationObservation(ctx, req, ai.GenerationConfig{Provider: "gemini", Model: m.name, Streaming: true, Sink: m.debug})
		observation = startedObservation
		ctx = generationCtx
		if gai.ObservationEnabled(ctx, m.debug) {
			fields := map[string]any{"max_tokens": req.MaxTokens}
			gai.AddObservationContent(ctx, m.debug, fields, "prompt", gai.ContentKindPrompt, observationPrompt(req))
			gai.EmitObservation(ctx, m.debug, gai.Observation{
				Name:   "gemini_stream_request",
				Source: "ai:gemini.Model.GenerateStream",
				Fields: fields,
			})
		}
		for resp, err := range client.Models.GenerateContentStream(generationCtx, m.name, contents, config) {
			if err != nil {
				streamErr = classifyProviderError(fmt.Errorf("error generating content stream: %w", err))
				if gai.ObservationEnabled(ctx, m.debug) {
					gai.EmitObservation(ctx, m.debug, gai.Observation{
						Name:   "gemini_stream_generation_failed",
						Source: "ai:gemini.Model.GenerateStream",
						Fields: map[string]any{
							"error": err.Error(),
						},
						Err: err,
					})
				}
				emit(ai.Token{Err: streamErr, Type: ai.TokenTypeErr, Text: streamErr.Error()})
				return
			}

			if resp == nil {
				continue
			}
			responseHasCompletion := false
			if resp.ResponseID != "" {
				completion.RequestID = resp.ResponseID
				hasCompletion = true
				responseHasCompletion = true
			}
			if resp.ModelVersion != "" {
				completion.Model = resp.ModelVersion
				hasCompletion = true
				responseHasCompletion = true
			}
			if resp.UsageMetadata != nil {
				completion.UsageReported = true
				completion.Usage = ai.Usage{
					InputTokens:     int(resp.UsageMetadata.PromptTokenCount),
					OutputTokens:    int(resp.UsageMetadata.CandidatesTokenCount),
					ReasoningTokens: int(resp.UsageMetadata.ThoughtsTokenCount),
					CachedTokens:    int(resp.UsageMetadata.CachedContentTokenCount),
					ToolUseTokens:   int(resp.UsageMetadata.ToolUsePromptTokenCount),
				}
				hasCompletion = true
				responseHasCompletion = true
			}
			if len(resp.Candidates) > 0 && resp.Candidates[0] != nil && resp.Candidates[0].FinishReason != "" {
				completion.FinishReason = string(resp.Candidates[0].FinishReason)
				hasCompletion = true
				responseHasCompletion = true
			}
			if responseHasCompletion {
				raw, marshalErr := json.Marshal(resp)
				if marshalErr != nil {
					streamErr = fmt.Errorf("encode Gemini completion metadata: %w", marshalErr)
					emit(ai.Token{Err: streamErr, Type: ai.TokenTypeErr, Text: streamErr.Error()})
					return
				}
				completion.Raw = append(completion.Raw[:0], raw...)
			}

			if len(resp.Candidates) == 0 || resp.Candidates[0] == nil || resp.Candidates[0].Content == nil {
				continue
			}

			for _, part := range resp.Candidates[0].Content.Parts {
				if part == nil || isEmptyGeminiPart(part) {
					continue
				}
				if hasUnsupportedGeminiPartPayload(part) {
					streamErr = fmt.Errorf("%w: Gemini stream output part", ai.ErrUnsupportedCapability)
					emit(ai.Token{Type: ai.TokenTypeErr, Err: streamErr, Text: streamErr.Error()})
					return
				}

				switch {
				case part.Text != "":
					token := buildTextToken(part)
					if !emit(token) {
						return
					}
				case part.FunctionCall != nil:
					rawPart, err := json.Marshal(part)
					if err != nil {
						encodeErr := fmt.Errorf("error encoding part: %w", err)
						streamErr = encodeErr
						if gai.ObservationEnabled(ctx, m.debug) {
							fields := map[string]any{
								"error": err.Error(),
							}
							gai.AddObservationContent(ctx, m.debug, fields, "part", gai.ContentKindToolInput, rawPart)
							gai.EmitObservation(ctx, m.debug, gai.Observation{
								Name:   "gemini_stream_part_encoding_failed",
								Source: "ai:gemini.Model.GenerateStream",
								Fields: fields,
								Err:    err,
							})
						}
						emit(ai.Token{Err: encodeErr, Type: ai.TokenTypeErr, Text: encodeErr.Error()})
						return
					}
					toolCall, err := mapFunctionCall(part.FunctionCall)
					if err != nil {
						mapErr := fmt.Errorf("error mapping function call: %w", err)
						streamErr = mapErr
						if gai.ObservationEnabled(ctx, m.debug) {
							fields := map[string]any{
								"error": err.Error(),
							}
							if gai.ObservationContentEnabled(ctx, m.debug, gai.ContentKindToolInput) {
								fields["function_call_name"] = part.FunctionCall.Name
								gai.AddObservationContent(ctx, m.debug, fields, "function_call_args", gai.ContentKindToolInput, part.FunctionCall.Args)
							}
							gai.EmitObservation(ctx, m.debug, gai.Observation{
								Name:   "gemini_stream_function_call_mapping_failed",
								Source: "ai:gemini.Model.GenerateStream",
								Fields: fields,
								Err:    err,
							})
						}
						emit(ai.Token{Err: mapErr, Type: ai.TokenTypeErr, Text: mapErr.Error()})
						return
					}
					toolCall.Extensions = thoughtExtensions(part.ThoughtSignature)
					if gai.ObservationEnabled(ctx, m.debug) {
						fields := map[string]any{}
						if gai.ObservationContentEnabled(ctx, m.debug, gai.ContentKindToolInput) {
							fields["tool_call_id"] = toolCall.ID
							gai.AddObservationContent(ctx, m.debug, fields, "tool_call_args", gai.ContentKindToolInput, toolCall.Args)
						}
						gai.EmitObservation(ctx, m.debug, gai.Observation{
							Name:   "gemini_stream_function_call_mapped",
							Source: "ai:gemini.Model.GenerateStream",
							Fields: fields,
						})
					}
					if !emit(ai.Token{
						Type:     ai.TokenTypeToolCall,
						Data:     rawPart,
						ToolCall: toolCall,
					}) {
						return
					}
				case len(part.ThoughtSignature) > 0:
					semantic := ai.ContentPart{Kind: ai.ContentExtension, Extensions: thoughtExtensions(part.ThoughtSignature)}
					if !emit(ai.Token{Type: ai.TokenTypePart, Part: &semantic}) {
						return
					}
				default:
					streamErr = fmt.Errorf("%w: Gemini stream output part", ai.ErrUnsupportedCapability)
					emit(ai.Token{Type: ai.TokenTypeErr, Err: streamErr, Text: streamErr.Error()})
					return
				}
			}
		}
		if hasCompletion {
			snapshot := completion
			snapshot.Raw = append(json.RawMessage(nil), completion.Raw...)
			emit(ai.Token{Type: ai.TokenTypeCompletion, Completion: &snapshot})
		}
	}()

	return out
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
	client, err := m.getClient(ctx)
	if err != nil {
		if gai.ObservationEnabled(ctx, m.debug) {
			gai.EmitObservation(ctx, m.debug, gai.Observation{
				Name:   "gemini_get_client_failed",
				Source: "ai:gemini.Model.Generate",
				Fields: map[string]any{
					"error": err.Error(),
				},
				Err: err,
			})
		}
		return nil, err
	}

	config, err := buildGenerateContentConfig(req)
	if err != nil {
		return nil, err
	}

	contents, err := nativeContents(req)
	if err != nil {
		return nil, err
	}
	ctx, observation := ai.StartGenerationObservation(ctx, req, ai.GenerationConfig{Provider: "gemini", Model: m.name, Sink: m.debug})
	if gai.ObservationEnabled(ctx, m.debug) {
		fields := map[string]any{"max_tokens": req.MaxTokens}
		gai.AddObservationContent(ctx, m.debug, fields, "prompt", gai.ContentKindPrompt, observationPrompt(req))
		gai.EmitObservation(ctx, m.debug, gai.Observation{
			Name:   "gemini_generate_request",
			Source: "ai:gemini.Model.Generate",
			Fields: fields,
		})
	}
	generationResult := ai.GenerationResult{}
	defer func() {
		generationResult.HTTPStatus = geminiHTTPStatus(err)
		generationResult.Err = err
		observation.Finish(generationResult)
	}()
	result, err := client.Models.GenerateContent(
		ctx,
		m.name,
		contents,
		config,
	)
	if err != nil {
		if gai.ObservationEnabled(ctx, m.debug) {
			gai.EmitObservation(ctx, m.debug, gai.Observation{
				Name:   "gemini_generate_content_failed",
				Source: "ai:gemini.Model.Generate",
				Fields: map[string]any{
					"error": err.Error(),
				},
				Err: err,
			})
		}
		return nil, err
	}

	inputTokens := 0
	outputTokens := 0
	if result.UsageMetadata != nil {
		inputTokens = int(result.UsageMetadata.PromptTokenCount)
		outputTokens = int(result.UsageMetadata.CandidatesTokenCount)
	}
	reasoningTokens := 0
	if result.UsageMetadata != nil {
		reasoningTokens = int(result.UsageMetadata.ThoughtsTokenCount)
	}
	semantic, err := mapCanonicalResponse(result)
	text, reasoning, toolCalls := semantic.Text(), semantic.Reasoning(), semantic.ToolCalls()
	if err != nil {
		return nil, err
	}
	if gai.ObservationEnabled(ctx, m.debug) {
		fields := map[string]any{
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
		}
		gai.AddObservationContent(ctx, m.debug, fields, "response_text", gai.ContentKindCompletion, text)
		gai.AddObservationContent(ctx, m.debug, fields, "reasoning", gai.ContentKindReasoning, reasoning)
		gai.EmitObservation(ctx, m.debug, gai.Observation{
			Name:   "gemini_generate_content_success",
			Source: "ai:gemini.Model.Generate",
			Fields: fields,
		})
	}
	raw, _ := json.Marshal(result)
	generationResult.ResponseModel = result.ModelVersion
	generationResult.RequestID = result.ResponseID
	generationResult.ToolCallCount = len(toolCalls)
	if result.UsageMetadata != nil {
		usage := ai.Usage{
			InputTokens: inputTokens, OutputTokens: outputTokens, ReasoningTokens: reasoningTokens,
			CachedTokens: int(result.UsageMetadata.CachedContentTokenCount), ToolUseTokens: int(result.UsageMetadata.ToolUsePromptTokenCount),
		}
		generationResult.Usage = &usage
	}
	if len(result.Candidates) > 0 && result.Candidates[0] != nil {
		generationResult.FinishReason = string(result.Candidates[0].FinishReason)
	}
	response = &ai.AIResponse{
		Text:            text,
		Reasoning:       reasoning,
		ToolCalls:       toolCalls,
		Raw:             raw,
		InputTokens:     inputTokens,
		OutputTokens:    outputTokens,
		ReasoningTokens: reasoningTokens,
		FinishReason:    generationResult.FinishReason,
	}
	response.SetMessage(semantic)
	return response, nil
}

// nativeContents maps only the conversation; system messages are mapped to
// GenerateContentConfig.SystemInstruction by buildGenerateContentConfig.
func nativeContents(req ai.AIRequest) ([]*genai.Content, error) {
	req, err := req.Normalized()
	if err != nil {
		return nil, err
	}
	out := make([]*genai.Content, 0, len(req.Messages))
	for _, m := range req.Messages {
		if err := rejectRequiredGeminiExtensions(m.Extensions); err != nil {
			return nil, err
		}
		if m.Role == ai.RoleSystem {
			continue
		}
		parts := make([]*genai.Part, 0, len(m.Parts))
		for _, part := range m.Parts {
			signature, err := geminiSignature(part.Extensions)
			if err != nil {
				return nil, err
			}
			switch part.Kind {
			case ai.ContentText, ai.ContentJSON, ai.ContentReasoning:
				text := part.Text
				if part.Kind == ai.ContentJSON {
					text = string(part.JSON)
				}
				parts = append(parts, &genai.Part{Text: text, Thought: part.Kind == ai.ContentReasoning, ThoughtSignature: signature})
			case ai.ContentToolCall:
				c := part.ToolCall
				callSignature, err := geminiSignature(c.Extensions)
				if err != nil {
					return nil, err
				}
				if len(callSignature) == 0 {
					callSignature = c.ThoughtSignatureBytes()
				}
				if len(signature) != 0 {
					return nil, fmt.Errorf("%w: Gemini signature must be attached to the tool call", ai.ErrUnsupportedCapability)
				}
				var args map[string]any
				if err := json.Unmarshal(c.Args, &args); err != nil {
					return nil, err
				}
				parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{ID: c.ID, Name: c.Name, Args: args}, ThoughtSignature: callSignature})
			case ai.ContentToolResult:
				if len(signature) > 0 {
					return nil, fmt.Errorf("%w: Gemini tool-result signature", ai.ErrUnsupportedCapability)
				}
				r := part.ToolResult
				var text strings.Builder
				for _, resultPart := range r.Parts {
					if _, err := geminiSignature(resultPart.Extensions); err != nil {
						return nil, err
					}
					if len(resultPart.Extensions) != 0 {
						return nil, fmt.Errorf("%w: Gemini result signatures", ai.ErrUnsupportedCapability)
					}
					switch resultPart.Kind {
					case ai.ContentText:
						text.WriteString(resultPart.Text)
					case ai.ContentJSON:
						text.Write(resultPart.JSON)
					default:
						return nil, fmt.Errorf("%w: Gemini result %q", ai.ErrUnsupportedCapability, resultPart.Kind)
					}
				}
				key := "output"
				if r.IsError {
					key = "error"
				}
				parts = append(parts, &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: r.ToolCallID, Name: r.Name, Response: map[string]any{key: text.String()}}})
			case ai.ContentExtension:
				if len(signature) > 0 {
					parts = append(parts, &genai.Part{ThoughtSignature: signature})
				}
			default:
				return nil, fmt.Errorf("%w: Gemini content %q", ai.ErrUnsupportedCapability, part.Kind)
			}
		}
		role := genai.RoleUser
		if m.Role == ai.RoleAssistant {
			role = genai.RoleModel
		}
		if len(parts) > 0 {
			out = append(out, &genai.Content{Role: role, Parts: parts})
		}
	}
	return out, nil
}

func geminiSignature(extensions []ai.Extension) ([]byte, error) {
	var signature []byte
	for _, ext := range extensions {
		if ext.Namespace == "google" && ext.Type == "thought_signature" {
			if err := json.Unmarshal(ext.Data, &signature); err != nil {
				return nil, fmt.Errorf("decode Gemini thought signature: %w", err)
			}
		} else if ext.Required {
			return nil, fmt.Errorf("%w: Gemini extension %s/%s", ai.ErrUnsupportedCapability, ext.Namespace, ext.Type)
		}
	}
	return signature, nil
}

func thoughtExtensions(signature []byte) []ai.Extension {
	if len(signature) == 0 {
		return nil
	}
	data, _ := json.Marshal(signature)
	return []ai.Extension{{Namespace: "google", Type: "thought_signature", Data: data, Required: true}}
}

func buildGenerateContentConfig(req ai.AIRequest) (*genai.GenerateContentConfig, error) {
	if err := req.ResponseFormat.Validate(); err != nil {
		return nil, err
	}

	var config *genai.GenerateContentConfig
	ensureConfig := func() *genai.GenerateContentConfig {
		if config == nil {
			config = &genai.GenerateContentConfig{}
		}
		return config
	}

	normalized, err := req.Normalized()
	if err != nil {
		return nil, err
	}
	for _, message := range normalized.Messages {
		if message.Role != ai.RoleSystem {
			continue
		}
		if len(message.Extensions) != 0 {
			return nil, fmt.Errorf("%w: Gemini system extensions", ai.ErrUnsupportedCapability)
		}
		system := ensureConfig().SystemInstruction
		if system == nil {
			system = &genai.Content{}
			ensureConfig().SystemInstruction = system
		}
		for _, part := range message.Parts {
			if len(part.Extensions) > 0 {
				return nil, fmt.Errorf("%w: Gemini system part extensions", ai.ErrUnsupportedCapability)
			}
			switch part.Kind {
			case ai.ContentText:
				system.Parts = append(system.Parts, &genai.Part{Text: part.Text})
			case ai.ContentJSON:
				system.Parts = append(system.Parts, &genai.Part{Text: string(part.JSON)})
			default:
				return nil, fmt.Errorf("%w: Gemini system content %q", ai.ErrUnsupportedCapability, part.Kind)
			}
		}
	}
	if req.MaxTokens > 0 {
		ensureConfig().MaxOutputTokens = int32(req.MaxTokens)
	}
	if len(req.Tools) > 0 {
		tools, err := mapGenerateContentTools(req.Tools)
		if err != nil {
			return nil, err
		}
		ensureConfig().Tools = tools
		toolConfig, err := mapGenerateContentToolConfig(req.ToolChoice)
		if err != nil {
			return nil, err
		}
		if toolConfig != nil {
			ensureConfig().ToolConfig = toolConfig
		}
	}
	if err := applyGenerateContentResponseFormat(ensureConfig, req.ResponseFormat); err != nil {
		return nil, err
	}
	if thinking := mapGenerateContentThinkingConfig(req.Reasoning); thinking != nil {
		ensureConfig().ThinkingConfig = thinking
	}
	return config, nil
}

func mapGenerateContentTools(definitions []ai.ToolDefinition) ([]*genai.Tool, error) {
	declarations := make([]*genai.FunctionDeclaration, 0, len(definitions))
	for _, definition := range definitions {
		if err := definition.Validate(); err != nil {
			return nil, err
		}
		var schema any
		if err := json.Unmarshal(definition.Parameters, &schema); err != nil {
			return nil, err
		}
		declarations = append(declarations, &genai.FunctionDeclaration{
			Name:                 definition.Name,
			Description:          definition.Description,
			ParametersJsonSchema: schema,
		})
	}
	return []*genai.Tool{{FunctionDeclarations: declarations}}, nil
}

func mapGenerateContentToolConfig(choice ai.ToolChoice) (*genai.ToolConfig, error) {
	if len(choice.Names) > 0 {
		switch choice.Mode {
		case ai.ToolChoiceAuto, "":
			return nil, fmt.Errorf("gemini tool choice %q with specific tool names is unsupported: Gemini SDK cannot enforce allowed tool names in auto mode", ai.ToolChoiceAuto)
		case ai.ToolChoiceNone:
			return nil, fmt.Errorf("gemini tool choice %q with specific tool names is invalid: no tools may be called", ai.ToolChoiceNone)
		}
	}

	var mode genai.FunctionCallingConfigMode
	switch choice.Mode {
	case ai.ToolChoiceNone:
		mode = genai.FunctionCallingConfigModeNone
	case ai.ToolChoiceRequired:
		mode = genai.FunctionCallingConfigModeAny
	case ai.ToolChoiceAuto, "":
		return nil, nil
	default:
		return nil, fmt.Errorf("unsupported gemini tool choice mode %q", choice.Mode)
	}
	return &genai.ToolConfig{
		FunctionCallingConfig: &genai.FunctionCallingConfig{
			Mode:                 mode,
			AllowedFunctionNames: append([]string(nil), choice.Names...),
		},
	}, nil
}

func applyGenerateContentResponseFormat(ensureConfig func() *genai.GenerateContentConfig, format ai.ResponseFormat) error {
	switch format.Type {
	case "", ai.ResponseFormatText:
		return nil
	case ai.ResponseFormatJSONObject:
		ensureConfig().ResponseMIMEType = "application/json"
		return nil
	case ai.ResponseFormatJSONSchema:
		var schema any
		if err := json.Unmarshal(format.Schema, &schema); err != nil {
			return err
		}
		cfg := ensureConfig()
		cfg.ResponseMIMEType = "application/json"
		cfg.ResponseJsonSchema = schema
		return nil
	default:
		return fmt.Errorf("%w: %s", ai.ErrInvalidResponseFormat, format.Type)
	}
}

func mapGenerateContentThinkingConfig(reasoning ai.ReasoningConfig) *genai.ThinkingConfig {
	if !reasoning.Enabled && !reasoning.IncludeThoughts && reasoning.BudgetTokens <= 0 && reasoning.Effort == "" {
		return nil
	}
	config := &genai.ThinkingConfig{
		IncludeThoughts: reasoning.IncludeThoughts,
	}
	if reasoning.BudgetTokens > 0 {
		budget := int32(reasoning.BudgetTokens)
		config.ThinkingBudget = &budget
	}
	switch reasoning.Effort {
	case ai.ReasoningEffortLow:
		config.ThinkingLevel = genai.ThinkingLevelLow
	case ai.ReasoningEffortMedium:
		config.ThinkingLevel = genai.ThinkingLevelMedium
	case ai.ReasoningEffortHigh:
		config.ThinkingLevel = genai.ThinkingLevelHigh
	}
	return config
}

func mapGenerateContentResponse(result *genai.GenerateContentResponse) (string, string, []ai.ToolCall, error) {
	message, err := mapCanonicalResponse(result)
	return message.Text(), message.Reasoning(), message.ToolCalls(), err
}

func mapCanonicalResponse(result *genai.GenerateContentResponse) (ai.Message, error) {
	message := ai.Message{Role: ai.RoleAssistant}
	if result != nil && len(result.Candidates) > 0 && result.Candidates[0] != nil && result.Candidates[0].Content != nil {
		for _, part := range result.Candidates[0].Content.Parts {
			if part == nil || isEmptyGeminiPart(part) {
				continue
			}
			if hasUnsupportedGeminiPartPayload(part) {
				return ai.Message{}, fmt.Errorf("%w: Gemini output part", ai.ErrUnsupportedCapability)
			}
			switch {
			case part.Text != "":
				kind := ai.ContentText
				if part.Thought {
					kind = ai.ContentReasoning
				}
				message.Parts = append(message.Parts, ai.ContentPart{Kind: kind, Text: part.Text, Extensions: thoughtExtensions(part.ThoughtSignature)})
			case part.FunctionCall != nil:
				call, err := mapFunctionCall(part.FunctionCall)
				if err != nil {
					return ai.Message{}, err
				}
				call.Extensions = thoughtExtensions(part.ThoughtSignature)
				message.Parts = append(message.Parts, ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: call})
			case len(part.ThoughtSignature) > 0:
				message.Parts = append(message.Parts, ai.ContentPart{Kind: ai.ContentExtension, Extensions: thoughtExtensions(part.ThoughtSignature)})
			default:
				return ai.Message{}, fmt.Errorf("%w: Gemini output part", ai.ErrUnsupportedCapability)
			}
		}
	}
	if len(message.Parts) == 0 {
		message.Parts = ai.TextParts("")
	}
	return message, nil
}

// isEmptyGeminiPart identifies no-op parts, including terminal empty text chunks.
// Flags, signatures, and unsupported payloads must not be mistaken for no content.
func isEmptyGeminiPart(part *genai.Part) bool {
	return part.Text == "" && !part.Thought && part.FunctionCall == nil &&
		len(part.ThoughtSignature) == 0 && !hasUnsupportedGeminiPartPayload(part)
}

// Reject unsupported content even when a supported field is also populated;
// selecting the first supported field would silently discard the other payload.
func hasUnsupportedGeminiPartPayload(part *genai.Part) bool {
	return part.MediaResolution != nil || part.CodeExecutionResult != nil ||
		part.ExecutableCode != nil || part.FileData != nil || part.FunctionResponse != nil ||
		part.InlineData != nil || part.VideoMetadata != nil || part.ToolCall != nil ||
		part.ToolResponse != nil || len(part.PartMetadata) != 0 ||
		part.AudioTranscription != nil || part.MediaProcessing != ""
}

func mapFunctionCall(functionCall *genai.FunctionCall) (*ai.ToolCall, error) {
	toolName := strings.TrimSpace(functionCall.Name)
	if toolName == "" {
		return nil, fmt.Errorf("function call name empty")
	}

	args, err := marshalArgs(functionCall.Args)
	if err != nil {
		return nil, err
	}
	id := functionCall.ID
	if id == "" {
		id = ai.GenerateToolCallID(toolName)
	}
	return &ai.ToolCall{
		ID:   id,
		Type: "function",
		Name: toolName,
		Args: args,
	}, nil
}

func buildTextToken(part *genai.Part) ai.Token {
	kind, typ := ai.ContentText, ai.TokenTypeText
	if part.Thought {
		kind, typ = ai.ContentReasoning, ai.TokenTypeThought
	}
	semantic := ai.ContentPart{Kind: kind, Text: part.Text, Extensions: thoughtExtensions(part.ThoughtSignature)}
	return ai.Token{Type: typ, Text: part.Text, Data: []byte(part.Text), Part: &semantic}
}

func marshalArgs(args map[string]any) (json.RawMessage, error) {
	if args == nil {
		return json.RawMessage("{}"), nil
	}

	raw, err := json.Marshal(args)
	if err != nil {
		return nil, err
	}

	return json.RawMessage(raw), nil
}

func (m *Model) getClient(ctx context.Context) (*genai.Client, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.api != nil {
		return m.api, nil
	}

	client, err := m.client.getClient(ctx)
	if err != nil {
		return nil, err
	}

	m.api = client
	return m.api, nil
}

type Tokenizer struct {
	modelName string
	mu        sync.Mutex
	local     *genaitokenizer.LocalTokenizer
}

func (t *Tokenizer) ID() string {
	return "gemini." + t.modelName
}

func (t *Tokenizer) CountTokens(ctx context.Context, text string) (tokens int, err error) {
	_, span := gai.StartOperationSpan(ctx, geminiTracerName, "ai.gemini", "ai.operation", "tokenizer.count_tokens",
		attribute.String("ai.provider", "gemini"),
		attribute.String("ai.model", t.modelName),
		attribute.String("ai.tokenizer", t.ID()),
		attribute.Int("ai.input_length", len(text)),
	)
	defer func() {
		span.SetAttributes(attribute.Int("ai.input_tokens", tokens))
		gai.EndSpan(span, err)
	}()

	local, err := t.getLocal()
	if err != nil {
		return 0, err
	}

	result, err := local.CountTokens(genai.Text(text), nil)
	if err != nil {
		return 0, err
	}
	return int(result.TotalTokens), nil
}

func (t *Tokenizer) Tokenize(ctx context.Context, text string) (tokens []string, err error) {
	_, span := gai.StartOperationSpan(ctx, geminiTracerName, "ai.gemini", "ai.operation", "tokenizer.tokenize",
		attribute.String("ai.provider", "gemini"),
		attribute.String("ai.model", t.modelName),
		attribute.String("ai.tokenizer", t.ID()),
		attribute.Int("ai.input_length", len(text)),
	)
	defer func() {
		span.SetAttributes(attribute.Int("ai.input_tokens", len(tokens)))
		gai.EndSpan(span, err)
	}()

	local, err := t.getLocal()
	if err != nil {
		return nil, err
	}

	result, err := local.ComputeTokens(genai.Text(text))
	if err != nil {
		return nil, err
	}

	for _, info := range result.TokensInfo {
		for _, token := range info.Tokens {
			tokens = append(tokens, string(token))
		}
	}
	return tokens, nil
}

func (t *Tokenizer) getLocal() (*genaitokenizer.LocalTokenizer, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.local != nil {
		return t.local, nil
	}

	local, err := genaitokenizer.NewLocalTokenizer(t.modelName)
	if err != nil {
		return nil, err
	}
	t.local = local
	return t.local, nil
}

func rejectRequiredGeminiExtensions(extensions []ai.Extension) error {
	for _, ext := range extensions {
		if ext.Required {
			return fmt.Errorf("%w: Gemini extension %s/%s", ai.ErrUnsupportedCapability, ext.Namespace, ext.Type)
		}
	}
	return nil
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
