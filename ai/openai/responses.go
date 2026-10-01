package openai

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

func (m *Model) generateResponses(ctx context.Context, req ai.AIRequest) (result *ai.AIResponse, err error) {
	params, err := buildResponsesParams(m.name, req)
	if err != nil {
		return nil, err
	}
	ctx, observation := ai.StartGenerationObservation(ctx, req, ai.GenerationConfig{Provider: "openai", Model: m.name, Sink: m.provider.debug})
	generationResult := ai.GenerationResult{}
	defer func() {
		generationResult.Err = err
		generationResult.HTTPStatus = openAIHTTPStatus(err)
		observation.Finish(generationResult)
	}()
	response, err := m.client(false).Responses.New(ctx, params)
	if err != nil {
		return nil, err
	}
	if response.Error.Message != "" {
		return nil, fmt.Errorf("OpenAI Responses API: %s", response.Error.Message)
	}
	if response.Status == "failed" {
		message := string(response.Error.Code)
		if message == "" {
			message = "response failed"
		}
		return nil, fmt.Errorf("OpenAI Responses API: %s", message)
	}
	result, err = responseFromResponses(response)
	if result != nil {
		generationResult.ResponseModel = string(response.Model)
		generationResult.RequestID = response.ID
		generationResult.FinishReason = string(response.Status)
		generationResult.ToolCallCount = len(result.ToolCalls)
		if response.JSON.Usage.Valid() {
			usage := ai.Usage{
				InputTokens: result.InputTokens, OutputTokens: result.OutputTokens,
				ReasoningTokens: result.ReasoningTokens, CachedTokens: int(response.Usage.InputTokensDetails.CachedTokens),
			}
			generationResult.Usage = &usage
		}
	}
	return result, err
}

func (m *Model) generateResponsesStream(ctx context.Context, out chan<- ai.Token, req ai.AIRequest) {
	params, err := buildResponsesParams(m.name, req)
	if err != nil {
		ai.SendToken(ctx, out, ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
		return
	}
	ctx, observation := ai.StartGenerationObservation(ctx, req, ai.GenerationConfig{Provider: "openai", Model: m.name, Streaming: true, Sink: m.provider.debug})
	var streamErr error
	generationResult := ai.GenerationResult{}
	defer func() {
		generationResult.Err = streamErr
		generationResult.HTTPStatus = openAIHTTPStatus(streamErr)
		observation.Finish(generationResult)
	}()
	emit := func(token ai.Token) bool {
		if token.Type == ai.TokenTypeErr && token.Err != nil {
			streamErr = token.Err
		}
		observation.ObserveToken(token)
		if ai.SendToken(ctx, out, token) {
			return true
		}
		streamErr = ctx.Err()
		return false
	}
	stream := m.client(true).Responses.NewStreaming(ctx, params)
	defer func() {
		if err := stream.Close(); err != nil && m.provider.debug != nil {
			gai.EmitObservation(ctx, m.provider.debug, gai.Observation{
				Name:   "stream_close_failed",
				Source: "ai:openai.Model.generateResponsesStream",
				Err:    err,
			})
		}
	}()
	for stream.Next() {
		event := stream.Current()
		switch event.Type {
		case "response.output_text.delta", "response.refusal.delta":
			if text := event.Delta.OfString; text != "" && !emit(ai.Token{Type: ai.TokenTypeText, Data: []byte(text), Text: text}) {
				return
			}
		case "response.completed":
			response := event.Response
			completion := ai.Completion{
				Provider: "openai", Model: string(response.Model), RequestID: response.ID,
				FinishReason: string(response.Status), Raw: json.RawMessage(response.RawJSON()),
			}
			if response.JSON.Usage.Valid() {
				completion.UsageReported = true
				completion.Usage = ai.Usage{
					InputTokens: int(response.Usage.InputTokens), OutputTokens: int(response.Usage.OutputTokens),
					ReasoningTokens: int(response.Usage.OutputTokensDetails.ReasoningTokens), CachedTokens: int(response.Usage.InputTokensDetails.CachedTokens),
				}
			}
			if !emit(ai.Token{Type: ai.TokenTypeCompletion, Completion: &completion}) {
				return
			}
		case "response.output_item.done":
			if event.Item.Type == "reasoning" {
				part := reasoningExtension(json.RawMessage(event.Item.RawJSON()))
				if !emit(ai.Token{Type: ai.TokenTypePart, Part: &part}) {
					return
				}
				continue
			}
			if event.Item.Type != "function_call" {
				continue
			}
			args := json.RawMessage(event.Item.Arguments)
			if !json.Valid(args) {
				err := fmt.Errorf("invalid JSON arguments for tool %q", event.Item.Name)
				streamErr = err
				emit(ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
				return
			}
			call := &ai.ToolCall{ID: event.Item.CallID, Type: "function", Name: event.Item.Name, Args: args}
			if !emit(ai.Token{Type: ai.TokenTypeToolCall, Data: []byte(args), ToolCall: call}) {
				return
			}
		case "error":
			err := fmt.Errorf("OpenAI Responses API: %s", event.Message)
			streamErr = err
			emit(ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		case "response.failed":
			response := event.Response
			message := response.Error.Message
			if message == "" {
				message = string(response.Error.Code)
			}
			if message == "" {
				message = "response failed"
			}
			err := fmt.Errorf("OpenAI Responses API: %s", message)
			if response.Error.Code != "" {
				err = ai.ClassifyProviderError(err, 0, string(response.Error.Code), response.ID, nil)
			}
			streamErr = err
			emit(ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		}
	}
	if err := stream.Err(); err != nil {
		streamErr = classifyProviderError(err)
		emit(ai.Token{Type: ai.TokenTypeErr, Err: streamErr, Text: streamErr.Error()})
	}
}

func buildResponsesParams(model string, req ai.AIRequest) (responses.ResponseNewParams, error) {
	params := responses.ResponseNewParams{
		Model:   model,
		Store:   param.NewOpt(false),
		Include: []responses.ResponseIncludable{responses.ResponseIncludableReasoningEncryptedContent},
	}
	req, err := req.Normalized()
	if err != nil {
		return responses.ResponseNewParams{}, err
	}
	input, err := mapResponsesMessages(req.Messages)
	if err != nil {
		return responses.ResponseNewParams{}, err
	}
	params.Input.OfInputItemList = input
	if req.MaxTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(req.MaxTokens))
	}
	switch req.ResponseFormat.Type {
	case "", ai.ResponseFormatText:
	case ai.ResponseFormatJSONObject:
		params.Text.Format.OfJSONObject = &shared.ResponseFormatJSONObjectParam{}
	case ai.ResponseFormatJSONSchema:
		var schema map[string]any
		if err := json.Unmarshal(req.ResponseFormat.Schema, &schema); err != nil {
			return responses.ResponseNewParams{}, err
		}
		params.Text.Format = responses.ResponseFormatTextConfigParamOfJSONSchema(req.ResponseFormat.Name, schema)
	default:
		return responses.ResponseNewParams{}, fmt.Errorf("%w: %s", ai.ErrInvalidResponseFormat, req.ResponseFormat.Type)
	}
	if req.Reasoning.Effort != "" {
		switch req.Reasoning.Effort {
		case ai.ReasoningEffortNone, ai.ReasoningEffortMinimal, ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh, ai.ReasoningEffortXHigh, ai.ReasoningEffortMax:
			params.Reasoning.Effort = shared.ReasoningEffort(req.Reasoning.Effort)
		default:
			return responses.ResponseNewParams{}, fmt.Errorf("%w: OpenAI reasoning effort %q", ai.ErrUnsupportedCapability, req.Reasoning.Effort)
		}
	}
	available := make(map[string]struct{}, len(req.Tools))
	for _, definition := range req.Tools {
		if err := definition.Validate(); err != nil {
			return responses.ResponseNewParams{}, err
		}
		var schema map[string]any
		if err := json.Unmarshal(definition.Parameters, &schema); err != nil {
			return responses.ResponseNewParams{}, fmt.Errorf("decode tool %q schema: %w", definition.Name, err)
		}
		params.Tools = append(params.Tools, responses.ToolUnionParam{OfFunction: &responses.FunctionToolParam{
			Name: definition.Name, Description: param.NewOpt(definition.Description), Parameters: schema, Strict: param.NewOpt(false),
		}})
		available[definition.Name] = struct{}{}
	}
	if len(req.Tools) > 0 {
		switch req.ToolChoice.Mode {
		case "", ai.ToolChoiceAuto:
			params.ToolChoice.OfToolChoiceMode = param.NewOpt(responses.ToolChoiceOptionsAuto)
		case ai.ToolChoiceNone:
			params.ToolChoice.OfToolChoiceMode = param.NewOpt(responses.ToolChoiceOptionsNone)
		case ai.ToolChoiceRequired:
			if len(req.ToolChoice.Names) == 1 {
				if _, ok := available[req.ToolChoice.Names[0]]; !ok {
					return responses.ResponseNewParams{}, fmt.Errorf("required tool %q is not defined", req.ToolChoice.Names[0])
				}
				params.ToolChoice.OfFunctionTool = &responses.ToolChoiceFunctionParam{Name: req.ToolChoice.Names[0]}
				break
			}
			if len(req.ToolChoice.Names) > 1 {
				return responses.ResponseNewParams{}, fmt.Errorf("%w: Responses API supports one named function tool", ai.ErrUnsupportedCapability)
			}
			params.ToolChoice.OfToolChoiceMode = param.NewOpt(responses.ToolChoiceOptionsRequired)
		default:
			return responses.ResponseNewParams{}, fmt.Errorf("%w: unsupported OpenAI tool choice mode %q", ai.ErrUnsupportedCapability, req.ToolChoice.Mode)
		}
	}
	return params, nil
}

func mapResponsesMessages(messages []ai.Message) (responses.ResponseInputParam, error) {
	input := make(responses.ResponseInputParam, 0, len(messages))
	for _, message := range messages {
		if err := message.Validate(); err != nil {
			return nil, err
		}
		if err := rejectRequiredExtensions(message.Extensions); err != nil {
			return nil, err
		}
		seenSignatures := map[string]struct{}{}
		appendReasoning := func(data json.RawMessage) error {
			if _, seen := seenSignatures[string(data)]; seen {
				return nil
			}
			var items []responses.ResponseReasoningItem
			if err := json.Unmarshal(data, &items); err != nil {
				return fmt.Errorf("decode OpenAI reasoning items: %w", err)
			}
			for _, item := range items {
				if item.Type != "reasoning" {
					return fmt.Errorf("invalid OpenAI reasoning item type %q", item.Type)
				}
				params := item.ToParam()
				input = append(input, responses.ResponseInputItemUnionParam{OfReasoning: &params})
			}
			seenSignatures[string(data)] = struct{}{}
			return nil
		}
		for _, part := range message.Parts {
			if part.Kind != ai.ContentExtension {
				if err := rejectRequiredExtensions(part.Extensions); err != nil {
					return nil, err
				}
			}
			switch part.Kind {
			case ai.ContentText, ai.ContentJSON:
				text := part.Text
				if part.Kind == ai.ContentJSON {
					text = string(part.JSON)
				}
				role := responses.EasyInputMessageRole(message.Role)
				input = append(input, responses.ResponseInputItemParamOfMessage(text, role))
			case ai.ContentExtension:
				for _, ext := range part.Extensions {
					if ext.Namespace == "openai" && ext.Type == "responses_reasoning" {
						if err := appendReasoning(ext.Data); err != nil {
							return nil, err
						}
					} else if ext.Required {
						return nil, fmt.Errorf("%w: OpenAI Responses extension %s/%s", ai.ErrUnsupportedCapability, ext.Namespace, ext.Type)
					}
				}
			case ai.ContentToolCall:
				call := part.ToolCall
				for _, ext := range call.Extensions {
					if ext.Namespace == "openai" && ext.Type == "responses_reasoning" {
						if err := appendReasoning(ext.Data); err != nil {
							return nil, err
						}
					} else if ext.Required {
						return nil, fmt.Errorf("%w: OpenAI Responses call extension %s/%s", ai.ErrUnsupportedCapability, ext.Namespace, ext.Type)
					}
				}
				if len(call.ThoughtSignature) > 0 {
					if err := appendReasoning(call.ThoughtSignature); err != nil {
						return nil, err
					}
				}
				input = append(input, responses.ResponseInputItemParamOfFunctionCall(string(call.Args), call.ID, call.Name))
			case ai.ContentToolResult:
				content, err := canonicalResultText(part.ToolResult)
				if err != nil {
					return nil, err
				}
				input = append(input, responses.ResponseInputItemParamOfFunctionCallOutput(part.ToolResult.ToolCallID, content))
			default:
				return nil, fmt.Errorf("%w: OpenAI Responses content %q", ai.ErrUnsupportedCapability, part.Kind)
			}
		}
	}
	return input, nil
}

func reasoningExtension(raw json.RawMessage) ai.ContentPart {
	data, _ := json.Marshal([]json.RawMessage{raw})
	return ai.ContentPart{Kind: ai.ContentExtension, Extensions: []ai.Extension{{Namespace: "openai", Type: "responses_reasoning", Data: data, Required: true}}}
}

func responseFromResponses(response *responses.Response) (*ai.AIResponse, error) {
	result := &ai.AIResponse{
		Raw: json.RawMessage(response.RawJSON()), FinishReason: string(response.Status),
		InputTokens: int(response.Usage.InputTokens), OutputTokens: int(response.Usage.OutputTokens), ReasoningTokens: int(response.Usage.OutputTokensDetails.ReasoningTokens),
	}
	message := ai.Message{Role: ai.RoleAssistant}
	for _, output := range response.Output {
		switch output.Type {
		case "reasoning":
			message.Parts = append(message.Parts, reasoningExtension(json.RawMessage(output.RawJSON())))
		case "message":
			for _, content := range output.Content {
				switch content.Type {
				case "output_text":
					message.Parts = append(message.Parts, ai.ContentPart{Kind: ai.ContentText, Text: content.Text})
				case "refusal":
					message.Parts = append(message.Parts, ai.ContentPart{Kind: ai.ContentText, Text: content.Refusal})
				default:
					return nil, fmt.Errorf("%w: OpenAI Responses output content %q", ai.ErrUnsupportedCapability, content.Type)
				}
			}
		case "function_call":
			args := json.RawMessage(output.Arguments)
			if !json.Valid(args) {
				return nil, fmt.Errorf("invalid JSON arguments for tool %q", output.Name)
			}
			message.Parts = append(message.Parts, ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: output.CallID, Type: "function", Name: output.Name, Args: args}})
		default:
			return nil, fmt.Errorf("%w: OpenAI Responses output %q", ai.ErrUnsupportedCapability, output.Type)
		}
	}
	if len(message.Parts) == 0 {
		message.Parts = ai.TextParts("")
	}
	result.SetMessage(message)
	return result, nil
}
