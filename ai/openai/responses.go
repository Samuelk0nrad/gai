package openai

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/streamtext"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

func (m *Model) generateResponses(ctx context.Context, req ai.AIRequest) (result *ai.AIResponse, err error) {
	params, err := m.responsesParams(req)
	if err != nil {
		return nil, err
	}
	ctx, observation := ai.StartGenerationObservation(ctx, req, ai.GenerationConfig{Provider: "openai", Model: m.name, Sink: m.provider.debug})
	generationResult := ai.GenerationResult{}
	defer func() {
		generationResult.Err = err
		generationResult.HTTPStatus = openAIHTTPStatus(err)
		if result != nil {
			generationResult.Output = &result.Message
		}
		observation.Finish(generationResult)
	}()
	response, err := m.client(false).Responses.New(ctx, params)
	if err != nil {
		return nil, err
	}
	if response.Error.Message != "" {
		return nil, fmt.Errorf("OpenAI Responses API: %s", response.Error.Message)
	}
	generationResult.ResponseModel = string(response.Model)
	generationResult.RequestID = response.ID
	generationResult.FinishReason = string(response.Status)
	if response.JSON.Usage.Valid() {
		usage := responsesUsage(response)
		generationResult.Usage = &usage
	}
	if response.Status == "failed" {
		message := string(response.Error.Code)
		if message == "" {
			message = "response failed"
		}
		return nil, fmt.Errorf("OpenAI Responses API: %s", message)
	}
	if response.Status == "incomplete" {
		return nil, responseIncompleteError(response)
	}
	result, err = responseFromResponses(response)
	if result != nil {
		generationResult.ToolCallCount = len(result.ToolCalls())
	}
	return result, err
}

func (m *Model) generateResponsesStream(ctx context.Context, out chan<- ai.Token, req ai.AIRequest) {
	params, err := m.responsesParams(req)
	if err != nil {
		ai.SendToken(ctx, out, ai.Token{Err: err})
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
	// run follows the text part the caller builds, so citations from a later
	// output_text content keep offsets into the merged text.
	var run streamtext.Run
	emit := func(token ai.Token) bool {
		if token.Type() == ai.TokenTypeErr && token.Err != nil {
			streamErr = token.Err
		}
		run.Observe(token)
		observation.ObserveToken(token)
		if ai.SendToken(ctx, out, token) {
			observation.Delivered(token)
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
			if text := event.Delta.OfString; text != "" && !emit(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: text}}) {
				return
			}
		case "response.completed":
			response := event.Response
			completion := completionFromResponses(&response)
			if !emit(ai.Token{Completion: &completion}) {
				return
			}
		case "response.incomplete":
			response := event.Response
			completion := completionFromResponses(&response)
			if !emit(ai.Token{Completion: &completion}) {
				return
			}
			err := responseIncompleteError(&response)
			streamErr = err
			emit(ai.Token{Err: err})
			return
		case "response.output_item.done":
			if event.Item.Type == "reasoning" {
				part := reasoningExtension(json.RawMessage(event.Item.RawJSON()))
				if !emit(ai.Token{Part: &part}) {
					return
				}
				continue
			}
			if isHostedToolItem(event.Item.Type) {
				part := hostedToolPart(json.RawMessage(event.Item.RawJSON()))
				if !emit(ai.Token{Part: &part}) {
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
				emit(ai.Token{Err: err})
				return
			}
			call := &ai.ToolCall{ID: event.Item.CallID, Type: "function", Name: event.Item.Name, Args: args}
			if !emit(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: call}}) {
				return
			}
		case "response.content_part.done":
			// The annotations arrive complete with their content part, after
			// its text deltas. An empty text token with extensions attaches
			// them to the text part those deltas built, which may begin with
			// the text of earlier contents.
			if event.Part.Type != "output_text" {
				continue
			}
			citations := annotationCitations(event.Part.JSON.Annotations.Raw(), event.Part.Text)
			if token, ok := run.AttachCitations(extensionNamespace, citations, ai.RuneLen(event.Part.Text)); ok && !emit(token) {
				return
			}
		case "error":
			err := fmt.Errorf("OpenAI Responses API: %s", event.Message)
			streamErr = err
			emit(ai.Token{Err: err})
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
			emit(ai.Token{Err: err})
			return
		}
	}
	if err := stream.Err(); err != nil {
		streamErr = classifyProviderError(err)
		emit(ai.Token{Err: streamErr})
	}
}

func responsesUsage(response *responses.Response) ai.Usage {
	return ai.Usage{
		InputTokens:     int(response.Usage.InputTokens),
		OutputTokens:    int(response.Usage.OutputTokens),
		ReasoningTokens: int(response.Usage.OutputTokensDetails.ReasoningTokens),
		CachedTokens:    int(response.Usage.InputTokensDetails.CachedTokens),
	}
}

func completionFromResponses(response *responses.Response) ai.Completion {
	completion := ai.Completion{
		Provider: "openai", Model: string(response.Model), RequestID: response.ID,
		FinishReason: string(response.Status), Raw: json.RawMessage(response.RawJSON()),
	}
	if response.JSON.Usage.Valid() {
		completion.UsageReported = true
		completion.Usage = responsesUsage(response)
	}
	return completion
}

func responseIncompleteError(response *responses.Response) error {
	reason := string(response.IncompleteDetails.Reason)
	if reason == "" {
		reason = "incomplete"
	}
	return &ai.TerminalError{Provider: "openai", Reason: reason}
}

func buildResponsesParams(model string, req ai.AIRequest) (responses.ResponseNewParams, error) {
	params := responses.ResponseNewParams{
		Model:   model,
		Store:   param.NewOpt(false),
		Include: []responses.ResponseIncludable{responses.ResponseIncludableReasoningEncryptedContent},
	}
	req = req.Copy()
	err := req.Validate()
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
		appendReasoning := func(data json.RawMessage) error {
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
			return nil
		}
		// A user message with media sends each run of text, JSON and media
		// as one message item with a content list. Text-only messages keep
		// one string item per part.
		hasMedia := partsContainMedia(message.Parts)
		if hasMedia && message.Role != ai.RoleUser {
			return nil, unsupportedMedia("media is only accepted in user messages and tool results, not %s messages", message.Role)
		}
		var content responses.ResponseInputMessageContentListParam
		flushContent := func() {
			if len(content) > 0 {
				input = append(input, responses.ResponseInputItemParamOfMessage(content, responses.EasyInputMessageRoleUser))
				content = nil
			}
		}
		for _, part := range message.Parts {
			if part.Kind != ai.ContentExtension {
				if err := rejectRequiredExtensions(part.Extensions); err != nil {
					return nil, err
				}
			}
			if hasMedia {
				switch part.Kind {
				case ai.ContentText:
					content = append(content, responses.ResponseInputContentParamOfInputText(part.Text))
					continue
				case ai.ContentJSON:
					content = append(content, responses.ResponseInputContentParamOfInputText(string(part.JSON)))
					continue
				case ai.ContentMedia:
					item, err := mediaInputContent(part.Media)
					if err != nil {
						return nil, err
					}
					content = append(content, item)
					continue
				default:
					flushContent()
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
					if ext.Namespace == extensionNamespace && ext.Type == "responses_reasoning" {
						if err := appendReasoning(ext.Data); err != nil {
							return nil, err
						}
					} else if item, ok, err := hostedToolExtensionInput(ext); ok {
						if err != nil {
							return nil, err
						}
						input = append(input, item)
					} else if ext.Required {
						return nil, fmt.Errorf("%w: OpenAI Responses extension %s/%s", ai.ErrUnsupportedCapability, ext.Namespace, ext.Type)
					}
				}
			case ai.ContentToolCall:
				call := part.ToolCall
				if err := rejectRequiredExtensions(call.Extensions); err != nil {
					return nil, err
				}
				input = append(input, responses.ResponseInputItemParamOfFunctionCall(string(call.Args), call.ID, call.Name))
			case ai.ContentToolResult:
				if partsContainMedia(part.ToolResult.Parts) {
					output, err := toolResultContent(part.ToolResult)
					if err != nil {
						return nil, err
					}
					input = append(input, functionCallOutputWithContent(part.ToolResult.ToolCallID, output))
					continue
				}
				output, err := canonicalResultText(part.ToolResult)
				if err != nil {
					return nil, err
				}
				input = append(input, responses.ResponseInputItemParamOfFunctionCallOutput(part.ToolResult.ToolCallID, output))
			default:
				return nil, fmt.Errorf("%w: OpenAI Responses content %q", ai.ErrUnsupportedCapability, part.Kind)
			}
		}
		flushContent()
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
					message.Parts = append(message.Parts, ai.ContentPart{Kind: ai.ContentText, Text: content.Text, Extensions: ai.CitationsExtension(extensionNamespace, annotationCitations(content.JSON.Annotations.Raw(), content.Text))})
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
			if isHostedToolItem(output.Type) {
				message.Parts = append(message.Parts, hostedToolPart(json.RawMessage(output.RawJSON())))
				continue
			}
			return nil, fmt.Errorf("%w: OpenAI Responses output %q", ai.ErrUnsupportedCapability, output.Type)
		}
	}
	if len(message.Parts) == 0 {
		message.Parts = ai.TextParts("")
	}
	result.SetMessage(message)
	return result, nil
}
