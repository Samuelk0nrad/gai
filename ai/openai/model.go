package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	sdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

func classifyProviderError(err error) error {
	var apiErr *sdk.Error
	if !errors.As(err, &apiErr) || apiErr == nil {
		return err
	}
	var header http.Header
	if apiErr.Response != nil {
		header = apiErr.Response.Header
	}
	return ai.ClassifyProviderError(err, apiErr.StatusCode, apiErr.Code, header.Get("x-request-id"), header)
}

// Model is an OpenAI chat-completions model.
type Model struct {
	name           string
	provider       *Provider
	chatHooks      []func(*sdk.ChatCompletionNewParams) error
	responsesHooks []func(*responses.ResponseNewParams) error
}

var _ ai.Model = (*Model)(nil)
var _ ai.ModelDescriber = (*Model)(nil)

func (m *Model) Name() string { return m.name }

func (m *Model) NativeTools() bool { return true }

func (m *Model) Close() error { return nil }

// Tokenizer returns a local tokenizer only when this model has a known encoding.
// Unknown models remain nil so callers retain their configured fallback behavior.
func (m *Model) Tokenizer() ai.Tokenizer {
	tokenizer, err := NewTokenizer(m.name)
	if err != nil {
		return nil
	}
	return tokenizer
}

func (m *Model) Descriptor() ai.ModelDescriptor {
	if facts, ok := m.provider.catalog.Lookup(m.name); ok {
		return effectiveOpenAIDescriptor(m.name, facts)
	}
	return openAIDescriptor(m.name)
}

func (m *Model) Generate(ctx context.Context, req ai.AIRequest) (result *ai.AIResponse, err error) {
	if err := ai.ValidateModelRequest(m, req); err != nil {
		return nil, err
	}
	if err := m.validateTransport(req); err != nil {
		return nil, err
	}
	if m.provider.transport == TransportResponses {
		return m.generateResponses(ctx, req)
	}
	params, err := m.chatCompletionParams(req, false)
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
	response, err := m.client(false).Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, err
	}
	result = &ai.AIResponse{
		Raw:             json.RawMessage(response.RawJSON()),
		InputTokens:     int(response.Usage.PromptTokens),
		OutputTokens:    int(response.Usage.CompletionTokens),
		ReasoningTokens: int(response.Usage.CompletionTokensDetails.ReasoningTokens),
	}
	generationResult.ResponseModel = response.Model
	generationResult.RequestID = response.ID
	if response.JSON.Usage.Valid() {
		usage := ai.Usage{
			InputTokens: result.InputTokens, OutputTokens: result.OutputTokens,
			ReasoningTokens: result.ReasoningTokens, CachedTokens: int(response.Usage.PromptTokensDetails.CachedTokens),
		}
		generationResult.Usage = &usage
	}
	if len(response.Choices) == 0 {
		return result, nil
	}
	message := response.Choices[0].Message
	generationResult.FinishReason = response.Choices[0].FinishReason
	result.Text = message.Content
	for _, call := range message.ToolCalls {
		args := json.RawMessage(strings.TrimSpace(call.Function.Arguments))
		if len(args) == 0 {
			args = json.RawMessage("{}")
		}
		if !json.Valid(args) {
			return nil, fmt.Errorf("invalid JSON arguments for tool %q", call.Function.Name)
		}
		result.ToolCalls = append(result.ToolCalls, ai.ToolCall{
			ID: call.ID, Type: "function", Name: call.Function.Name, Args: args,
		})
	}
	generationResult.ToolCallCount = len(result.ToolCalls)
	return result, nil
}

func (m *Model) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)
	go func() {
		defer close(out)
		if err := ai.ValidateModelRequest(m, req); err != nil {
			ai.SendToken(ctx, out, ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		}
		if err := m.validateTransport(req); err != nil {
			ai.SendToken(ctx, out, ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		}
		if m.provider.transport == TransportResponses {
			m.generateResponsesStream(ctx, out, req)
			return
		}
		params, err := m.chatCompletionParams(req, true)
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
		stream := m.client(true).Chat.Completions.NewStreaming(ctx, params)
		defer func() {
			if err := stream.Close(); err != nil && m.provider.debug != nil {
				gai.EmitObservation(ctx, m.provider.debug, gai.Observation{
					Name:   "stream_close_failed",
					Source: "ai:openai.Model.GenerateStream",
					Err:    err,
				})
			}
		}()
		calls := map[int64]*streamToolCall{}
		completion := ai.Completion{Provider: "openai"}
		defer func() { mergeOpenAICompletionResult(&generationResult, completion) }()
		for stream.Next() {
			chunk := stream.Current()
			if chunk.ID != "" {
				completion.RequestID = chunk.ID
			}
			if chunk.Model != "" {
				completion.Model = chunk.Model
			}
			if len(chunk.Choices) == 0 {
				if chunk.JSON.Usage.Valid() {
					completion.UsageReported = true
					completion.Usage = ai.Usage{
						InputTokens:     int(chunk.Usage.PromptTokens),
						OutputTokens:    int(chunk.Usage.CompletionTokens),
						ReasoningTokens: int(chunk.Usage.CompletionTokensDetails.ReasoningTokens),
						CachedTokens:    int(chunk.Usage.PromptTokensDetails.CachedTokens),
					}
					completion.Raw = append(completion.Raw[:0], []byte(chunk.RawJSON())...)
					snapshot := completion
					snapshot.Raw = append(json.RawMessage(nil), completion.Raw...)
					if !emit(ai.Token{Type: ai.TokenTypeCompletion, Completion: &snapshot}) {
						return
					}
				}
				continue
			}
			choice := chunk.Choices[0]
			if choice.FinishReason != "" {
				completion.FinishReason = string(choice.FinishReason)
			}
			if text := choice.Delta.Content; text != "" {
				if !emit(ai.Token{Type: ai.TokenTypeText, Data: []byte(text), Text: text}) {
					return
				}
			}
			for _, delta := range choice.Delta.ToolCalls {
				call := calls[delta.Index]
				if call == nil {
					call = &streamToolCall{}
					calls[delta.Index] = call
				}
				call.id.WriteString(delta.ID)
				call.typ = firstNonEmpty(call.typ, delta.Type)
				call.name.WriteString(delta.Function.Name)
				call.arguments.WriteString(delta.Function.Arguments)
			}
			if choice.FinishReason == "tool_calls" {
				if !sendStreamToolCalls(emit, calls) {
					return
				}
			}
		}
		if err := stream.Err(); err != nil {
			streamErr = classifyProviderError(err)
			emit(ai.Token{Type: ai.TokenTypeErr, Err: streamErr, Text: streamErr.Error()})
			return
		}
		sendStreamToolCalls(emit, calls)
	}()
	return out
}

func mergeOpenAICompletionResult(result *ai.GenerationResult, completion ai.Completion) {
	if completion.Model != "" {
		result.ResponseModel = completion.Model
	}
	if completion.RequestID != "" {
		result.RequestID = completion.RequestID
	}
	if completion.FinishReason != "" {
		result.FinishReason = completion.FinishReason
	}
	if completion.UsageReported {
		usage := completion.Usage
		result.Usage = &usage
	}
}

func openAIHTTPStatus(err error) int {
	var apiErr *sdk.Error
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode
	}
	return 0
}

type streamToolCall struct {
	id, name, arguments strings.Builder
	typ                 string
	emitted             bool
}

func sendStreamToolCalls(emit func(ai.Token) bool, calls map[int64]*streamToolCall) bool {
	indices := make([]int64, 0, len(calls))
	for index := range calls {
		indices = append(indices, index)
	}
	sort.Slice(indices, func(i, j int) bool { return indices[i] < indices[j] })
	for _, index := range indices {
		call := calls[index]
		if call.emitted {
			continue
		}
		call.emitted = true
		name := call.name.String()
		if strings.TrimSpace(name) == "" {
			continue
		}
		args := call.arguments.String()
		if !json.Valid([]byte(args)) {
			err := fmt.Errorf("invalid JSON arguments for tool %q", name)
			emit(ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return false
		}
		toolCall := &ai.ToolCall{ID: call.id.String(), Type: firstNonEmpty(call.typ, "function"), Name: name, Args: json.RawMessage(args)}
		if !emit(ai.Token{Type: ai.TokenTypeToolCall, Data: []byte(args), ToolCall: toolCall}) {
			return false
		}
	}
	return true
}

func buildChatCompletionParams(model string, req ai.AIRequest, stream bool) (sdk.ChatCompletionNewParams, error) {
	params := sdk.ChatCompletionNewParams{Model: shared.ChatModel(model), Messages: []sdk.ChatCompletionMessageParamUnion{sdk.UserMessage(req.Prompt)}}
	if len(req.Messages) > 0 {
		messages, err := mapNativeMessages(req.Messages)
		if err != nil {
			return sdk.ChatCompletionNewParams{}, err
		}
		params.Messages = messages
	}
	if req.MaxTokens > 0 {
		params.MaxCompletionTokens = param.NewOpt(int64(req.MaxTokens))
	}
	if stream {
		params.StreamOptions.IncludeUsage = param.NewOpt(true)
	}
	if err := applyTools(&params, req.Tools, req.ToolChoice); err != nil {
		return sdk.ChatCompletionNewParams{}, err
	}
	if err := applyResponseFormat(&params, req.ResponseFormat); err != nil {
		return sdk.ChatCompletionNewParams{}, err
	}
	switch req.Reasoning.Effort {
	case "":
	case ai.ReasoningEffortNone, ai.ReasoningEffortMinimal, ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh, ai.ReasoningEffortXHigh, ai.ReasoningEffortMax:
		params.ReasoningEffort = shared.ReasoningEffort(req.Reasoning.Effort)
	default:
		return sdk.ChatCompletionNewParams{}, fmt.Errorf("%w: OpenAI reasoning effort %q", ai.ErrUnsupportedCapability, req.Reasoning.Effort)
	}
	return params, nil
}

func mapNativeMessages(messages []ai.RequestMessage) ([]sdk.ChatCompletionMessageParamUnion, error) {
	out := make([]sdk.ChatCompletionMessageParamUnion, 0, len(messages))
	for _, m := range messages {
		if m.Role == ai.RequestMessageRoleUser {
			out = append(out, sdk.UserMessage(m.Text))
			continue
		}
		if m.Role == ai.RequestMessageRoleTool {
			content := m.ToolResult.Content
			if m.ToolResult.IsError {
				encoded, _ := json.Marshal(map[string]string{"error": content})
				content = string(encoded)
			}
			out = append(out, sdk.ToolMessage(content, m.ToolResult.ToolCallID))
			continue
		}
		a := sdk.ChatCompletionAssistantMessageParam{}
		if m.Text != "" {
			a.Content.OfString = param.NewOpt(m.Text)
		}
		for _, c := range m.ToolCalls {
			a.ToolCalls = append(a.ToolCalls, sdk.ChatCompletionMessageToolCallParam{ID: c.ID, Function: sdk.ChatCompletionMessageToolCallFunctionParam{Name: c.Name, Arguments: string(c.Arguments)}})
		}
		out = append(out, sdk.ChatCompletionMessageParamUnion{OfAssistant: &a})
	}
	return out, nil
}

func openAIDescriptor(model string) ai.ModelDescriptor {
	d := ai.ModelDescriptor{
		Model: model, NativeMessages: ai.FeatureSupportSupported, NativeTools: ai.FeatureSupportSupported,
		ToolChoiceModes: []ai.ToolChoiceMode{ai.ToolChoiceAuto, ai.ToolChoiceNone, ai.ToolChoiceRequired},
		Usage:           ai.FeatureSupportSupported, FinishReason: ai.FeatureSupportSupported, StreamingUsage: ai.FeatureSupportSupported,
		JSONOutput: ai.FeatureSupportSupported, JSONSchemaOutput: ai.FeatureSupportSupported,
	}
	if efforts := gpt5ReasoningEfforts(model); len(efforts) > 0 {
		d.ReasoningEffort = ai.FeatureSupportSupported
		d.ReasoningEfforts = efforts
	} else if isGPT5ReasoningFamily(model) {
		// Unknown GPT-5 revisions must not inherit another revision's effort set.
	} else if isOpenAIReasoningFamily(model) {
		d.ReasoningEffort = ai.FeatureSupportSupported
		d.ReasoningEfforts = []ai.ReasoningEffort{ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh}
	} else if isKnownNonReasoningModel(model) {
		d.Reasoning, d.ReasoningEffort = ai.FeatureSupportUnsupported, ai.FeatureSupportUnsupported
	}
	return d
}

// validateTransport rejects known endpoint/model combinations before opening an
// HTTP stream. Responses transport is the supported path for these requests.
func (m *Model) validateTransport(req ai.AIRequest) error {
	if m.provider.transport == TransportChatCompletions && isGPT5ReasoningFamily(m.name) && len(req.Tools) > 0 && req.Reasoning.Effort != "" {
		return fmt.Errorf("%w: OpenAI GPT-5 models with function tools and an explicit reasoning effort require the Responses transport", ai.ErrUnsupportedCapability)
	}
	return nil
}

func isGPT5ReasoningFamily(model string) bool {
	name := strings.ToLower(strings.TrimSpace(model))
	return strings.HasPrefix(name, "gpt-5") && (len(name) == len("gpt-5") || name[len("gpt-5")] == '.' || name[len("gpt-5")] == '-')
}

func gpt5ReasoningEfforts(model string) []ai.ReasoningEffort {
	name := strings.ToLower(strings.TrimSpace(model))
	switch {
	case name == "gpt-5" || strings.HasPrefix(name, "gpt-5-"):
		return []ai.ReasoningEffort{ai.ReasoningEffortMinimal, ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh}
	case name == "gpt-5.1" || strings.HasPrefix(name, "gpt-5.1-"):
		return []ai.ReasoningEffort{ai.ReasoningEffortNone, ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh}
	case name == "gpt-5.6" || strings.HasPrefix(name, "gpt-5.6-"):
		return []ai.ReasoningEffort{ai.ReasoningEffortNone, ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh, ai.ReasoningEffortXHigh, ai.ReasoningEffortMax}
	default:
		return nil
	}
}

func isOpenAIReasoningFamily(model string) bool {
	if isGPT5ReasoningFamily(model) {
		return true
	}
	name := strings.ToLower(strings.TrimSpace(model))
	if len(name) < 2 || name[0] != 'o' {
		return false
	}
	i := 1
	for i < len(name) && name[i] >= '0' && name[i] <= '9' {
		i++
	}
	return i > 1 && (i == len(name) || name[i] == '.' || name[i] == '-')
}

func isKnownNonReasoningModel(model string) bool {
	switch model {
	case GPT41, GPT41Mini, GPT41Nano, GPT4o, GPT4oMini:
		return true
	default:
		return false
	}
}

func applyTools(params *sdk.ChatCompletionNewParams, definitions []ai.ToolDefinition, choice ai.ToolChoice) error {
	if len(definitions) == 0 {
		return nil
	}
	params.Tools = make([]sdk.ChatCompletionToolParam, 0, len(definitions))
	available := make(map[string]struct{}, len(definitions))
	for _, definition := range definitions {
		if err := definition.Validate(); err != nil {
			return err
		}
		var schema shared.FunctionParameters
		if err := json.Unmarshal(definition.Parameters, &schema); err != nil {
			return fmt.Errorf("decode tool %q schema: %w", definition.Name, err)
		}
		params.Tools = append(params.Tools, sdk.ChatCompletionToolParam{Function: shared.FunctionDefinitionParam{Name: definition.Name, Description: param.NewOpt(definition.Description), Parameters: schema}})
		available[definition.Name] = struct{}{}
	}
	if choice.Mode == ai.ToolChoiceRequired && len(choice.Names) > 0 {
		allowed := make(map[string]struct{}, len(choice.Names))
		for _, name := range choice.Names {
			if _, ok := available[name]; !ok {
				return fmt.Errorf("required tool %q is not defined", name)
			}
			allowed[name] = struct{}{}
		}
		if len(choice.Names) > 1 {
			filtered := make([]sdk.ChatCompletionToolParam, 0, len(allowed))
			for _, tool := range params.Tools {
				if _, ok := allowed[tool.Function.Name]; ok {
					filtered = append(filtered, tool)
				}
			}
			params.Tools = filtered
		}
	}
	switch choice.Mode {
	case "", ai.ToolChoiceAuto, ai.ToolChoiceNone, ai.ToolChoiceRequired:
		mode := string(choice.Mode)
		if mode == "" {
			mode = string(ai.ToolChoiceAuto)
		}
		params.ToolChoice.OfAuto = param.NewOpt(mode)
		if choice.Mode == ai.ToolChoiceRequired && len(choice.Names) == 1 {
			params.ToolChoice = sdk.ChatCompletionToolChoiceOptionParamOfChatCompletionNamedToolChoice(sdk.ChatCompletionNamedToolChoiceFunctionParam{Name: choice.Names[0]})
		}
		return nil
	default:
		return fmt.Errorf("unsupported openai tool choice mode %q", choice.Mode)
	}
}

func applyResponseFormat(params *sdk.ChatCompletionNewParams, format ai.ResponseFormat) error {
	switch format.Type {
	case "", ai.ResponseFormatText:
		return nil
	case ai.ResponseFormatJSONObject:
		params.ResponseFormat.OfJSONObject = &shared.ResponseFormatJSONObjectParam{}
		return nil
	case ai.ResponseFormatJSONSchema:
		var schema any
		if err := json.Unmarshal(format.Schema, &schema); err != nil {
			return err
		}
		params.ResponseFormat.OfJSONSchema = &shared.ResponseFormatJSONSchemaParam{JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{Name: format.Name, Schema: schema}}
		return nil
	default:
		return fmt.Errorf("%w: %s", ai.ErrInvalidResponseFormat, format.Type)
	}
}

func (m *Model) client(streaming bool) *sdk.Client { return m.provider.sdkClient(streaming) }

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
