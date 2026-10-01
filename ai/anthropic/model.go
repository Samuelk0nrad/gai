package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	antropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/internal/modelcatalog"
	"go.opentelemetry.io/otel/attribute"
)

const (
	anthropicTracerName = "github.com/lace-ai/gai/ai/anthropic"
	defaultMaxTokens    = 4096
	minThinkingTokens   = 1024
)

type Model struct {
	name         string
	client       *Provider
	debug        gai.ObservationSink
	messageHooks []func(*antropic.MessageNewParams) error
}

var _ ai.Model = (*Model)(nil)
var _ ai.ModelDescriber = (*Model)(nil)

func (m *Model) Name() string      { return m.name }
func (m *Model) NativeTools() bool { return true }
func (m *Model) Close() error      { return nil }

// TokenCounter uses a local estimate for automatic budgeting. Tokenizer is
// still available explicitly, but may perform network I/O.
func (m *Model) TokenCounter() ai.TokenCounter { return ai.TextTokenEstimator{} }

func (m *Model) Tokenizer() ai.Tokenizer {
	return &Tokenizer{modelName: m.name, client: m.client, debug: m.debug}
}

func (m *Model) Descriptor() ai.ModelDescriptor {
	if facts, ok := m.client.catalog.Lookup(m.name); ok {
		return effectiveAnthropicDescriptor(m.name, facts)
	}
	return effectiveAnthropicDescriptor(m.name, ai.ModelDescriptor{Model: m.name})
}

func anthropicAdapterDescriptor(model string) ai.ModelDescriptor {
	return ai.ModelDescriptor{Model: model, NativeMessages: ai.FeatureSupportSupported, NativeTools: ai.FeatureSupportSupported,
		ToolChoiceModes: []ai.ToolChoiceMode{ai.ToolChoiceAuto, ai.ToolChoiceNone, ai.ToolChoiceRequired},
		Usage:           ai.FeatureSupportSupported, FinishReason: ai.FeatureSupportSupported, StreamingUsage: ai.FeatureSupportSupported,
		JSONOutput: ai.FeatureSupportUnsupported, JSONSchemaOutput: ai.FeatureSupportSupported,
		Reasoning: ai.FeatureSupportSupported, ReasoningEffort: ai.FeatureSupportSupported,
		ReasoningEfforts: []ai.ReasoningEffort{ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh},
	}
}

func anthropicLocalFacts(model string) ai.ModelDescriptor {
	d := ai.ModelDescriptor{Model: model}
	if supportsAdaptiveThinking(model) {
		d.Reasoning = ai.FeatureSupportSupported
		d.ReasoningEffort = ai.FeatureSupportSupported
		d.ReasoningEfforts = []ai.ReasoningEffort{ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh}
	} else if isKnownModel(model) {
		d.Reasoning = ai.FeatureSupportSupported
		d.ReasoningEffort = ai.FeatureSupportUnsupported
	}
	return d
}

func anthropicProviderDefaults(model string) ai.ModelDescriptor {
	d := anthropicAdapterDescriptor(model)
	d.Reasoning = ai.FeatureSupportUnknown
	d.ReasoningEffort = ai.FeatureSupportUnknown
	return d
}

func effectiveAnthropicDescriptor(model string, catalog ai.ModelDescriptor) ai.ModelDescriptor {
	adapter := anthropicAdapterDescriptor(model)
	facts := anthropicProviderDefaults(model)
	facts = modelcatalog.OverrideModelDescriptor(facts, anthropicLocalFacts(model))
	facts = modelcatalog.OverrideModelDescriptor(facts, catalog)
	return modelcatalog.IntersectModelDescriptors(adapter, facts)
}

// sdkClient is deliberately created from the provider's fields for each call.
// Those fields are package-level test seams and callers must never inherit SDK
// environment defaults or retries.
func (p *Provider) sdkClient() antropic.Client {
	return antropic.NewClient(
		option.WithoutEnvironmentDefaults(),
		option.WithAPIKey(p.apiKey),
		option.WithBaseURL(p.baseURL),
		option.WithHTTPClient(p.httpClient),
		option.WithMaxRetries(0),
	)
}

func buildMessagesRequest(req ai.AIRequest, descriptor ai.ModelDescriptor) (antropic.MessageNewParams, error) {
	if err := req.ValidateMessages(); err != nil {
		return antropic.MessageNewParams{}, err
	}
	if err := req.ResponseFormat.Validate(); err != nil {
		return antropic.MessageNewParams{}, err
	}
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	p := antropic.MessageNewParams{
		Model:     antropic.Model(descriptor.Model),
		MaxTokens: int64(maxTokens),
		Messages:  []antropic.MessageParam{antropic.NewUserMessage(antropic.NewTextBlock(req.Prompt))},
	}
	if len(req.Messages) > 0 {
		msgs, err := mapNativeMessages(req.Messages)
		if err != nil {
			return antropic.MessageNewParams{}, err
		}
		p.Messages = msgs
	}
	if len(req.Tools) > 0 {
		tools, err := mapTools(req.Tools)
		if err != nil {
			return antropic.MessageNewParams{}, err
		}
		p.Tools = tools
		choice, err := mapToolChoice(req.ToolChoice)
		if err != nil {
			return antropic.MessageNewParams{}, err
		}
		p.ToolChoice = choice
	}
	format, err := mapResponseFormat(req.ResponseFormat)
	if err != nil {
		return antropic.MessageNewParams{}, err
	}
	if format != nil {
		p.OutputConfig = *format
	}
	if req.Reasoning.Effort != "" {
		if descriptor.ReasoningEffort == ai.FeatureSupportUnsupported {
			return antropic.MessageNewParams{}, fmt.Errorf("%w: anthropic reasoning effort is unsupported by %s", ai.ErrUnsupportedCapability, descriptor.Model)
		}
		switch req.Reasoning.Effort {
		case ai.ReasoningEffortLow:
			p.OutputConfig.Effort = antropic.OutputConfigEffortLow
		case ai.ReasoningEffortMedium:
			p.OutputConfig.Effort = antropic.OutputConfigEffortMedium
		case ai.ReasoningEffortHigh:
			p.OutputConfig.Effort = antropic.OutputConfigEffortHigh
		default:
			return antropic.MessageNewParams{}, fmt.Errorf("%w: %s", ai.ErrUnsupportedCapability, req.Reasoning.Effort)
		}
	}
	if req.Reasoning.Enabled || req.Reasoning.IncludeThoughts || req.Reasoning.BudgetTokens > 0 || req.Reasoning.Effort != "" {
		if req.Reasoning.BudgetTokens > 0 {
			if req.Reasoning.BudgetTokens < minThinkingTokens || req.Reasoning.BudgetTokens >= maxTokens {
				return antropic.MessageNewParams{}, fmt.Errorf("%w: anthropic thinking budget must be at least %d and less than max_tokens", ai.ErrUnsupportedCapability, minThinkingTokens)
			}
			thinking := antropic.ThinkingConfigParamOfEnabled(int64(req.Reasoning.BudgetTokens))
			if req.Reasoning.IncludeThoughts {
				thinking.OfEnabled.Display = antropic.ThinkingConfigEnabledDisplaySummarized
			} else {
				thinking.OfEnabled.Display = antropic.ThinkingConfigEnabledDisplayOmitted
			}
			p.Thinking = thinking
		} else if descriptor.ReasoningEffort != ai.FeatureSupportUnsupported {
			display := antropic.ThinkingConfigAdaptiveDisplayOmitted
			if req.Reasoning.IncludeThoughts {
				display = antropic.ThinkingConfigAdaptiveDisplaySummarized
			}
			p.Thinking = antropic.ThinkingConfigParamUnion{OfAdaptive: &antropic.ThinkingConfigAdaptiveParam{Display: display}}
		} else {
			return antropic.MessageNewParams{}, fmt.Errorf("%w: anthropic adaptive thinking is unsupported by %s", ai.ErrUnsupportedCapability, descriptor.Model)
		}
		if p.ToolChoice.OfAny != nil || p.ToolChoice.OfTool != nil {
			return antropic.MessageNewParams{}, fmt.Errorf("%w: anthropic thinking is incompatible with required tool choice", ai.ErrUnsupportedCapability)
		}
	}
	return p, nil
}

func mapNativeMessages(messages []ai.RequestMessage) ([]antropic.MessageParam, error) {
	out := make([]antropic.MessageParam, 0, len(messages))
	toolResults := []antropic.ContentBlockParamUnion{}
	flushToolResults := func() {
		if len(toolResults) == 0 {
			return
		}
		out = append(out, antropic.NewUserMessage(toolResults...))
		toolResults = nil
	}
	for _, m := range messages {
		if m.Role == ai.RequestMessageRoleTool {
			toolResults = append(toolResults, antropic.NewToolResultBlock(m.ToolResult.ToolCallID, m.ToolResult.Content, m.ToolResult.IsError))
			continue
		}
		flushToolResults()
		if m.Role == ai.RequestMessageRoleUser {
			out = append(out, antropic.NewUserMessage(antropic.NewTextBlock(m.Text)))
			continue
		}
		blocks := []antropic.ContentBlockParamUnion{}
		if m.Text != "" {
			blocks = append(blocks, antropic.NewTextBlock(m.Text))
		}
		for _, c := range m.ToolCalls {
			var input any
			if err := json.Unmarshal(c.Arguments, &input); err != nil {
				return nil, err
			}
			blocks = append(blocks, antropic.NewToolUseBlock(c.ID, input, c.Name))
		}
		out = append(out, antropic.NewAssistantMessage(blocks...))
	}
	flushToolResults()
	return out, nil
}

func mapTools(defs []ai.ToolDefinition) ([]antropic.ToolUnionParam, error) {
	tools := make([]antropic.ToolUnionParam, 0, len(defs))
	for _, def := range defs {
		if err := def.Validate(); err != nil {
			return nil, err
		}
		var schema map[string]any
		if err := json.Unmarshal(def.Parameters, &schema); err != nil {
			return nil, fmt.Errorf("decode anthropic tool schema: %w", err)
		}
		tool := antropic.ToolUnionParamOfTool(antropic.ToolInputSchemaParam{ExtraFields: schema}, def.Name)
		if def.Description != "" {
			tool.OfTool.Description = param.NewOpt(def.Description)
		}
		tools = append(tools, tool)
	}
	return tools, nil
}

func mapToolChoice(choice ai.ToolChoice) (antropic.ToolChoiceUnionParam, error) {
	names := choice.Names
	switch choice.Mode {
	case "", ai.ToolChoiceAuto:
		if len(names) != 0 {
			return antropic.ToolChoiceUnionParam{}, fmt.Errorf("%w: anthropic auto tool choice cannot restrict names", ai.ErrUnsupportedCapability)
		}
		return antropic.ToolChoiceUnionParam{OfAuto: &antropic.ToolChoiceAutoParam{}}, nil
	case ai.ToolChoiceNone:
		if len(names) != 0 {
			return antropic.ToolChoiceUnionParam{}, fmt.Errorf("%w: anthropic none tool choice cannot name tools", ai.ErrUnsupportedCapability)
		}
		none := antropic.NewToolChoiceNoneParam()
		return antropic.ToolChoiceUnionParam{OfNone: &none}, nil
	case ai.ToolChoiceRequired:
		if len(names) == 0 {
			return antropic.ToolChoiceUnionParam{OfAny: &antropic.ToolChoiceAnyParam{}}, nil
		}
		if len(names) == 1 && strings.TrimSpace(names[0]) != "" {
			return antropic.ToolChoiceParamOfTool(strings.TrimSpace(names[0])), nil
		}
		return antropic.ToolChoiceUnionParam{}, fmt.Errorf("%w: anthropic required tool choice accepts at most one named tool", ai.ErrUnsupportedCapability)
	default:
		return antropic.ToolChoiceUnionParam{}, fmt.Errorf("unsupported anthropic tool choice mode %q", choice.Mode)
	}
}

func mapResponseFormat(format ai.ResponseFormat) (*antropic.OutputConfigParam, error) {
	switch format.Type {
	case "", ai.ResponseFormatText:
		return nil, nil
	case ai.ResponseFormatJSONObject:
		return nil, fmt.Errorf("%w: anthropic JSON object responses require a JSON schema", ai.ErrUnsupportedCapability)
	case ai.ResponseFormatJSONSchema:
		var schema map[string]any
		if err := json.Unmarshal(format.Schema, &schema); err != nil {
			return nil, fmt.Errorf("decode anthropic response schema: %w", err)
		}
		return &antropic.OutputConfigParam{Format: antropic.JSONOutputFormatParam{Schema: schema}}, nil
	default:
		return nil, fmt.Errorf("%w: %s", ai.ErrInvalidResponseFormat, format.Type)
	}
}

func (m *Model) Generate(ctx context.Context, req ai.AIRequest) (response *ai.AIResponse, err error) {
	if err := ai.ValidateModelRequest(m, req); err != nil {
		return nil, err
	}
	payload, err := m.messageParams(req)
	if err != nil {
		return nil, err
	}
	client := m.client.sdkClient()
	ctx, observation := ai.StartGenerationObservation(ctx, req, ai.GenerationConfig{Provider: "anthropic", Model: m.name, Sink: m.debug})
	if gai.ObservationEnabled(ctx, m.debug) {
		fields := map[string]any{}
		gai.AddObservationContent(ctx, m.debug, fields, "prompt", gai.ContentKindPrompt, req.Prompt)
		gai.EmitObservation(ctx, m.debug, gai.Observation{Name: "anthropic_generate_request", Source: "ai:anthropic.Model.Generate", Fields: fields})
	}
	generationResult := ai.GenerationResult{}
	defer func() {
		generationResult.Err = err
		generationResult.HTTPStatus = anthropicHTTPStatus(err)
		observation.Finish(generationResult)
	}()
	message, err := client.Messages.New(ctx, payload)
	if err != nil {
		return nil, localError(err)
	}
	text, thinking, calls, err := mapMessageContent(message.Content)
	if err != nil {
		return nil, err
	}
	input := int(message.Usage.InputTokens + message.Usage.CacheCreationInputTokens + message.Usage.CacheReadInputTokens)
	output := int(message.Usage.OutputTokens)
	usage := ai.Usage{
		InputTokens: input, OutputTokens: output,
		ReasoningTokens: int(message.Usage.OutputTokensDetails.ThinkingTokens),
		CachedTokens:    int(message.Usage.CacheReadInputTokens), CacheCreationTokens: int(message.Usage.CacheCreationInputTokens),
	}
	generationResult.ResponseModel = string(message.Model)
	generationResult.RequestID = message.ID
	generationResult.FinishReason = string(message.StopReason)
	generationResult.ToolCallCount = len(calls)
	if message.JSON.Usage.Valid() {
		generationResult.Usage = &usage
	}
	if gai.ObservationEnabled(ctx, m.debug) {
		fields := map[string]any{"input_tokens": input, "output_tokens": output}
		gai.AddObservationContent(ctx, m.debug, fields, "response_text", gai.ContentKindCompletion, text)
		gai.AddObservationContent(ctx, m.debug, fields, "reasoning", gai.ContentKindReasoning, thinking)
		gai.EmitObservation(ctx, m.debug, gai.Observation{Name: "anthropic_generate_success", Source: "ai:anthropic.Model.Generate", Fields: fields})
	}
	return &ai.AIResponse{Text: text, Reasoning: thinking, ToolCalls: calls, Raw: json.RawMessage(message.RawJSON()), FinishReason: string(message.StopReason), InputTokens: input, OutputTokens: output, ReasoningTokens: usage.ReasoningTokens}, nil
}

func mapMessageContent(blocks []antropic.ContentBlockUnion) (string, string, []ai.ToolCall, error) {
	var text, thinking strings.Builder
	calls := make([]ai.ToolCall, 0)
	for i, block := range blocks {
		switch block.Type {
		case "text":
			text.WriteString(block.Text)
		case "thinking":
			thinking.WriteString(block.Thinking)
		case "tool_use":
			name := strings.TrimSpace(block.Name)
			if name == "" {
				return "", "", nil, fmt.Errorf("content[%d]: tool use name empty", i)
			}
			args := block.Input
			if len(args) == 0 || string(args) == "null" {
				args = json.RawMessage("{}")
			}
			if !json.Valid(args) {
				return "", "", nil, fmt.Errorf("content[%d]: tool use input is invalid JSON", i)
			}
			id := strings.TrimSpace(block.ID)
			if id == "" {
				id = ai.GenerateToolCallID(name)
			}
			calls = append(calls, ai.ToolCall{ID: id, Type: "function", Name: name, Args: append(json.RawMessage(nil), args...)})
		}
	}
	return text.String(), thinking.String(), calls, nil
}

func localError(err error) error {
	var sdkErr *antropic.Error
	if !errors.As(err, &sdkErr) {
		return err
	}
	message := strings.TrimSpace(sdkErr.RawJSON())
	var payload antropic.ErrorResponse
	if json.Unmarshal([]byte(sdkErr.RawJSON()), &payload) == nil && payload.Error.Message != "" {
		message = payload.Error.Message
	}
	return &Error{StatusCode: sdkErr.StatusCode, Type: string(sdkErr.Type()), Message: message}
}

func classifyProviderError(err error) error {
	var sdkErr *antropic.Error
	if !errors.As(err, &sdkErr) || sdkErr == nil {
		return err
	}
	var header http.Header
	if sdkErr.Response != nil {
		header = sdkErr.Response.Header
	}
	return ai.ClassifyProviderError(err, sdkErr.StatusCode, string(sdkErr.Type()), sdkErr.RequestID, header)
}

func (m *Model) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)
	go func() {
		ctx := ctx
		var streamErr error
		var observation *ai.GenerationObservation
		generationResult := ai.GenerationResult{}
		defer func() {
			generationResult.Err = streamErr
			generationResult.HTTPStatus = anthropicHTTPStatus(streamErr)
			observation.Finish(generationResult)
			close(out)
		}()
		emit := func(token ai.Token) bool {
			observation.ObserveToken(token)
			if ai.SendToken(ctx, out, token) {
				return true
			}
			streamErr = ctx.Err()
			return false
		}
		if err := ai.ValidateModelRequest(m, req); err != nil {
			streamErr = err
			emit(ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		}
		payload, err := m.messageParams(req)
		if err != nil {
			streamErr = err
			emit(ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		}
		client := m.client.sdkClient()
		generationCtx, startedObservation := ai.StartGenerationObservation(ctx, req, ai.GenerationConfig{Provider: "anthropic", Model: m.name, Streaming: true, Sink: m.debug})
		observation = startedObservation
		ctx = generationCtx
		stream := client.Messages.NewStreaming(generationCtx, payload)
		defer stream.Close()
		blocks := map[int64]*streamBlock{}
		completion := ai.Completion{Provider: "anthropic"}
		for stream.Next() {
			event := stream.Current()
			switch event.Type {
			case "message_start":
				completion.RequestID = event.Message.ID
				completion.Model = string(event.Message.Model)
				if event.Message.JSON.Usage.Valid() {
					completion.UsageReported = true
					completion.Usage.InputTokens = int(event.Message.Usage.InputTokens + event.Message.Usage.CacheCreationInputTokens + event.Message.Usage.CacheReadInputTokens)
					completion.Usage.CachedTokens = int(event.Message.Usage.CacheReadInputTokens)
					completion.Usage.CacheCreationTokens = int(event.Message.Usage.CacheCreationInputTokens)
				}
			case "message_delta":
				completion.UsageReported = true
				if input := int(event.Usage.InputTokens + event.Usage.CacheCreationInputTokens + event.Usage.CacheReadInputTokens); input != 0 {
					completion.Usage.InputTokens = input
				}
				completion.Usage.OutputTokens = int(event.Usage.OutputTokens)
				completion.Usage.ReasoningTokens = int(event.Usage.OutputTokensDetails.ThinkingTokens)
				if event.Usage.CacheReadInputTokens != 0 {
					completion.Usage.CachedTokens = int(event.Usage.CacheReadInputTokens)
				}
				if event.Usage.CacheCreationInputTokens != 0 {
					completion.Usage.CacheCreationTokens = int(event.Usage.CacheCreationInputTokens)
				}
				completion.FinishReason = string(event.Delta.StopReason)
				completion.Raw = append(completion.Raw[:0], []byte(event.RawJSON())...)
				snapshot := completion
				snapshot.Raw = append(json.RawMessage(nil), completion.Raw...)
				if !emit(ai.Token{Type: ai.TokenTypeCompletion, Completion: &snapshot}) {
					return
				}
			case "content_block_start":
				blocks[event.Index] = &streamBlock{typ: event.ContentBlock.Type, id: event.ContentBlock.ID, name: event.ContentBlock.Name}
			case "content_block_delta":
				block := blocks[event.Index]
				if block == nil {
					streamErr = fmt.Errorf("anthropic stream delta for unknown block %d", event.Index)
					break
				}
				switch event.Delta.Type {
				case "text_delta":
					if event.Delta.Text != "" && !emit(ai.Token{Type: ai.TokenTypeText, Text: event.Delta.Text, Data: []byte(event.Delta.Text)}) {
						return
					}
				case "thinking_delta":
					if event.Delta.Thinking != "" && !emit(ai.Token{Type: ai.TokenTypeThought, Text: event.Delta.Thinking, Data: []byte(event.Delta.Thinking)}) {
						return
					}
				case "input_json_delta":
					block.input.WriteString(event.Delta.PartialJSON)
				}
			case "content_block_stop":
				block := blocks[event.Index]
				if block == nil {
					streamErr = fmt.Errorf("anthropic stream stop for unknown block %d", event.Index)
					break
				}
				delete(blocks, event.Index)
				if block.typ == "tool_use" {
					call, callErr := streamToolCall(block)
					if callErr != nil {
						streamErr = callErr
						break
					}
					if !emit(ai.Token{Type: ai.TokenTypeToolCall, Data: append([]byte(nil), call.Args...), ToolCall: call}) {
						return
					}
				}
			}
			if streamErr != nil {
				break
			}
		}
		if streamErr == nil {
			streamErr = classifyProviderError(stream.Err())
		}
		if streamErr == nil && len(blocks) != 0 {
			streamErr = fmt.Errorf("anthropic stream ended with %d open content block(s)", len(blocks))
		}
		if streamErr != nil && !errors.Is(streamErr, context.Canceled) {
			emit(ai.Token{Type: ai.TokenTypeErr, Err: streamErr, Text: streamErr.Error()})
		}
	}()
	return ai.DetectToolCallsInStream(ctx, out, m.debug)
}

type streamBlock struct {
	typ, id, name string
	input         strings.Builder
}

func streamToolCall(block *streamBlock) (*ai.ToolCall, error) {
	name := strings.TrimSpace(block.name)
	if name == "" {
		return nil, errors.New("anthropic stream tool use name empty")
	}
	args := json.RawMessage(block.input.String())
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}
	if !json.Valid(args) {
		return nil, errors.New("anthropic stream tool use input is invalid JSON")
	}
	id := strings.TrimSpace(block.id)
	if id == "" {
		id = ai.GenerateToolCallID(name)
	}
	return &ai.ToolCall{ID: id, Type: "function", Name: name, Args: append(json.RawMessage(nil), args...)}, nil
}

func anthropicHTTPStatus(err error) int {
	var providerErr *ai.ProviderError
	if errors.As(err, &providerErr) {
		return providerErr.StatusCode
	}
	var anthropicErr *Error
	if errors.As(err, &anthropicErr) {
		return anthropicErr.StatusCode
	}
	return 0
}

type Tokenizer struct {
	modelName string
	client    *Provider
	debug     gai.ObservationSink
}

func (t *Tokenizer) ID() string { return "anthropic." + t.modelName }
func (t *Tokenizer) Tokenize(ctx context.Context, text string) (tokens []string, err error) {
	_, span := gai.StartOperationSpan(ctx, anthropicTracerName, "ai.anthropic", "ai.operation", "tokenizer.tokenize", attribute.String("ai.provider", "anthropic"), attribute.String("ai.model", t.modelName))
	err = ai.ErrTokenizerUnsupported
	defer gai.EndSpan(span, err)
	return nil, err
}
func (t *Tokenizer) CountTokens(ctx context.Context, text string) (tokens int, err error) {
	ctx, span := gai.StartOperationSpan(ctx, anthropicTracerName, "ai.anthropic", "ai.operation", "tokenizer.count_tokens", attribute.String("ai.provider", "anthropic"), attribute.String("ai.model", t.modelName), attribute.Int("ai.input_length", len(text)))
	defer func() { span.SetAttributes(attribute.Int("ai.input_tokens", tokens)); gai.EndSpan(span, err) }()
	client := t.client.sdkClient()
	count, err := client.Messages.CountTokens(ctx, antropic.MessageCountTokensParams{Model: antropic.Model(t.modelName), Messages: []antropic.MessageParam{antropic.NewUserMessage(antropic.NewTextBlock(text))}})
	if err != nil {
		return 0, localError(err)
	}
	return int(count.InputTokens), nil
}
