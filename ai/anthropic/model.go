package anthropic

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/jsonvalue"
	"github.com/Samuelk0nrad/gai/internal/mediainput"
	"github.com/Samuelk0nrad/gai/internal/modelcatalog"
	"github.com/Samuelk0nrad/gai/internal/streamtext"
	antropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
)

const (
	anthropicTracerName = "github.com/Samuelk0nrad/gai/ai/anthropic"
	defaultMaxTokens    = 4096
	minThinkingTokens   = 1024
)

type Model struct {
	name         string
	client       *Provider
	debug        gai.ObservationSink
	messageHooks []func(*antropic.MessageNewParams) error
	// fallbacks is the model's server-side fallbacks value. When fallbacksSet,
	// it overrides the provider-wide option; nil then turns fallbacks off.
	fallbacks    any
	fallbacksSet bool
	// fallbacksErr reports an invalid WithModelRefusalFallbacks from TypedModel.
	fallbacksErr error
}

var _ ai.Model = (*Model)(nil)
var _ ai.ModelDescriber = (*Model)(nil)

func (m *Model) Name() string { return m.name }
func (m *Model) Close() error { return nil }

// TokenCounter supplies the generic local estimator without provider I/O.
func (m *Model) TokenCounter() ai.TokenCounter { return ai.TextTokenEstimator{} }

func (m *Model) Descriptor() ai.ModelDescriptor {
	mode := m.client.thinkingMode(m.name)
	if facts, ok := m.client.catalog.Lookup(m.name); ok {
		return effectiveAnthropicDescriptor(m.name, facts, mode)
	}
	return effectiveAnthropicDescriptor(m.name, ai.ModelDescriptor{Model: m.name}, mode)
}

func anthropicAdapterDescriptor(model string) ai.ModelDescriptor {
	return ai.ModelDescriptor{Model: model, NativeMessages: ai.FeatureSupportSupported, NativeTools: ai.FeatureSupportSupported,
		ToolChoiceModes: []ai.ToolChoiceMode{ai.ToolChoiceAuto, ai.ToolChoiceNone, ai.ToolChoiceRequired},
		Usage:           ai.FeatureSupportSupported, FinishReason: ai.FeatureSupportSupported, StreamingUsage: ai.FeatureSupportSupported,
		JSONOutput: ai.FeatureSupportUnsupported, JSONSchemaOutput: ai.FeatureSupportSupported,
		Reasoning: ai.FeatureSupportSupported, ReasoningEffort: ai.FeatureSupportSupported,
		ReasoningEfforts: append([]ai.ReasoningEffort(nil), effortsAll...),
	}
}

// anthropicLocalFacts describes a known model without the live catalog. The
// catalog does not report tool-choice restrictions, so ToolChoiceModes always
// comes from here.
func anthropicLocalFacts(model string) ai.ModelDescriptor {
	d := ai.ModelDescriptor{Model: model}
	facts, ok := lookupModelFacts(model)
	if !ok {
		return d
	}
	d.Reasoning = ai.FeatureSupportSupported
	if len(facts.efforts) > 0 {
		d.ReasoningEffort = ai.FeatureSupportSupported
		d.ReasoningEfforts = append([]ai.ReasoningEffort(nil), facts.efforts...)
	} else {
		d.ReasoningEffort = ai.FeatureSupportUnsupported
	}
	if !facts.forcedToolChoice {
		d.ToolChoiceModes = []ai.ToolChoiceMode{ai.ToolChoiceAuto, ai.ToolChoiceNone}
	}
	return d
}

func anthropicProviderDefaults(model string) ai.ModelDescriptor {
	d := anthropicAdapterDescriptor(model)
	d.Reasoning = ai.FeatureSupportUnknown
	d.ReasoningEffort = ai.FeatureSupportUnknown
	return d
}

// effectiveAnthropicDescriptor combines the adapter, local and catalog facts.
// GAI sends effort only with adaptive thinking, so a budget-only model never
// advertises effort, even when the catalog reports it: an effort-only request
// would fail locally.
func effectiveAnthropicDescriptor(model string, catalog ai.ModelDescriptor, mode thinkingMode) ai.ModelDescriptor {
	adapter := anthropicAdapterDescriptor(model)
	facts := anthropicProviderDefaults(model)
	facts = modelcatalog.OverrideModelDescriptor(facts, anthropicLocalFacts(model))
	facts = modelcatalog.OverrideModelDescriptor(facts, catalog)
	descriptor := modelcatalog.IntersectModelDescriptors(adapter, facts)
	if mode == thinkingBudget {
		descriptor.ReasoningEffort = ai.FeatureSupportUnsupported
		descriptor.ReasoningEfforts = nil
	}
	return descriptor
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

// requestRules are the model rules that ai.ModelDescriptor cannot express.
type requestRules struct {
	thinking              thinkingMode
	midConversationSystem bool
}

func buildMessagesRequest(req ai.AIRequest, descriptor ai.ModelDescriptor, rules requestRules) (antropic.MessageNewParams, error) {
	req = req.Copy()
	err := req.Validate()
	if err != nil {
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
	}
	conversationStarted := false
	for _, message := range req.Messages {
		if message.Role != ai.RoleSystem {
			conversationStarted = true
			continue
		}
		if conversationStarted {
			if !rules.midConversationSystem {
				return antropic.MessageNewParams{}, fmt.Errorf("%w: Anthropic model %s only supports leading system messages", ai.ErrUnsupportedCapability, descriptor.Model)
			}
			continue
		}
		if err := unsupportedExtensions(message.Extensions); err != nil {
			return antropic.MessageNewParams{}, err
		}
		for _, part := range message.Parts {
			if err := unsupportedExtensions(part.Extensions); err != nil {
				return antropic.MessageNewParams{}, err
			}
			text := part.Text
			if part.Kind == ai.ContentJSON {
				text = string(part.JSON)
			} else if part.Kind != ai.ContentText {
				return antropic.MessageNewParams{}, fmt.Errorf("%w: Anthropic system content %q", ai.ErrUnsupportedCapability, part.Kind)
			}
			p.System = append(p.System, antropic.TextBlockParam{Text: text})
		}
	}
	msgs, err := mapNativeMessages(req.Messages, rules.midConversationSystem)
	if err != nil {
		return antropic.MessageNewParams{}, err
	}
	p.Messages = msgs
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
		case ai.ReasoningEffortXHigh:
			p.OutputConfig.Effort = antropic.OutputConfigEffortXhigh
		case ai.ReasoningEffortMax:
			p.OutputConfig.Effort = antropic.OutputConfigEffortMax
		default:
			return antropic.MessageNewParams{}, fmt.Errorf("%w: %s", ai.ErrUnsupportedCapability, req.Reasoning.Effort)
		}
	}
	if req.Reasoning.Enabled || req.Reasoning.IncludeThoughts || req.Reasoning.BudgetTokens > 0 || req.Reasoning.Effort != "" {
		if req.Reasoning.BudgetTokens > 0 {
			if rules.thinking == thinkingAdaptive {
				return antropic.MessageNewParams{}, fmt.Errorf("%w: anthropic model %s accepts only adaptive thinking; set Reasoning.Effort instead of Reasoning.BudgetTokens", ai.ErrUnsupportedCapability, descriptor.Model)
			}
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
		} else if adaptiveThinking(rules.thinking, descriptor) {
			display := antropic.ThinkingConfigAdaptiveDisplayOmitted
			if req.Reasoning.IncludeThoughts {
				display = antropic.ThinkingConfigAdaptiveDisplaySummarized
			}
			p.Thinking = antropic.ThinkingConfigParamUnion{OfAdaptive: &antropic.ThinkingConfigAdaptiveParam{Display: display}}
		} else {
			return antropic.MessageNewParams{}, fmt.Errorf("%w: anthropic adaptive thinking is unsupported by %s; set Reasoning.BudgetTokens", ai.ErrUnsupportedCapability, descriptor.Model)
		}
		if p.ToolChoice.OfAny != nil || p.ToolChoice.OfTool != nil {
			return antropic.MessageNewParams{}, fmt.Errorf("%w: anthropic thinking is incompatible with required tool choice", ai.ErrUnsupportedCapability)
		}
	}
	return p, nil
}

// adaptiveThinking reports whether a request without a budget can use adaptive
// thinking. Models GAI knows nothing about fall back to effort support, which
// is how the API introduced adaptive thinking.
func adaptiveThinking(mode thinkingMode, descriptor ai.ModelDescriptor) bool {
	switch mode {
	case thinkingAdaptive, thinkingAdaptiveOrBudget:
		return true
	case thinkingBudget:
		return false
	default:
		return descriptor.ReasoningEffort != ai.FeatureSupportUnsupported
	}
}

// mapNativeMessages maps the conversation. Leading system messages belong to
// the top-level system field and are skipped. Later system messages become
// role "system" entries when midConversationSystem is set; callers reject
// them otherwise.
func mapNativeMessages(messages []ai.Message, midConversationSystem bool) ([]antropic.MessageParam, error) {
	out := make([]antropic.MessageParam, 0, len(messages))
	conversationStarted := false
	toolResults := []antropic.ContentBlockParamUnion{}
	flushToolResults := func() {
		if len(toolResults) > 0 {
			out = append(out, antropic.NewUserMessage(toolResults...))
			toolResults = nil
		}
	}
	for _, m := range messages {
		if err := m.Validate(); err != nil {
			return nil, err
		}
		if err := unsupportedExtensions(m.Extensions); err != nil {
			return nil, err
		}
		if m.Role == ai.RoleSystem {
			if !conversationStarted || !midConversationSystem {
				continue
			}
			blocks, err := systemBlocks(m)
			if err != nil {
				return nil, err
			}
			flushToolResults()
			if len(blocks) == 0 {
				continue
			}
			if last := len(out) - 1; last >= 0 && out[last].Role == antropic.MessageParamRoleSystem {
				out[last].Content = append(out[last].Content, blocks...)
			} else {
				out = append(out, antropic.MessageParam{Role: antropic.MessageParamRoleSystem, Content: blocks})
			}
			continue
		}
		conversationStarted = true
		parts := m.Parts
		if m.Role == ai.RoleAssistant {
			parts = dropPreFallbackState(parts)
		}
		blocks := []antropic.ContentBlockParamUnion{}
		for _, part := range parts {
			switch part.Kind {
			case ai.ContentText, ai.ContentJSON:
				if err := unsupportedExtensions(part.Extensions); err != nil {
					return nil, err
				}
				text := part.Text
				if part.Kind == ai.ContentJSON {
					text = string(part.JSON)
				}
				blocks = append(blocks, antropic.NewTextBlock(text))
			case ai.ContentReasoning:
				signature, err := anthropicExtension(part.Extensions, "thinking_signature")
				if err != nil {
					return nil, err
				}
				if signature == "" {
					return nil, fmt.Errorf("%w: Anthropic reasoning requires its signature", ai.ErrUnsupportedCapability)
				}
				blocks = append(blocks, antropic.NewThinkingBlock(signature, part.Text))
			case ai.ContentToolCall:
				if err := unsupportedExtensions(part.Extensions); err != nil {
					return nil, err
				}
				c := part.ToolCall
				if err := unsupportedExtensions(c.Extensions); err != nil {
					return nil, err
				}
				var input any
				if err := json.Unmarshal(c.Args, &input); err != nil {
					return nil, err
				}
				blocks = append(blocks, antropic.NewToolUseBlock(c.ID, input, c.Name))
			case ai.ContentToolResult:
				if err := unsupportedExtensions(part.Extensions); err != nil {
					return nil, err
				}
				block, err := toolResultBlock(part.ToolResult)
				if err != nil {
					return nil, err
				}
				blocks = append(blocks, block)
			case ai.ContentMedia:
				if err := unsupportedExtensions(part.Extensions); err != nil {
					return nil, err
				}
				if m.Role != ai.RoleUser {
					return nil, mediainput.Unsupported("anthropic", "media is only accepted in user messages and tool results, not %s messages", m.Role)
				}
				block, err := mediaBlock(part.Media)
				if err != nil {
					return nil, err
				}
				blocks = append(blocks, block)
			case ai.ContentExtension:
				raw, err := hostedBlock(part.Extensions)
				if err != nil {
					return nil, err
				}
				if raw != nil {
					// Hosted tool extensions are optional, so any required
					// extension beside one is state this part cannot replay.
					if err := unsupportedExtensions(part.Extensions); err != nil {
						return nil, err
					}
					blocks = append(blocks, param.Override[antropic.ContentBlockParamUnion](raw))
					continue
				}
				if hasAnthropicExtension(part.Extensions, fallbackExtension) {
					raw, err := anthropicRawExtension(part.Extensions, fallbackExtension)
					if err != nil {
						return nil, err
					}
					blocks = append(blocks, param.Override[antropic.ContentBlockParamUnion](raw))
					continue
				}
				data, err := anthropicExtension(part.Extensions, "redacted_thinking")
				if err != nil {
					return nil, err
				}
				if data != "" {
					blocks = append(blocks, antropic.NewRedactedThinkingBlock(data))
				}
			default:
				return nil, fmt.Errorf("%w: Anthropic content %q", ai.ErrUnsupportedCapability, part.Kind)
			}
		}
		if m.Role == ai.RoleTool {
			toolResults = append(toolResults, blocks...)
			continue
		}
		flushToolResults()
		if len(blocks) == 0 {
			continue
		}
		if m.Role == ai.RoleUser {
			out = append(out, antropic.NewUserMessage(blocks...))
		} else {
			out = append(out, antropic.NewAssistantMessage(blocks...))
		}
	}
	flushToolResults()
	if err := validateSystemPlacement(out); err != nil {
		return nil, err
	}
	return out, nil
}

// systemBlocks maps a mid-conversation system message to text blocks.
func systemBlocks(m ai.Message) ([]antropic.ContentBlockParamUnion, error) {
	blocks := make([]antropic.ContentBlockParamUnion, 0, len(m.Parts))
	for _, part := range m.Parts {
		if err := unsupportedExtensions(part.Extensions); err != nil {
			return nil, err
		}
		switch part.Kind {
		case ai.ContentText:
			blocks = append(blocks, antropic.NewTextBlock(part.Text))
		case ai.ContentJSON:
			blocks = append(blocks, antropic.NewTextBlock(string(part.JSON)))
		default:
			return nil, fmt.Errorf("%w: Anthropic system content %q", ai.ErrUnsupportedCapability, part.Kind)
		}
	}
	return blocks, nil
}

// validateSystemPlacement applies the API's rule for mid-conversation system
// messages: each follows a user turn (tool results count) and is either last
// or followed by an assistant turn.
func validateSystemPlacement(messages []antropic.MessageParam) error {
	for i, message := range messages {
		if message.Role != antropic.MessageParamRoleSystem {
			continue
		}
		if i == 0 || messages[i-1].Role != antropic.MessageParamRoleUser {
			return fmt.Errorf("%w: Anthropic mid-conversation system messages must follow a user or tool message", ai.ErrUnsupportedCapability)
		}
		if i+1 < len(messages) && messages[i+1].Role != antropic.MessageParamRoleAssistant {
			return fmt.Errorf("%w: Anthropic mid-conversation system messages must be last or followed by an assistant message", ai.ErrUnsupportedCapability)
		}
	}
	return nil
}

const fallbackExtension = "fallback"

// dropPreFallbackState removes the reasoning, redacted reasoning, tool calls
// and hosted server tool blocks (server_tool_use and its results) that
// precede the last fallback block. They belong to the declined attempt: the
// API rejects them on replay, and a tool call before the switch was never
// part of the answer the fallback model produced. Text and the fallback
// blocks stay in place.
func dropPreFallbackState(parts []ai.ContentPart) []ai.ContentPart {
	last := -1
	for i, part := range parts {
		if part.Kind == ai.ContentExtension && hasAnthropicExtension(part.Extensions, fallbackExtension) {
			last = i
		}
	}
	if last < 0 {
		return parts
	}
	kept := make([]ai.ContentPart, 0, len(parts))
	for i, part := range parts {
		if i < last && isAttemptState(part) {
			continue
		}
		kept = append(kept, part)
	}
	return kept
}

// dropDeclinedToolCalls removes tool calls from a response before a fallback
// block. Reasoning stays in the response, as it does in a stream, which cannot
// hold it back without losing streaming; dropPreFallbackState removes it on
// replay.
func dropDeclinedToolCalls(parts []ai.ContentPart) []ai.ContentPart {
	kept := parts[:0]
	for _, part := range parts {
		if part.Kind != ai.ContentToolCall {
			kept = append(kept, part)
		}
	}
	return kept
}

func isAttemptState(part ai.ContentPart) bool {
	switch part.Kind {
	case ai.ContentReasoning, ai.ContentToolCall:
		return true
	case ai.ContentExtension:
		return hasAnthropicExtension(part.Extensions, "redacted_thinking") || hasHostedBlock(part.Extensions)
	default:
		return false
	}
}

func hasAnthropicExtension(extensions []ai.Extension, kind string) bool {
	for _, ext := range extensions {
		if ext.Namespace == "anthropic" && ext.Type == kind {
			return true
		}
	}
	return false
}

// anthropicRawExtension returns the JSON data of the anthropic extension kind,
// or nil when it is absent. Any other required extension is rejected.
func anthropicRawExtension(extensions []ai.Extension, kind string) (json.RawMessage, error) {
	var value json.RawMessage
	for _, ext := range extensions {
		if ext.Namespace == "anthropic" && ext.Type == kind {
			if !json.Valid(ext.Data) {
				return nil, fmt.Errorf("decode Anthropic %s: invalid JSON", kind)
			}
			value = append(json.RawMessage(nil), ext.Data...)
		} else if ext.Required {
			return nil, fmt.Errorf("%w: Anthropic extension %s/%s", ai.ErrUnsupportedCapability, ext.Namespace, ext.Type)
		}
	}
	return value, nil
}

// fallbackPart preserves a fallback block so it replays where it appeared.
func fallbackPart(raw string) ai.ContentPart {
	return ai.ContentPart{Kind: ai.ContentExtension, Extensions: []ai.Extension{{Namespace: "anthropic", Type: fallbackExtension, Data: json.RawMessage(raw), Required: true}}}
}

// fallbackTarget returns the model a fallback block switched to.
func fallbackTarget(raw string) string {
	var block struct {
		To struct {
			Model string `json:"model"`
		} `json:"to"`
	}
	if json.Unmarshal([]byte(raw), &block) != nil {
		return ""
	}
	return block.To.Model
}

func unsupportedExtensions(extensions []ai.Extension) error {
	for _, ext := range extensions {
		if ext.Required {
			return fmt.Errorf("%w: Anthropic extension %s/%s", ai.ErrUnsupportedCapability, ext.Namespace, ext.Type)
		}
	}
	return nil
}

// anthropicExtension returns the string value of the anthropic extension kind.
func anthropicExtension(extensions []ai.Extension, kind string) (string, error) {
	raw, err := anthropicRawExtension(extensions, kind)
	if err != nil || raw == nil {
		return "", err
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", fmt.Errorf("decode Anthropic %s: %w", kind, err)
	}
	return value, nil
}

func anthropicExtensions(kind, value string) []ai.Extension {
	if value == "" {
		return nil
	}
	data, _ := json.Marshal(value)
	return []ai.Extension{{Namespace: "anthropic", Type: kind, Data: data, Required: true}}
}

func mapTools(defs []ai.ToolDefinition) ([]antropic.ToolUnionParam, error) {
	tools := make([]antropic.ToolUnionParam, 0, len(defs))
	for _, def := range defs {
		if err := def.Validate(); err != nil {
			return nil, err
		}
		var schema map[string]any
		if err := jsonvalue.Decode(def.Parameters, &schema); err != nil {
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
		if err := jsonvalue.Decode(format.Schema, &schema); err != nil {
			return nil, fmt.Errorf("decode anthropic response schema: %w", err)
		}
		return &antropic.OutputConfigParam{Format: antropic.JSONOutputFormatParam{Schema: schema}}, nil
	default:
		return nil, fmt.Errorf("%w: %s", ai.ErrInvalidResponseFormat, format.Type)
	}
}

func (m *Model) Generate(ctx context.Context, req ai.AIRequest) (response *ai.AIResponse, err error) {
	req = req.Copy()
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
		gai.AddObservationContent(ctx, m.debug, fields, "prompt", gai.ContentKindPrompt, observationPrompt(req))
		gai.EmitObservation(ctx, m.debug, gai.Observation{Name: "anthropic_generate_request", Source: "ai:anthropic.Model.Generate", Fields: fields})
	}
	generationResult := ai.GenerationResult{}
	defer func() {
		generationResult.Err = err
		generationResult.HTTPStatus = anthropicHTTPStatus(err)
		if response != nil {
			generationResult.Output = &response.Message
		}
		observation.Finish(generationResult)
	}()
	options, _ := m.requestOptions(req, payload)
	// A long server tool turn can stop with pause_turn. The paused turn is
	// sent back and continued; the result combines every request's content
	// and usage. Raw, the request ID and the stop details are the last
	// request's.
	var message *antropic.Message
	var content []antropic.ContentBlockUnion
	var usage ai.Usage
	usageReported := false
	base := payload.Messages
	for continuation := 0; ; continuation++ {
		message, err = client.Messages.New(ctx, payload, options...)
		if err != nil {
			// The paused requests before this one were billed.
			if usageReported {
				generationResult.Usage = &usage
			}
			return nil, localError(err)
		}
		content = append(content, message.Content...)
		usage.Add(ai.Usage{
			InputTokens:     int(message.Usage.InputTokens + message.Usage.CacheCreationInputTokens + message.Usage.CacheReadInputTokens),
			OutputTokens:    int(message.Usage.OutputTokens),
			ReasoningTokens: int(message.Usage.OutputTokensDetails.ThinkingTokens),
			CachedTokens:    int(message.Usage.CacheReadInputTokens), CacheCreationTokens: int(message.Usage.CacheCreationInputTokens),
		})
		usageReported = usageReported || message.JSON.Usage.Valid()
		if message.StopReason != antropic.StopReasonPauseTurn || continuation == maxPauseContinuations {
			break
		}
		payload.Messages = continuePausedTurn(base, content)
	}
	if usageReported {
		generationResult.Usage = &usage
	}
	semantic, err := mapCanonicalMessageContent(content)
	text, thinking, calls := semantic.Text(), semantic.Reasoning(), semantic.ToolCalls()
	if err != nil {
		return nil, err
	}
	input, output := usage.InputTokens, usage.OutputTokens
	generationResult.ResponseModel = string(message.Model)
	generationResult.RequestID = message.ID
	generationResult.FinishReason = string(message.StopReason)
	generationResult.ToolCallCount = len(calls)
	if err := anthropicTerminalError(string(message.StopReason), message.StopDetails); err != nil {
		return nil, err
	}
	if gai.ObservationEnabled(ctx, m.debug) {
		fields := map[string]any{"input_tokens": input, "output_tokens": output}
		gai.AddObservationContent(ctx, m.debug, fields, "response_text", gai.ContentKindCompletion, text)
		gai.AddObservationContent(ctx, m.debug, fields, "reasoning", gai.ContentKindReasoning, thinking)
		gai.EmitObservation(ctx, m.debug, gai.Observation{Name: "anthropic_generate_success", Source: "ai:anthropic.Model.Generate", Fields: fields})
	}
	response = &ai.AIResponse{Raw: json.RawMessage(message.RawJSON()), FinishReason: string(message.StopReason), InputTokens: input, OutputTokens: output, ReasoningTokens: usage.ReasoningTokens}
	response.SetMessage(semantic)
	return response, nil
}

// mapCanonicalMessageContent maps a response. A malformed tool call fails the
// response unless a later fallback block shows it belonged to a declined
// attempt, matching GenerateStream.
func mapCanonicalMessageContent(blocks []antropic.ContentBlockUnion) (ai.Message, error) {
	message := ai.Message{Role: ai.RoleAssistant}
	var toolErr error
	for i, block := range blocks {
		switch block.Type {
		case "text":
			var citations []json.RawMessage
			for _, citation := range block.Citations {
				if raw := streamtext.RawObject(citation.RawJSON()); raw != nil {
					citations = append(citations, raw)
				}
			}
			message.Parts = append(message.Parts, ai.ContentPart{Kind: ai.ContentText, Text: block.Text, Extensions: ai.CitationsExtension(extensionNamespace, textCitations(citations, block.Text))})
		case "server_tool_use":
			raw := streamtext.RawObject(block.RawJSON())
			if raw == nil {
				return ai.Message{}, fmt.Errorf("content[%d]: server tool use block is invalid JSON", i)
			}
			message.Parts = append(message.Parts, serverToolUsePart(raw))
		case "thinking":
			message.Parts = append(message.Parts, ai.ContentPart{Kind: ai.ContentReasoning, Text: block.Thinking, Extensions: anthropicExtensions("thinking_signature", block.Signature)})
		case "redacted_thinking":
			message.Parts = append(message.Parts, ai.ContentPart{Kind: ai.ContentExtension, Extensions: anthropicExtensions("redacted_thinking", block.Data)})
		case fallbackExtension:
			raw := block.RawJSON()
			if !json.Valid([]byte(raw)) {
				return ai.Message{}, fmt.Errorf("content[%d]: fallback block is invalid JSON", i)
			}
			// The declined attempt's tool calls, valid or not, are dropped.
			toolErr = nil
			message.Parts = dropDeclinedToolCalls(message.Parts)
			message.Parts = append(message.Parts, fallbackPart(raw))
		case "tool_use":
			name := strings.TrimSpace(block.Name)
			if name == "" {
				toolErr = cmp.Or(toolErr, fmt.Errorf("content[%d]: tool use name empty", i))
				continue
			}
			args := block.Input
			if len(args) == 0 || string(args) == "null" {
				args = json.RawMessage("{}")
			}
			if !json.Valid(args) {
				toolErr = cmp.Or(toolErr, fmt.Errorf("content[%d]: tool use input is invalid JSON", i))
				continue
			}
			id := strings.TrimSpace(block.ID)
			if id == "" {
				id = ai.GenerateToolCallID(name)
			}
			message.Parts = append(message.Parts, ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: id, Type: "function", Name: name, Args: append(json.RawMessage(nil), args...)}})
		default:
			if serverToolResultBlocks[block.Type] {
				raw := streamtext.RawObject(block.RawJSON())
				if raw == nil {
					return ai.Message{}, fmt.Errorf("content[%d]: %s block is invalid JSON", i, block.Type)
				}
				message.Parts = append(message.Parts, serverToolResultPart(raw))
				continue
			}
			return ai.Message{}, fmt.Errorf("%w: Anthropic output %q", ai.ErrUnsupportedCapability, block.Type)
		}
	}
	if toolErr != nil {
		return ai.Message{}, toolErr
	}
	if len(message.Parts) == 0 {
		message.Parts = ai.TextParts("")
	}
	return message, nil
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
	out := make(chan ai.Token, 8)
	go func() {
		ctx := ctx
		var streamErr error
		var observation *ai.GenerationObservation
		generationResult := ai.GenerationResult{}
		// Text-encoded tool calls are detected in this goroutine, so the
		// generation observes the tokens its caller receives.
		// run follows the text part the caller builds from delivered tokens,
		// so a cited text block's citations cover its text in that part.
		var run streamtext.Run
		detector := ai.NewToolCallDetector(ctx, m.debug, func(token ai.Token) bool {
			token = placeCitations(&run, token)
			run.Observe(token)
			if !ai.SendToken(ctx, out, token) {
				return false
			}
			observation.Delivered(token)
			return true
		})
		defer func() {
			// A failed final flush means the context ended before the
			// buffered tokens were delivered.
			if !detector.Close() && streamErr == nil {
				streamErr = ctx.Err()
			}
			generationResult.Err = streamErr
			generationResult.HTTPStatus = anthropicHTTPStatus(streamErr)
			observation.Finish(generationResult)
			close(out)
		}()
		emit := func(token ai.Token) bool {
			observation.ObserveToken(token)
			if detector.Push(token) {
				return true
			}
			streamErr = detector.Err()
			if streamErr == nil {
				streamErr = ctx.Err()
			}
			return false
		}
		// With fallbacks on, a fallback block can follow a tool call of the
		// declined attempt. From the first tool_use on, parts are held in
		// order so those calls can be dropped before anyone runs them.
		fallbacksOn := false
		holding := false
		var held []ai.Token
		emitPart := func(token ai.Token) bool {
			if holding {
				held = append(held, token)
				return true
			}
			return emit(token)
		}
		release := func(dropToolCalls bool) bool {
			pending := held
			held, holding = nil, false
			for _, token := range pending {
				if dropToolCalls && token.Part != nil && token.Part.Kind == ai.ContentToolCall {
					continue
				}
				if !emit(token) {
					return false
				}
			}
			return true
		}
		req = req.Copy()
		if err := ai.ValidateModelRequest(m, req); err != nil {
			streamErr = err
			emit(ai.Token{Err: err})
			return
		}
		payload, err := m.messageParams(req)
		if err != nil {
			streamErr = err
			emit(ai.Token{Err: err})
			return
		}
		client := m.client.sdkClient()
		generationCtx, startedObservation := ai.StartGenerationObservation(ctx, req, ai.GenerationConfig{Provider: "anthropic", Model: m.name, Streaming: true, Sink: m.debug})
		observation = startedObservation
		ctx = generationCtx
		// Detection and its sink calls are application time.
		detector.PauseIn(generationCtx)
		options, on := m.requestOptions(req, payload)
		fallbacksOn = on
		var pendingToolErr error
		completion := ai.Completion{Provider: "anthropic"}
		var stopDetails antropic.RefusalStopDetails
		// A long server tool turn can stop with pause_turn. The paused turn is
		// sent back and continued in a new request, and its parts keep
		// streaming as one generation. prior holds the usage of the paused
		// requests; completion.Usage holds the current request's.
		// initialInput is the first request's input, which a budget
		// checkpoint needs, and withheld is set while a paused request's
		// usage waits for the final completion.
		var prior ai.Usage
		initialInput := 0
		withheld := false
		var blocks map[int64]*streamBlock
		base := payload.Messages
		var pausedContent []antropic.ContentBlockUnion
		// Each request's stream is closed before the next one starts, and the
		// last one when the generation ends.
		closeStream := func() error { return nil }
		defer func() { _ = closeStream() }()
		for continuation := 0; ; continuation++ {
			_ = closeStream()
			stream := client.Messages.NewStreaming(generationCtx, payload, options...)
			closeStream = stream.Close
			blocks = map[int64]*streamBlock{}
			var turn antropic.Message
			var turnErr error
			paused := false
			for stream.Next() {
				event := stream.Current()
				if err := turn.Accumulate(event); err != nil && turnErr == nil {
					turnErr = err
				}
				switch event.Type {
				case "message_start":
					completion.RequestID = event.Message.ID
					completion.Model = string(event.Message.Model)
					if event.Message.JSON.Usage.Valid() {
						completion.UsageReported = true
						completion.Usage.InputTokens = int(event.Message.Usage.InputTokens + event.Message.Usage.CacheCreationInputTokens + event.Message.Usage.CacheReadInputTokens)
						// message_start can already report output tokens. A
						// message_delta replaces them with its cumulative count, but
						// a request that fails before one still reports them.
						completion.Usage.OutputTokens = int(event.Message.Usage.OutputTokens)
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
					stopDetails = event.Delta.StopDetails
					if !release(false) {
						return
					}
					completion.Raw = append(completion.Raw[:0], []byte(event.RawJSON())...)
					if completion.FinishReason == string(antropic.StopReasonPauseTurn) && continuation < maxPauseContinuations {
						// The completion is reported once, after the last request.
						paused, withheld = true, true
						if continuation == 0 {
							initialInput = completion.Usage.InputTokens
						}
						continue
					}
					snapshot := combinedCompletion(completion, prior, initialInput)
					withheld = false
					if !emit(ai.Token{Completion: &snapshot}) {
						return
					}
				case "content_block_start":
					blocks[event.Index] = &streamBlock{typ: event.ContentBlock.Type, id: event.ContentBlock.ID, name: event.ContentBlock.Name}
					block := blocks[event.Index]
					block.signature.WriteString(event.ContentBlock.Signature)
					switch event.ContentBlock.Type {
					case "text":
						block.text.WriteString(event.ContentBlock.Text)
						if event.ContentBlock.Text != "" && !emitPart(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: event.ContentBlock.Text}}) {
							return
						}
					case "thinking":
						if event.ContentBlock.Thinking != "" && !emitPart(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentReasoning, Text: event.ContentBlock.Thinking}}) {
							return
						}
					case "redacted_thinking":
						part := ai.ContentPart{Kind: ai.ContentExtension, Extensions: anthropicExtensions("redacted_thinking", event.ContentBlock.Data)}
						if !emitPart(ai.Token{Part: &part}) {
							return
						}
					case "tool_use":
						holding = holding || fallbacksOn
					case "server_tool_use":
						block.start = event.ContentBlock.RawJSON()
					case fallbackExtension:
						// The declined attempt's tool calls never reach the caller,
						// and its stream-level tool errors no longer apply.
						raw := event.ContentBlock.RawJSON()
						if !json.Valid([]byte(raw)) {
							streamErr = errors.New("anthropic stream fallback block is invalid JSON")
							break
						}
						pendingToolErr = nil
						if !release(true) {
							return
						}
						if model := fallbackTarget(raw); model != "" {
							completion.Model = model
						}
						part := fallbackPart(raw)
						if !emit(ai.Token{Part: &part}) {
							return
						}
					default:
						if serverToolResultBlocks[event.ContentBlock.Type] {
							raw := streamtext.RawObject(event.ContentBlock.RawJSON())
							if raw == nil {
								streamErr = fmt.Errorf("anthropic stream %s block is invalid JSON", event.ContentBlock.Type)
								break
							}
							part := serverToolResultPart(raw)
							if !emitPart(ai.Token{Part: &part}) {
								return
							}
							break
						}
						streamErr = fmt.Errorf("%w: Anthropic stream content %q", ai.ErrUnsupportedCapability, event.ContentBlock.Type)
					}
				case "content_block_delta":
					block := blocks[event.Index]
					if block == nil {
						streamErr = fmt.Errorf("anthropic stream delta for unknown block %d", event.Index)
						break
					}
					switch event.Delta.Type {
					case "text_delta":
						block.text.WriteString(event.Delta.Text)
						if event.Delta.Text != "" && !emitPart(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: event.Delta.Text}}) {
							return
						}
					case "thinking_delta":
						if event.Delta.Thinking != "" && !emitPart(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentReasoning, Text: event.Delta.Thinking}}) {
							return
						}
					case "signature_delta":
						block.signature.WriteString(event.Delta.Signature)
					case "input_json_delta":
						block.input.WriteString(event.Delta.PartialJSON)
					case "citations_delta":
						if raw := streamtext.RawObject(event.Delta.Citation.RawJSON()); raw != nil {
							block.citations = append(block.citations, raw)
						}
					}
				case "content_block_stop":
					block := blocks[event.Index]
					if block == nil {
						streamErr = fmt.Errorf("anthropic stream stop for unknown block %d", event.Index)
						break
					}
					delete(blocks, event.Index)
					if block.typ == "thinking" && block.signature.Len() > 0 {
						part := ai.ContentPart{Kind: ai.ContentReasoning, Extensions: anthropicExtensions("thinking_signature", block.signature.String())}
						if !emitPart(ai.Token{Part: &part}) {
							return
						}
					}
					if block.typ == "text" && len(block.citations) > 0 {
						// The citations attach to the text part the block's deltas
						// built; delivery shifts them onto its text.
						part := ai.ContentPart{Kind: ai.ContentText, Extensions: ai.CitationsExtension(extensionNamespace, textCitations(block.citations, block.text.String()))}
						if !emitPart(ai.Token{Part: &part}) {
							return
						}
					}
					if block.typ == "server_tool_use" {
						part := serverToolUsePart(streamedServerToolUse(block.start, block.input.String()))
						if !emitPart(ai.Token{Part: &part}) {
							return
						}
					}
					if block.typ == "tool_use" {
						call, callErr := streamToolCall(block)
						if callErr != nil {
							if pendingToolErr == nil {
								pendingToolErr = callErr
							}
							continue
						}
						if !emitPart(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: call}}) {
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
			if streamErr != nil || !paused {
				break
			}
			if turnErr != nil {
				streamErr = fmt.Errorf("anthropic stream: rebuild paused turn: %w", turnErr)
				break
			}
			prior.Add(completion.Usage)
			completion.Usage = ai.Usage{}
			pausedContent = append(pausedContent, turn.Content...)
			payload.Messages = continuePausedTurn(base, pausedContent)
		}
		// A stream that ends cleanly without message_delta still delivers the
		// parts held back for a possible fallback.
		if streamErr == nil && !release(false) {
			return
		}
		if streamErr == nil {
			if terminalErr := anthropicTerminalError(completion.FinishReason, stopDetails); terminalErr != nil {
				streamErr = terminalErr
			} else {
				streamErr = pendingToolErr
			}
		}
		if streamErr != nil && withheld {
			// A later request failed after paused ones were billed. Their
			// usage, and any the failed request reported, comes before the
			// error. The paused request's stop reason and payload do not
			// describe this outcome.
			snapshot := combinedCompletion(completion, prior, initialInput)
			snapshot.FinishReason, snapshot.Raw = "", nil
			err := streamErr
			emit(ai.Token{Completion: &snapshot})
			streamErr = err
		}
		if streamErr != nil && !errors.Is(streamErr, context.Canceled) {
			emit(ai.Token{Err: streamErr})
		}
	}()
	return out
}

// combinedCompletion reports a generation that may span several requests:
// prior is the usage of the paused requests before the current one, and
// initialInput the first request's input when there were any.
func combinedCompletion(current ai.Completion, prior ai.Usage, initialInput int) ai.Completion {
	snapshot := current
	snapshot.Usage = prior
	snapshot.Usage.Add(current.Usage)
	snapshot.InitialInputTokens = initialInput
	snapshot.Raw = append(json.RawMessage(nil), current.Raw...)
	return snapshot
}

type streamBlock struct {
	typ, id, name string
	input         strings.Builder
	signature     strings.Builder
	// text and citations collect a text block's content for its citations.
	text      strings.Builder
	citations []json.RawMessage
	// start is the raw content_block_start block of a server_tool_use.
	start string
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
