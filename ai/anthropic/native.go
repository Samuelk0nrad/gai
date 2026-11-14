package anthropic

import (
	"github.com/Samuelk0nrad/gai/ai"
	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// ModelOption configures a concrete model at construction.
type ModelOption func(*Model)

// WithMessageParams appends a hook run on freshly mapped parameters for both
// Generate and GenerateStream. Hooks must be concurrency-safe, must not retain
// parameter pointers, and must preserve the portable tool/message protocol.
// Captured mutable data is caller-owned. Model identity is reserved by GAI.
// Use SDKClient when native responses or stream events are required.
func WithMessageParams(hook func(*sdk.MessageNewParams) error) ModelOption {
	return func(m *Model) {
		if hook != nil {
			m.messageHooks = append(m.messageHooks, hook)
		}
	}
}

// SDKClient returns an independent configured SDK client. Native calls preserve
// SDK types and bypass GAI preflight, observations, normalization, and retries.
// SDK retries are disabled. The configured HTTP client timeout still applies.
func (p *Provider) SDKClient() (*sdk.Client, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	client := p.sdkClient()
	return &client, nil
}

func (m *Model) messageParams(req ai.AIRequest) (sdk.MessageNewParams, error) {
	params, err := buildMessagesRequest(req.Copy(), m.Descriptor(), m.requestRules())
	if err != nil {
		return params, err
	}
	if m.client.promptCache != nil {
		params.CacheControl = *m.client.promptCache
	}
	for _, hook := range m.messageHooks {
		if err := hook(&params); err != nil {
			return params, err
		}
	}
	params.Model = sdk.Model(m.name)
	return params, nil
}

func (m *Model) requestRules() requestRules {
	facts, _ := lookupModelFacts(m.name)
	return requestRules{thinking: m.client.thinkingMode(m.name), midConversationSystem: facts.midConversationSystem}
}

// fallbacksValue returns the server-side fallbacks field for this model, or
// nil when fallbacks are off. The provider-wide option applies only to models
// known to have server-side fallbacks; a model option always applies.
func (m *Model) fallbacksValue() any {
	if m.fallbacksSet {
		return m.fallbacks
	}
	if facts, ok := lookupModelFacts(m.name); ok && facts.serverFallbacks {
		return m.client.fallbacks
	}
	return nil
}

// requestOptions adds the fallbacks beta to a generation request and reports
// whether fallbacks are on. The field is set on the serialized body, so hooks
// cannot drop it by replacing extra fields, and count_tokens never sends it. A
// hook that sets fallbacks itself wins: a value turns fallbacks on, and nil
// turns them off and removes the field. A history that replays fallback
// blocks also needs the beta header.
func (m *Model) requestOptions(req ai.AIRequest, params sdk.MessageNewParams) ([]option.RequestOption, bool) {
	fallbacks := m.fallbacksValue()
	var opts []option.RequestOption
	if hookValue, hookSet := params.ExtraFields()[fallbacksBodyField]; hookSet {
		fallbacks = hookValue
		if fallbacks == nil {
			opts = append(opts, option.WithJSONDel(fallbacksBodyField))
		}
	} else if fallbacks != nil {
		opts = append(opts, option.WithJSONSet(fallbacksBodyField, fallbacks))
	}
	on := fallbacks != nil
	if on || hasFallbackHistory(req.Messages) {
		opts = append(opts, option.WithHeaderAdd(anthropicBetaHeader, fallbackBetaHeader))
	}
	return opts, on
}

// hasFallbackHistory reports whether messages replay a fallback block.
func hasFallbackHistory(messages []ai.Message) bool {
	for _, message := range messages {
		for _, part := range message.Parts {
			if part.Kind == ai.ContentExtension && hasAnthropicExtension(part.Extensions, fallbackExtension) {
				return true
			}
		}
	}
	return false
}
