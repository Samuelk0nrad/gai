package anthropic

import (
	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/lace-ai/gai/ai"
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
	params, err := buildMessagesRequest(req.Copy(), m.Descriptor())
	if err != nil {
		return params, err
	}
	for _, hook := range m.messageHooks {
		if err := hook(&params); err != nil {
			return params, err
		}
	}
	params.Model = sdk.Model(m.name)
	return params, nil
}
