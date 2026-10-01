package gemini

import (
	"context"

	"github.com/lace-ai/gai/ai"
	"google.golang.org/genai"
)

// ModelOption configures a concrete model at construction.
type ModelOption func(*Model)

// WithGenerateContentConfig appends a hook run on fresh config for synchronous
// and streaming calls. Hooks must be concurrency-safe and must not retain config
// pointers. Captured mutable data is caller-owned. Preserve portable tool/message
// semantics; use SDKClient for native contents, responses, or stream events.
func WithGenerateContentConfig(hook func(*genai.GenerateContentConfig) error) ModelOption {
	return func(m *Model) {
		if hook != nil {
			m.configHooks = append(m.configHooks, hook)
		}
	}
}

// SDKClient creates an independent configured SDK client. Calls retain native
// SDK types and bypass GAI preflight, observations, normalization, and retries.
// The SDK's transport/retry behavior and configured HTTP timeout apply.
func (p *Provider) SDKClient(ctx context.Context) (*genai.Client, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return p.getClient(ctx)
}

func (m *Model) generateContentConfig(req ai.AIRequest) (*genai.GenerateContentConfig, error) {
	config, err := buildGenerateContentConfig(req.Copy())
	if err != nil {
		return nil, err
	}
	if config == nil && len(m.configHooks) > 0 {
		config = &genai.GenerateContentConfig{}
	}
	for _, hook := range m.configHooks {
		if err := hook(config); err != nil {
			return nil, err
		}
	}
	return config, nil
}
