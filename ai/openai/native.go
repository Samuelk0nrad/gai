package openai

import (
	"github.com/lace-ai/gai/ai"
	sdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

// ModelOption configures a concrete model at construction.
type ModelOption func(*Model)

// WithChatCompletionParams appends a hook run after portable request mapping.
// Hooks receive fresh parameters on each call, must be concurrency-safe, and
// must not retain parameter pointers. Captured mutable data is caller-owned.
// Model identity and streaming usage are reserved by GAI. Changes to portable
// messages/tools must retain the agent's protocol; use SDKClient for native events.
func WithChatCompletionParams(hook func(*sdk.ChatCompletionNewParams) error) ModelOption {
	return func(m *Model) {
		if hook != nil {
			m.chatHooks = append(m.chatHooks, hook)
		}
	}
}

// WithResponsesParams is the Responses-transport counterpart of
// WithChatCompletionParams. It has the same ownership and concurrency contract.
func WithResponsesParams(hook func(*responses.ResponseNewParams) error) ModelOption {
	return func(m *Model) {
		if hook != nil {
			m.responsesHooks = append(m.responsesHooks, hook)
		}
	}
}

// SDKClient returns an independent SDK client with this provider's endpoint,
// credentials, and transport. Native calls retain SDK types and bypass GAI
// preflight, observations, normalization, and retries. SDK retries are disabled.
// Set a context deadline: the native client has no whole-response timeout.
func (p *Provider) SDKClient() (*sdk.Client, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return p.sdkClient(true), nil
}

func (p *Provider) sdkClient(streaming bool) *sdk.Client {
	httpClient := p.httpClient
	if streaming {
		httpClient = p.streamingHTTPClient()
	}
	client := sdk.NewClient(option.WithAPIKey(p.apiKey), option.WithBaseURL(p.baseURL), option.WithHTTPClient(httpClient), option.WithMaxRetries(0))
	return &client
}

func (m *Model) chatCompletionParams(req ai.AIRequest, streaming bool) (sdk.ChatCompletionNewParams, error) {
	params, err := buildChatCompletionParams(m.name, req.Copy(), streaming)
	if err != nil {
		return params, err
	}
	for _, hook := range m.chatHooks {
		if err := hook(&params); err != nil {
			return params, err
		}
	}
	params.Model = shared.ChatModel(m.name)
	if streaming {
		params.StreamOptions.IncludeUsage = param.NewOpt(true)
	} else {
		params.StreamOptions = sdk.ChatCompletionStreamOptionsParam{}
	}
	return params, nil
}

func (m *Model) responsesParams(req ai.AIRequest) (responses.ResponseNewParams, error) {
	params, err := buildResponsesParams(m.name, req.Copy())
	if err != nil {
		return params, err
	}
	for _, hook := range m.responsesHooks {
		if err := hook(&params); err != nil {
			return params, err
		}
	}
	params.Model = shared.ResponsesModel(m.name)
	return params, nil
}
