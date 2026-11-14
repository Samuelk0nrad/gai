package anthropic

import (
	"errors"
	"fmt"
	"net/http"
	"strings"

	antropic "github.com/anthropics/anthropic-sdk-go"
)

// Option configures a provider at construction.
type Option func(*Provider)

// WithBaseURL sets the endpoint used by discovery, portable generation, and
// native access. Use a trusted endpoint; requests carry the provider API key.
func WithBaseURL(baseURL string) Option { return func(p *Provider) { p.baseURL = baseURL } }

// WithHTTPClient uses a shallow copy of client. Its transport and callbacks
// remain caller-owned and must be safe for concurrent use. Nil keeps the default.
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		if client != nil {
			copy := *client
			p.httpClient = &copy
		}
	}
}

// PromptCacheTTL is the lifetime of a prompt cache entry.
type PromptCacheTTL string

const (
	// PromptCacheTTL5m is the default five-minute cache lifetime.
	PromptCacheTTL5m PromptCacheTTL = "5m"
	// PromptCacheTTL1h is the one-hour cache lifetime, written at a higher price.
	PromptCacheTTL1h PromptCacheTTL = "1h"
)

// WithPromptCaching turns on automatic prompt caching for every generation and
// token count request. GAI sets the top-level cache_control field, so the API
// places one breakpoint on the last cacheable block and moves it forward as a
// conversation grows. In a tool loop each iteration then reads the tools,
// system prompt and earlier history from the cache. An empty ttl selects
// PromptCacheTTL5m. Any other value than the PromptCacheTTL constants makes
// Validate fail. The last WithPromptCaching option wins.
//
// The automatic breakpoint counts toward the API's limit of four breakpoints
// per request, so WithMessageParams hooks may add at most three more.
// Prefixes shorter than the model's minimum cacheable length are not cached.
// Check Usage.CachedTokens and Usage.CacheCreationTokens to confirm hits.
func WithPromptCaching(ttl PromptCacheTTL) Option {
	return func(p *Provider) {
		p.promptCache, p.promptCacheErr = nil, nil
		switch ttl {
		case "", PromptCacheTTL5m:
			p.promptCache = &antropic.CacheControlEphemeralParam{TTL: antropic.CacheControlEphemeralTTLTTL5m}
		case PromptCacheTTL1h:
			p.promptCache = &antropic.CacheControlEphemeralParam{TTL: antropic.CacheControlEphemeralTTLTTL1h}
		default:
			p.promptCacheErr = fmt.Errorf("anthropic: unsupported prompt cache TTL %q", ttl)
		}
	}
}

// WithRefusalFallbacks turns on server-side refusal fallbacks, using the
// "default" fallback mode, for the provider's models that have them: Opus 5,
// Opus 5.5, Sonnet 5.5, Fable 5.1, Fable 5.2, Mythos 5.1 and Mythos 5.2. Other
// models, including models outside GAI's catalog, send no fallbacks unless
// WithModelRefusalFallbacks is set. When a model's safety
// classifiers decline a request, the API re-serves it on Anthropic's
// recommended fallback model within the same call, routed by refusal category.
// The response then names the fallback model.
//
// This is a beta feature (header server-side-fallback-2026-07-01), and not
// every platform offers it. Use WithModelRefusalFallbacks to name fallback
// models for one model, and WithoutRefusalFallbacks to turn them off for one. A response that still ends in a refusal returns a
// *RefusalError.
func WithRefusalFallbacks() Option {
	return func(p *Provider) { p.fallbacks = defaultFallbacks }
}

// WithModelRefusalFallbacks turns on server-side refusal fallbacks for one
// model, including a model GAI has no facts for, and overrides
// WithRefusalFallbacks. Each name must be in the requested
// model's allowed fallback models, or the API rejects the request. With no
// names the model uses the "default" fallback mode. An empty name makes
// TypedModel fail. The last fallbacks option wins.
func WithModelRefusalFallbacks(models ...string) ModelOption {
	return func(m *Model) {
		m.fallbacksSet, m.fallbacksErr = true, nil
		if len(models) == 0 {
			m.fallbacks = defaultFallbacks
			return
		}
		entries := make([]map[string]string, 0, len(models))
		for _, model := range models {
			model = strings.TrimSpace(model)
			if model == "" {
				m.fallbacks, m.fallbacksErr = nil, errors.New("anthropic: empty refusal fallback model name")
				return
			}
			entries = append(entries, map[string]string{"model": model})
		}
		m.fallbacks = entries
	}
}

// WithoutRefusalFallbacks turns server-side refusal fallbacks off for one
// model, overriding WithRefusalFallbacks. A refusal then returns a
// *RefusalError instead of being re-served by another model. The last
// fallbacks option wins.
func WithoutRefusalFallbacks() ModelOption {
	return func(m *Model) {
		m.fallbacks, m.fallbacksSet, m.fallbacksErr = nil, true, nil
	}
}

const (
	defaultFallbacks    = "default"
	fallbackBetaHeader  = "server-side-fallback-2026-07-01"
	fallbacksBodyField  = "fallbacks"
	anthropicBetaHeader = "anthropic-beta"
)
