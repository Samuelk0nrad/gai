package ollama

import (
	"net/http"
	"strings"

	"github.com/lace-ai/gai/ai"
)

// Option configures a provider at construction.
type Option func(*Provider)

// WithBaseURL sets the Ollama API origin and optional path prefix.
func WithBaseURL(baseURL string) Option {
	return func(p *Provider) { p.baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/") }
}

// WithHTTPClient uses a shallow copy of client. The transport and callbacks
// remain caller-owned and must be safe for concurrent use.
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		if client != nil {
			copy := *client
			p.httpClient = &copy
		}
	}
}

// WithBearerToken configures optional bearer authentication for protected
// Ollama-compatible endpoints.
func WithBearerToken(token string) Option {
	return func(p *Provider) { p.bearerToken = strings.TrimSpace(token) }
}

// Options contains native Ollama generation settings not represented by
// ai.AIRequest. Pointer fields distinguish omission from an explicit zero.
type Options struct {
	NumCtx      *int     `json:"num_ctx,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	Seed        *int     `json:"seed,omitempty"`
}

func (o Options) copy() Options {
	o.NumCtx = clonePtr(o.NumCtx)
	o.Temperature = clonePtr(o.Temperature)
	o.TopP = clonePtr(o.TopP)
	o.Seed = clonePtr(o.Seed)
	return o
}

func clonePtr[T any](value *T) *T {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

// ModelOption configures an immutable, shareable model value.
type ModelOption func(*Model)

// WithOptions snapshots native Ollama settings.
func WithOptions(options Options) ModelOption {
	options = options.copy()
	return func(model *Model) { model.options = options.copy() }
}

// WithToolSupport records whether the selected model is known to support
// native tool calling. Unknown is the safe default for arbitrary model names.
func WithToolSupport(support ai.FeatureSupport) ModelOption {
	return func(model *Model) { model.toolSupport = support }
}

// With returns an independent model copy with additional options applied.
func (m *Model) With(options ...ModelOption) *Model {
	derived := *m
	derived.options = m.options.copy()
	for _, option := range options {
		if option != nil {
			option(&derived)
		}
	}
	return &derived
}
