// Package ollama adapts Ollama's native /api/chat protocol to GAI.
package ollama

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

// Provider resolves arbitrary Ollama model names without model discovery.
type Provider struct {
	baseURL     string
	httpClient  *http.Client
	bearerToken string
	debug       gai.ObservationSink
}

var _ ai.Provider = (*Provider)(nil)

// New constructs an Ollama provider. Authentication is optional because local
// Ollama endpoints do not require it by default.
func New(debug gai.ObservationSink, options ...Option) *Provider {
	p := &Provider{
		baseURL:    "http://localhost:11434",
		httpClient: &http.Client{},
		debug:      debug,
	}
	for _, option := range options {
		if option != nil {
			option(p)
		}
	}
	return p
}

func (p *Provider) Name() string { return "ollama" }

func (p *Provider) Validate() error {
	if p == nil {
		return ai.ErrNilProvider
	}
	if strings.TrimSpace(p.baseURL) == "" || p.httpClient == nil {
		return ai.ErrProviderInvalid
	}
	if _, err := p.baseOrigin(); err != nil {
		return fmt.Errorf("%w: %v", ai.ErrProviderInvalid, err)
	}
	return nil
}

func (p *Provider) baseOrigin() (*url.URL, error) {
	base, err := url.Parse(p.baseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid Ollama base URL: %w", err)
	}
	if (base.Scheme != "http" && base.Scheme != "https") || base.Host == "" || base.User != nil || base.RawQuery != "" || base.Fragment != "" {
		return nil, fmt.Errorf("ollama base URL must be an HTTP origin with an optional path")
	}
	return base, nil
}

func (p *Provider) requestClient() (*http.Client, error) {
	base, err := p.baseOrigin()
	if err != nil {
		return nil, err
	}
	client := *p.httpClient
	originalRedirect := client.CheckRedirect
	client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		check := func() error {
			if req == nil || req.URL == nil || !sameOrigin(base, req.URL) {
				return fmt.Errorf("ollama redirect changes origin")
			}
			if req.Host != "" && !sameAuthority(base, req.Host) {
				return fmt.Errorf("ollama redirect changes host")
			}
			return nil
		}
		if err := check(); err != nil {
			return err
		}
		if originalRedirect != nil {
			if err := originalRedirect(req, via); err != nil {
				return err
			}
			// A caller callback may mutate the destination.
			return check()
		}
		if len(via) >= 10 {
			return fmt.Errorf("ollama request stopped after 10 redirects")
		}
		return nil
	}
	return &client, nil
}

func sameOrigin(base, target *url.URL) bool {
	return target != nil && strings.EqualFold(base.Scheme, target.Scheme) && sameAuthority(base, target.Host)
}

func sameAuthority(base *url.URL, host string) bool {
	basePort := base.Port()
	if basePort == "" {
		if base.Scheme == "https" {
			basePort = "443"
		} else {
			basePort = "80"
		}
	}
	target, err := url.Parse("//" + host)
	if err != nil || target.User != nil || target.Hostname() == "" || target.Path != "" || target.RawQuery != "" || target.Fragment != "" {
		return false
	}
	targetPort := target.Port()
	if targetPort == "" {
		if base.Scheme == "https" {
			targetPort = "443"
		} else {
			targetPort = "80"
		}
	}
	return strings.EqualFold(base.Hostname(), target.Hostname()) && basePort == targetPort
}

func (p *Provider) Model(name string) (ai.Model, error) {
	return p.TypedModel(name)
}

// TypedModel resolves any non-empty Ollama model name. It never performs
// discovery or a network request.
func (p *Provider) TypedModel(name string, options ...ModelOption) (*Model, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, ai.ErrModelNotFound
	}
	model := &Model{name: name, provider: p, toolSupport: ai.FeatureSupportUnknown}
	for _, option := range options {
		if option != nil {
			option(model)
		}
	}
	return model, nil
}
