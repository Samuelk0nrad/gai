package mistral

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/lace-ai/gai/ai"
)

// ModelOption configures a concrete model at construction.
type ModelOption func(*Model)

// ChatCompletionOptions contains Mistral-specific sampling settings. Pointer
// fields distinguish omission from explicit zero/false. Use NativeClient for
// native messages, hosted tools, other endpoints, or additional provider fields.
type ChatCompletionOptions struct {
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	RandomSeed  *int     `json:"random_seed,omitempty"`
	SafePrompt  *bool    `json:"safe_prompt,omitempty"`
}

// WithChatCompletionOptions snapshots settings for synchronous and streaming
// calls. Later mutation of the supplied values does not affect the model.
func WithChatCompletionOptions(options ChatCompletionOptions) ModelOption {
	options = options.copy()
	return func(m *Model) { m.chatOptions = options.copy() }
}

func (o ChatCompletionOptions) copy() ChatCompletionOptions {
	if o.Temperature != nil {
		v := *o.Temperature
		o.Temperature = &v
	}
	if o.TopP != nil {
		v := *o.TopP
		o.TopP = &v
	}
	if o.RandomSeed != nil {
		v := *o.RandomSeed
		o.RandomSeed = &v
	}
	if o.SafePrompt != nil {
		v := *o.SafePrompt
		o.SafePrompt = &v
	}
	return o
}

func (m *Model) chatCompletionRequest(req ai.AIRequest, streaming bool) (chatCompletionRequest, error) {
	payload, err := buildChatCompletionRequest(req, m.name, streaming)
	payload.ChatCompletionOptions = m.chatOptions.copy()
	return payload, err
}

// NativeClient provides native HTTP access, not an SDK or a normalized response.
// Encode caller-defined typed request structs and decode native response structs,
// or consume raw response bodies/events. Responses are never translated by GAI.
type NativeClient struct {
	baseURL *url.URL
	apiKey  string
	client  *http.Client
}

// NativeClient returns an independent client sharing the provider's transport.
// Requests and redirects are restricted to the configured origin before auth is
// attached. Use context deadlines: native streaming has no whole-body timeout.
// Native calls bypass GAI preflight, observations, normalization, and retries.
func (p *Provider) NativeClient() (*NativeClient, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	base, err := url.Parse(strings.TrimRight(p.baseURL, "/") + "/")
	if err != nil {
		return nil, fmt.Errorf("mistral native base URL: %w", err)
	}
	if (base.Scheme != "http" && base.Scheme != "https") || base.Host == "" || base.User != nil || base.RawQuery != "" || base.Fragment != "" {
		return nil, fmt.Errorf("mistral native base URL must be an HTTP origin with an optional path")
	}
	if p.httpClient == nil {
		return nil, fmt.Errorf("mistral native client: nil HTTP client")
	}
	client := *p.httpClient
	client.Timeout = 0
	originalRedirect := client.CheckRedirect
	native := &NativeClient{baseURL: base, apiKey: p.apiKey, client: &client}
	client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if err := native.checkURL(req.URL); err != nil {
			return err
		}
		if req.Host != "" && !strings.EqualFold(req.Host, base.Host) {
			return fmt.Errorf("mistral native redirect changes host")
		}
		if originalRedirect != nil {
			if err := originalRedirect(req, via); err != nil {
				return err
			}
			// A callback may have changed the destination.
			if err := native.checkURL(req.URL); err != nil {
				return err
			}
			if req.Host != "" && !strings.EqualFold(req.Host, base.Host) {
				return fmt.Errorf("mistral native redirect changes host")
			}
		}
		if len(via) >= 10 {
			return fmt.Errorf("mistral native request stopped after 10 redirects")
		}
		return nil
	}
	return native, nil
}

func (c *NativeClient) checkURL(u *url.URL) error {
	if u == nil || u.User != nil || u.Scheme != c.baseURL.Scheme || !strings.EqualFold(u.Host, c.baseURL.Host) {
		return fmt.Errorf("mistral native request must use the configured origin")
	}
	return nil
}

// Do clones req and its headers, resolves relative URLs against the configured
// base URL, and attaches authentication. The context and body retain standard
// http.Client ownership semantics; callers must close the returned response body.
// HTTP error statuses are returned unchanged for native callers to handle.
func (c *NativeClient) Do(req *http.Request) (*http.Response, error) {
	if c == nil || c.client == nil || c.baseURL == nil {
		return nil, fmt.Errorf("mistral native client is not configured")
	}
	if req == nil || req.URL == nil {
		return nil, fmt.Errorf("mistral native request is nil")
	}
	copy := req.Clone(req.Context())
	copy.URL = c.baseURL.ResolveReference(copy.URL)
	if err := c.checkURL(copy.URL); err != nil {
		return nil, err
	}
	if copy.Host != "" && !strings.EqualFold(copy.Host, c.baseURL.Host) {
		return nil, fmt.Errorf("mistral native request changes host")
	}
	if copy.Header == nil {
		copy.Header = make(http.Header)
	}
	copy.Header.Set("Authorization", "Bearer "+c.apiKey)
	return c.client.Do(copy)
}
