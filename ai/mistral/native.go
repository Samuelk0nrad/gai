package mistral

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"strings"

	"github.com/lace-ai/gai/ai"
)

// ModelOption configures a concrete model at construction.
type ModelOption func(*Model)

// ChatCompletionOptions contains typed Mistral chat controls that the portable
// ai.AIRequest does not model. Pointer fields distinguish omission from an
// explicit zero/false value; a nil Stop slice or Prediction is omitted.
//
// Precedence: the portable request exclusively owns model, messages,
// max_tokens, tools, tool_choice, response_format, reasoning_effort, and
// streaming. These options have no fields for those controls, so they can
// neither override nor be overridden by portable settings. Options are
// validated before every request; invalid values fail without a provider call.
//
// Use NativeClient for native messages, hosted tools, multiple completions,
// prompt_mode, guardrails, service_tier, metadata, or other endpoints.
type ChatCompletionOptions struct {
	// Temperature is the sampling temperature. It must be non-negative.
	Temperature *float64 `json:"temperature,omitempty"`
	// TopP is the nucleus-sampling probability mass, between 0 and 1.
	TopP *float64 `json:"top_p,omitempty"`
	// RandomSeed makes sampling deterministic across calls. It must be non-negative.
	RandomSeed *int `json:"random_seed,omitempty"`
	// SafePrompt injects Mistral's safety prompt before the conversation.
	SafePrompt *bool `json:"safe_prompt,omitempty"`
	// Stop ends generation at any of these sequences. Entries must be non-empty.
	Stop []string `json:"stop,omitempty"`
	// PresencePenalty penalizes words or phrases that already appeared.
	PresencePenalty *float64 `json:"presence_penalty,omitempty"`
	// FrequencyPenalty penalizes words in proportion to their frequency.
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	// ParallelToolCalls controls whether the model may call several tools in
	// one turn. Mistral's default is true; set false to request one call.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`
	// Prediction supplies expected output to speed up generation.
	Prediction *Prediction `json:"prediction,omitempty"`
	// PromptCacheKey groups requests that share a prompt prefix for caching.
	PromptCacheKey *string `json:"prompt_cache_key,omitempty"`
}

// Prediction is Mistral's predicted-output control.
type Prediction struct {
	// Content is the expected completion. It must be non-empty.
	Content string
}

// MarshalJSON encodes the documented {"type":"content"} prediction shape.
func (p Prediction) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type    string `json:"type"`
		Content string `json:"content"`
	}{"content", p.Content})
}

// UnmarshalJSON decodes the documented prediction shape.
func (p *Prediction) UnmarshalJSON(data []byte) error {
	var wire struct {
		Type    string `json:"type"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	if wire.Type != "" && wire.Type != "content" {
		return fmt.Errorf("mistral prediction type %q", wire.Type)
	}
	p.Content = wire.Content
	return nil
}

// ErrInvalidChatCompletionOptions reports invalid typed Mistral options.
var ErrInvalidChatCompletionOptions = errors.New("invalid mistral chat completion options")

// Validate checks option values that can be verified locally.
func (o ChatCompletionOptions) Validate() error {
	invalid := func(format string, args ...any) error {
		return fmt.Errorf("%w: %s", ErrInvalidChatCompletionOptions, fmt.Sprintf(format, args...))
	}
	finite := func(name string, v *float64) error {
		if v != nil && (math.IsNaN(*v) || math.IsInf(*v, 0)) {
			return invalid("%s must be finite", name)
		}
		return nil
	}
	for _, field := range []struct {
		name  string
		value *float64
	}{{"temperature", o.Temperature}, {"top_p", o.TopP}, {"presence_penalty", o.PresencePenalty}, {"frequency_penalty", o.FrequencyPenalty}} {
		if err := finite(field.name, field.value); err != nil {
			return err
		}
	}
	if o.Temperature != nil && *o.Temperature < 0 {
		return invalid("temperature must be non-negative")
	}
	if o.TopP != nil && (*o.TopP < 0 || *o.TopP > 1) {
		return invalid("top_p must be between 0 and 1")
	}
	if o.RandomSeed != nil && *o.RandomSeed < 0 {
		return invalid("random_seed must be non-negative")
	}
	for i, stop := range o.Stop {
		if stop == "" {
			return invalid("stop[%d] is empty", i)
		}
	}
	if o.Prediction != nil && o.Prediction.Content == "" {
		return invalid("prediction content is empty")
	}
	if o.PromptCacheKey != nil && strings.TrimSpace(*o.PromptCacheKey) == "" {
		return invalid("prompt_cache_key is empty")
	}
	return nil
}

// WithChatCompletionOptions snapshots settings for synchronous and streaming
// calls. Later mutation of the supplied values does not affect the model.
// Invalid values are reported by Generate and GenerateStream.
func WithChatCompletionOptions(options ChatCompletionOptions) ModelOption {
	options = options.copy()
	return func(m *Model) { m.chatOptions = options.copy() }
}

// With returns an independent copy of m with options applied, sharing the
// provider. m is not modified. WithChatCompletionOptions replaces the whole
// option set, so build per-call options from the complete desired values.
func (m *Model) With(options ...ModelOption) *Model {
	derived := *m
	derived.chatOptions = m.chatOptions.copy()
	for _, option := range options {
		if option != nil {
			option(&derived)
		}
	}
	return &derived
}

func (o ChatCompletionOptions) copy() ChatCompletionOptions {
	o.Temperature = clonePtr(o.Temperature)
	o.TopP = clonePtr(o.TopP)
	o.RandomSeed = clonePtr(o.RandomSeed)
	o.SafePrompt = clonePtr(o.SafePrompt)
	o.PresencePenalty = clonePtr(o.PresencePenalty)
	o.FrequencyPenalty = clonePtr(o.FrequencyPenalty)
	o.ParallelToolCalls = clonePtr(o.ParallelToolCalls)
	o.Prediction = clonePtr(o.Prediction)
	o.PromptCacheKey = clonePtr(o.PromptCacheKey)
	o.Stop = append([]string(nil), o.Stop...)
	return o
}

func clonePtr[T any](v *T) *T {
	if v == nil {
		return nil
	}
	copy := *v
	return &copy
}

func (m *Model) chatCompletionRequest(req ai.AIRequest, streaming bool) (chatCompletionRequest, error) {
	options := m.chatOptions.copy()
	if err := options.Validate(); err != nil {
		return chatCompletionRequest{}, err
	}
	payload, err := buildChatCompletionRequest(req, m.name, streaming)
	if err != nil {
		return chatCompletionRequest{}, err
	}
	payload.ChatCompletionOptions = options
	return payload, nil
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
// Redirect callbacks decide whether to proceed; accepted redirects always use
// this client's provider authentication, even if a callback changes the headers.
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
		if req.Host != "" && !native.sameAuthority(req.Host) {
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
			if req.Host != "" && !native.sameAuthority(req.Host) {
				return fmt.Errorf("mistral native redirect changes host")
			}
		}
		if len(via) >= 10 {
			return fmt.Errorf("mistral native request stopped after 10 redirects")
		}
		// net/http may strip sensitive headers before CheckRedirect, including
		// for equivalent authorities on older Go versions. Restore only our
		// provider credential, and only after every destination check succeeds.
		if req.Header == nil {
			req.Header = make(http.Header)
		}
		req.Header.Set("Authorization", "Bearer "+native.apiKey)
		return nil
	}
	return native, nil
}

// sameAuthority permits omission of the configured scheme's default port.
// Compare the remaining authority verbatim (ignoring case), rather than parsing
// a Host override as a URL: userinfo, paths, or other suffixes must not disappear
// during normalization. Nondefault ports and IPv6 brackets remain significant.
func (c *NativeClient) sameAuthority(host string) bool {
	defaultPort := ":80"
	if c.baseURL.Scheme == "https" {
		defaultPort = ":443"
	}
	return strings.EqualFold(strings.TrimSuffix(host, defaultPort), strings.TrimSuffix(c.baseURL.Host, defaultPort))
}

func (c *NativeClient) checkURL(u *url.URL) error {
	if u == nil || u.User != nil || u.Scheme != c.baseURL.Scheme || !c.sameAuthority(u.Host) {
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
	if copy.Host != "" && !c.sameAuthority(copy.Host) {
		return nil, fmt.Errorf("mistral native request changes host")
	}
	if copy.Header == nil {
		copy.Header = make(http.Header)
	}
	copy.Header.Set("Authorization", "Bearer "+c.apiKey)
	return c.client.Do(copy)
}
