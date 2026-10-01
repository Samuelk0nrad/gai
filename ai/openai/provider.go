package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/internal/modelcatalog"
	"github.com/lace-ai/gai/internal/syncutil"
)

const modelDiscoveryTimeout = 10 * time.Second

// Provider resolves models served by OpenAI.
type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	debug      gai.ObservationSink
	catalog    modelcatalog.ModelCatalogCache
	catalogMu  syncutil.ContextMutex
	transport  Transport
}

// Transport selects the OpenAI API endpoint used by models from this provider.
type Transport string

const (
	TransportChatCompletions Transport = "chat_completions"
	TransportResponses       Transport = "responses"
)

// Option configures an OpenAI provider.
type Option func(*Provider)

// WithResponsesTransport selects OpenAI's Responses API. It supports native
// function-call/result history and reasoning models that require this endpoint.
func WithResponsesTransport() Option {
	return func(p *Provider) { p.transport = TransportResponses }
}

var _ ai.Provider = (*Provider)(nil)
var _ ai.ModelCatalogProvider = (*Provider)(nil)

func New(apiKey string, debug gai.ObservationSink, options ...Option) *Provider {
	p := &Provider{
		apiKey:     apiKey,
		baseURL:    "https://api.openai.com/v1",
		httpClient: &http.Client{},
		debug:      debug,
		transport:  TransportChatCompletions,
	}
	for _, option := range options {
		if option != nil {
			option(p)
		}
	}
	return p
}

func (p *Provider) Validate() error {
	if p == nil {
		return ai.ErrNilProvider
	}
	if strings.TrimSpace(p.apiKey) == "" {
		return ErrInvalidAPIKey
	}
	return nil
}

func (p *Provider) Name() string { return "openai" }

// streamingHTTPClient returns a client whose request lifetime is controlled by
// the caller's context, rather than the provider's non-streaming timeout.
func (p *Provider) streamingHTTPClient() *http.Client {
	client := *p.httpClient
	client.Timeout = 0
	return &client
}

func (p *Provider) Model(name string) (ai.Model, error) {
	model, err := p.TypedModel(name)
	if err != nil {
		return nil, err
	}
	return model, nil
}

// TypedModel resolves a concrete model without discovery. Options are applied
// once at construction; configured models may be shared between runs.
func (p *Provider) TypedModel(name string, options ...ModelOption) (*Model, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, ai.ErrModelNotFound
	}
	model := &Model{name: name, provider: p}
	for _, option := range options {
		if option != nil {
			option(model)
		}
	}
	return model, nil
}

func (p *Provider) ListModels() ([]string, error) {
	descriptors, err := p.ListModelDescriptors(context.Background())
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(descriptors))
	for _, descriptor := range descriptors {
		names = append(names, descriptor.Model)
	}
	return names, nil
}

// ListModelDescriptors returns a cached point-in-time model catalog. Discovery
// is the only capability path that may perform network I/O.
func (p *Provider) ListModelDescriptors(ctx context.Context) ([]ai.ModelDescriptor, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := p.catalogMu.Lock(ctx); err != nil {
		return nil, err
	}
	defer p.catalogMu.Unlock()
	if cached, ok := p.catalog.Load(); ok {
		return p.effectiveDescriptors(cached), nil
	}
	discoveryCtx, cancel := context.WithTimeout(ctx, modelDiscoveryTimeout)
	defer cancel()
	discovered, err := p.listModelCatalog(discoveryCtx)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return p.fallbackDescriptors(), nil
	}
	p.catalog.Replace(discovered)
	return p.effectiveDescriptors(discovered), nil
}

func (p *Provider) listModelCatalog(ctx context.Context) ([]ai.ModelDescriptor, error) {
	if p.httpClient == nil {
		return nil, fmt.Errorf("openai model discovery: nil HTTP client")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(p.baseURL, "/")+"/models", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	res, err := p.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode < http.StatusOK || res.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("openai model discovery: unexpected status %s", res.Status)
	}

	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(res.Body).Decode(&payload); err != nil {
		return nil, err
	}

	descriptors := make([]ai.ModelDescriptor, 0, len(payload.Data))
	for _, model := range payload.Data {
		if name := strings.TrimSpace(model.ID); isChatCapableModel(name) {
			descriptors = append(descriptors, ai.ModelDescriptor{Model: name})
		}
	}
	return descriptors, nil
}

// isChatCapableModel excludes model families that the Models endpoint lists but
// that cannot be used with this provider's Chat Completions adapter. The
// endpoint does not expose per-endpoint capabilities, so unknown IDs remain
// discoverable for compatible custom and future chat models.
func isChatCapableModel(name string) bool {
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" {
		return false
	}
	for _, prefix := range []string{
		"dall-e-",
		"gpt-image-",
		"text-embedding-",
		"text-moderation-",
		"omni-moderation-",
		"whisper-",
		"tts-",
		"sora-",
		"computer-use-",
	} {
		if strings.HasPrefix(name, prefix) {
			return false
		}
	}
	return name != "gpt-5.6-auto"
}

func (p *Provider) fallbackDescriptors() []ai.ModelDescriptor {
	descriptors := make([]ai.ModelDescriptor, 0, len(models))
	for _, model := range models {
		descriptors = append(descriptors, openAIDescriptor(model))
	}
	return descriptors
}

func (p *Provider) effectiveDescriptors(facts []ai.ModelDescriptor) []ai.ModelDescriptor {
	descriptors := make([]ai.ModelDescriptor, 0, len(facts))
	for _, fact := range facts {
		descriptors = append(descriptors, effectiveOpenAIDescriptor(fact.Model, fact))
	}
	return descriptors
}

func effectiveOpenAIDescriptor(model string, catalog ai.ModelDescriptor) ai.ModelDescriptor {
	// The adapter descriptor is the maintained baseline. OpenAI's Models API
	// currently returns model IDs only, so omitted remote facts must retain that
	// baseline rather than downgrading known adapter support to Unknown.
	adapter := openAIDescriptor(model)
	facts := modelcatalog.OverrideModelDescriptor(adapter, catalog)
	return modelcatalog.IntersectModelDescriptors(adapter, facts)
}

func isKnownModel(name string) bool {
	for _, model := range models {
		if name == model {
			return true
		}
	}
	return false
}
