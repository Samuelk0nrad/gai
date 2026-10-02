package mistral

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/internal/modelcatalog"
	"github.com/lace-ai/gai/internal/syncutil"
)

const modelDiscoveryTimeout = 10 * time.Second

type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	debug      gai.ObservationSink
	catalog    modelcatalog.ModelCatalogCache
	catalogMu  syncutil.ContextMutex
	// vision holds discovered image-input support from the same snapshot as
	// catalog. It is written only with catalogMu held, after catalog.Replace.
	visionMu sync.RWMutex
	vision   map[string]bool
}

var _ ai.Provider = (*Provider)(nil)
var _ ai.ModelCatalogProvider = (*Provider)(nil)

func New(apiKey string, debug gai.ObservationSink, options ...Option) *Provider {
	p := &Provider{
		apiKey:  apiKey,
		baseURL: "https://api.mistral.ai",
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
		debug: debug,
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

func (p *Provider) Name() string {
	return "mistral"
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

	modelName := strings.TrimSpace(name)
	if modelName == "" {
		return nil, ai.ErrModelNotFound
	}
	model := &Model{
		name:   modelName,
		client: p,
		debug:  p.debug,
	}
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

// ListModelDescriptors returns a cached point-in-time model catalog.
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
	discovered, vision, err := p.listModelCatalog(discoveryCtx)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return p.fallbackDescriptors(), nil
	}
	p.catalog.Replace(discovered)
	p.visionMu.Lock()
	p.vision = vision
	p.visionMu.Unlock()
	return p.effectiveDescriptors(discovered), nil
}

// visionSupport reports discovered image-input support. It reads only the
// cached snapshot and never triggers discovery.
func (p *Provider) visionSupport(model string) (supported, known bool) {
	p.visionMu.RLock()
	defer p.visionMu.RUnlock()
	supported, known = p.vision[model]
	return supported, known
}

func (p *Provider) listModelCatalog(ctx context.Context) ([]ai.ModelDescriptor, map[string]bool, error) {
	if p.httpClient == nil {
		return nil, nil, fmt.Errorf("mistral model discovery: nil HTTP client")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(p.baseURL, "/")+"/v1/models", nil)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	res, err := p.httpClient.Do(req)
	if err != nil {
		return nil, nil, err
	}
	defer res.Body.Close()
	if res.StatusCode < http.StatusOK || res.StatusCode >= http.StatusMultipleChoices {
		return nil, nil, fmt.Errorf("mistral model discovery: unexpected status %s", res.Status)
	}

	var payload struct {
		Data []struct {
			ID           string `json:"id"`
			Capabilities struct {
				CompletionChat  *bool `json:"completion_chat"`
				FunctionCalling *bool `json:"function_calling"`
				Reasoning       *bool `json:"reasoning"`
				Vision          *bool `json:"vision"`
			} `json:"capabilities"`
		} `json:"data"`
	}
	if err := json.NewDecoder(res.Body).Decode(&payload); err != nil {
		return nil, nil, err
	}

	descriptors := make([]ai.ModelDescriptor, 0, len(payload.Data))
	vision := make(map[string]bool)
	for _, model := range payload.Data {
		name := strings.TrimSpace(model.ID)
		if name == "" || model.Capabilities.CompletionChat != nil && !*model.Capabilities.CompletionChat {
			continue
		}
		facts := ai.ModelDescriptor{Model: name}
		if model.Capabilities.FunctionCalling != nil {
			facts.NativeTools = featureSupport(*model.Capabilities.FunctionCalling)
		}
		// The catalog flag marks reasoning models. Adjustable-reasoning models
		// may report false, so false is not treated as a known limitation, and
		// true says nothing about which reasoning_effort values are accepted.
		if model.Capabilities.Reasoning != nil && *model.Capabilities.Reasoning {
			facts.Reasoning = ai.FeatureSupportSupported
		}
		if model.Capabilities.Vision != nil {
			vision[name] = *model.Capabilities.Vision
		}
		descriptors = append(descriptors, facts)
	}
	return descriptors, vision, nil
}

func (p *Provider) fallbackDescriptors() []ai.ModelDescriptor {
	descriptors := make([]ai.ModelDescriptor, 0, len(models))
	for _, model := range models {
		descriptors = append(descriptors, mistralAdapterDescriptor(model))
	}
	return descriptors
}

func (p *Provider) effectiveDescriptors(facts []ai.ModelDescriptor) []ai.ModelDescriptor {
	descriptors := make([]ai.ModelDescriptor, 0, len(facts))
	for _, fact := range facts {
		descriptors = append(descriptors, effectiveMistralDescriptor(fact.Model, fact))
	}
	return descriptors
}

// effectiveMistralDescriptor lets catalog facts replace static per-model facts,
// then bounds the result by what the adapter implements.
func effectiveMistralDescriptor(model string, catalog ai.ModelDescriptor) ai.ModelDescriptor {
	facts := modelcatalog.OverrideModelDescriptor(mistralAdapterDescriptor(model), catalog)
	return modelcatalog.IntersectModelDescriptors(mistralImplementationDescriptor(model), facts)
}

func featureSupport(supported bool) ai.FeatureSupport {
	if supported {
		return ai.FeatureSupportSupported
	}
	return ai.FeatureSupportUnsupported
}

func containsModel(models []string, name string) bool {
	for _, modelName := range models {
		if modelName == name {
			return true
		}
	}
	return false
}
