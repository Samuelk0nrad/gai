// Package anthropic implements GAI models using Anthropic's Messages API.
package anthropic

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	antropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/internal/modelcatalog"
	"github.com/lace-ai/gai/internal/syncutil"
)

const (
	apiVersion            = "2023-06-01"
	modelDiscoveryTimeout = 10 * time.Second
)

var ErrInvalidAPIKey = errors.New("invalid API key")

// Provider is an Anthropic Messages API provider.
type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	debug      gai.ObservationSink
	catalog    modelcatalog.ModelCatalogCache
	catalogMu  syncutil.ContextMutex
}

var _ ai.Provider = (*Provider)(nil)
var _ ai.ModelCatalogProvider = (*Provider)(nil)

// New creates an Anthropic provider using apiKey.
func New(apiKey string, debug gai.ObservationSink, options ...Option) *Provider {
	p := &Provider{apiKey: strings.TrimSpace(apiKey), baseURL: "https://api.anthropic.com", httpClient: &http.Client{}, debug: debug}
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

func (p *Provider) Name() string { return "anthropic" }

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
	model := &Model{name: name, client: p, debug: p.debug}
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
	client := p.sdkClient()
	pager := client.Models.ListAutoPaging(ctx, antropic.ModelListParams{})
	descriptors := make([]ai.ModelDescriptor, 0)
	for pager.Next() {
		model := pager.Current()
		if name := strings.TrimSpace(model.ID); name != "" {
			descriptors = append(descriptors, anthropicCatalogFacts(name, model))
		}
	}
	if err := pager.Err(); err != nil {
		return nil, err
	}
	return descriptors, nil
}

func (p *Provider) fallbackDescriptors() []ai.ModelDescriptor {
	descriptors := make([]ai.ModelDescriptor, 0, len(models))
	for _, model := range models {
		descriptors = append(descriptors, effectiveAnthropicDescriptor(model, ai.ModelDescriptor{Model: model}))
	}
	return descriptors
}

func (p *Provider) effectiveDescriptors(facts []ai.ModelDescriptor) []ai.ModelDescriptor {
	descriptors := make([]ai.ModelDescriptor, 0, len(facts))
	for _, fact := range facts {
		descriptors = append(descriptors, effectiveAnthropicDescriptor(fact.Model, fact))
	}
	return descriptors
}

func anthropicCatalogFacts(name string, model antropic.ModelInfo) ai.ModelDescriptor {
	facts := ai.ModelDescriptor{Model: name}
	capabilities := model.Capabilities
	if capabilities.JSON.Thinking.Valid() && capabilities.Thinking.JSON.Supported.Valid() {
		facts.Reasoning = featureSupport(capabilities.Thinking.Supported)
	}
	if capabilities.JSON.Effort.Valid() && capabilities.Effort.JSON.Supported.Valid() {
		facts.ReasoningEffort = featureSupport(capabilities.Effort.Supported)
		efforts := []struct {
			value   ai.ReasoningEffort
			present bool
			support bool
		}{
			{ai.ReasoningEffortLow, capabilities.Effort.JSON.Low.Valid() && capabilities.Effort.Low.JSON.Supported.Valid(), capabilities.Effort.Low.Supported},
			{ai.ReasoningEffortMedium, capabilities.Effort.JSON.Medium.Valid() && capabilities.Effort.Medium.JSON.Supported.Valid(), capabilities.Effort.Medium.Supported},
			{ai.ReasoningEffortHigh, capabilities.Effort.JSON.High.Valid() && capabilities.Effort.High.JSON.Supported.Valid(), capabilities.Effort.High.Supported},
		}
		allPresent := true
		for _, effort := range efforts {
			allPresent = allPresent && effort.present
		}
		// A partial provider response cannot prove omitted effort levels are
		// unsupported, so preserve Unknown by publishing no restrictive list.
		if allPresent {
			for _, effort := range efforts {
				if effort.support {
					facts.ReasoningEfforts = append(facts.ReasoningEfforts, effort.value)
				}
			}
		}
	}
	if capabilities.JSON.StructuredOutputs.Valid() && capabilities.StructuredOutputs.JSON.Supported.Valid() {
		support := featureSupport(capabilities.StructuredOutputs.Supported)
		facts.JSONOutput = support
		facts.JSONSchemaOutput = support
	}
	return facts
}

func featureSupport(supported bool) ai.FeatureSupport {
	if supported {
		return ai.FeatureSupportSupported
	}
	return ai.FeatureSupportUnsupported
}
