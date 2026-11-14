// Package anthropic implements GAI models using Anthropic's Messages API.
package anthropic

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/modelcatalog"
	"github.com/Samuelk0nrad/gai/internal/syncutil"
	antropic "github.com/anthropics/anthropic-sdk-go"
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
	// catalogThinking holds the thinking modes reported by the live catalog.
	catalogThinking atomic.Pointer[map[string]thinkingMode]
	// promptCache is the top-level cache_control sent when caching is on.
	promptCache *antropic.CacheControlEphemeralParam
	// fallbacks is the provider-wide server-side fallbacks value, or nil.
	fallbacks any
	// promptCacheErr reports an invalid WithPromptCaching TTL from Validate.
	promptCacheErr error
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
	return p.promptCacheErr
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
	if model.fallbacksErr != nil {
		return nil, model.fallbacksErr
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
	discovered, thinking, err := p.listModelCatalog(discoveryCtx)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return p.fallbackDescriptors(), nil
	}
	p.catalogThinking.Store(&thinking)
	p.catalog.Replace(discovered)
	return p.effectiveDescriptors(discovered), nil
}

func (p *Provider) listModelCatalog(ctx context.Context) ([]ai.ModelDescriptor, map[string]thinkingMode, error) {
	client := p.sdkClient()
	pager := client.Models.ListAutoPaging(ctx, antropic.ModelListParams{})
	descriptors := make([]ai.ModelDescriptor, 0)
	thinking := map[string]thinkingMode{}
	for pager.Next() {
		model := pager.Current()
		if name := strings.TrimSpace(model.ID); name != "" {
			descriptors = append(descriptors, anthropicCatalogFacts(name, model))
			if mode := anthropicCatalogThinking(model); mode != thinkingUnknown {
				thinking[name] = mode
			}
		}
	}
	if err := pager.Err(); err != nil {
		return nil, nil, err
	}
	return descriptors, thinking, nil
}

// thinkingMode returns the thinking configuration model accepts. The live
// catalog wins over the local table, and unknown models report thinkingUnknown.
func (p *Provider) thinkingMode(model string) thinkingMode {
	if modes := p.catalogThinking.Load(); modes != nil {
		if mode, ok := (*modes)[model]; ok {
			return mode
		}
	}
	if facts, ok := lookupModelFacts(model); ok {
		return facts.thinking
	}
	return thinkingUnknown
}

// anthropicCatalogThinking reads the accepted thinking types. A response that
// omits either type cannot prove the other is the only one.
func anthropicCatalogThinking(model antropic.ModelInfo) thinkingMode {
	capabilities := model.Capabilities
	if !capabilities.JSON.Thinking.Valid() || !capabilities.Thinking.JSON.Types.Valid() {
		return thinkingUnknown
	}
	types := capabilities.Thinking.Types
	if !types.JSON.Adaptive.Valid() || !types.Adaptive.JSON.Supported.Valid() || !types.JSON.Enabled.Valid() || !types.Enabled.JSON.Supported.Valid() {
		return thinkingUnknown
	}
	switch adaptive, enabled := types.Adaptive.Supported, types.Enabled.Supported; {
	case adaptive && enabled:
		return thinkingAdaptiveOrBudget
	case adaptive:
		return thinkingAdaptive
	case enabled:
		return thinkingBudget
	default:
		return thinkingUnknown
	}
}

func (p *Provider) fallbackDescriptors() []ai.ModelDescriptor {
	descriptors := make([]ai.ModelDescriptor, 0, len(models))
	for _, model := range models {
		descriptors = append(descriptors, effectiveAnthropicDescriptor(model, ai.ModelDescriptor{Model: model}, p.thinkingMode(model)))
	}
	return descriptors
}

func (p *Provider) effectiveDescriptors(facts []ai.ModelDescriptor) []ai.ModelDescriptor {
	descriptors := make([]ai.ModelDescriptor, 0, len(facts))
	for _, fact := range facts {
		descriptors = append(descriptors, effectiveAnthropicDescriptor(fact.Model, fact, p.thinkingMode(fact.Model)))
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
			{ai.ReasoningEffortXHigh, capabilities.Effort.JSON.Xhigh.Valid() && capabilities.Effort.Xhigh.JSON.Supported.Valid(), capabilities.Effort.Xhigh.Supported},
			{ai.ReasoningEffortMax, capabilities.Effort.JSON.Max.Valid() && capabilities.Effort.Max.JSON.Supported.Valid(), capabilities.Effort.Max.Supported},
		}
		// low, medium and high must all be present: a partial response cannot
		// prove an omitted level is unsupported, so no restrictive list is
		// published. xhigh and max were added later; a response that omits
		// them predates them, so they count only when reported as supported.
		const baseLevels = 3
		basePresent := true
		for _, effort := range efforts[:baseLevels] {
			basePresent = basePresent && effort.present
		}
		if basePresent {
			for _, effort := range efforts {
				if effort.present && effort.support {
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
