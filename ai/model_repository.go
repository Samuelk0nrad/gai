package ai

import (
	"context"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/lace-ai/gai"
)

// ModelRepository stores providers and resolves their models.
//
// A repository is safe to use only after construction with
// NewModelRepository. It is not safe for concurrent mutation.
type ModelRepository struct {
	providers map[string]Provider
	debug     gai.ObservationSink
}

// NewModelRepository creates an empty provider registry.
//
// When debug is non-nil, repository operations emit diagnostic events.
func NewModelRepository(debug gai.ObservationSink) *ModelRepository {
	return &ModelRepository{
		providers: make(map[string]Provider),
		debug:     debug,
	}
}

// Validate checks whether the repository can be used.
func (r *ModelRepository) Validate() error {
	if r == nil {
		return ErrNilModelRepository
	}
	return nil
}

// RegisterProvider registers provider under Provider.Name, calling Validate
// first when provider implements ProviderValidator. Nil providers, including
// typed nil values, return ErrNilProvider. Empty or whitespace-only names
// return ErrProviderInvalid before optional validation runs.
// It returns ErrProviderAlreadyExists when that name is already registered.
func (r *ModelRepository) RegisterProvider(ctx context.Context, provider Provider) error {
	if err := r.Validate(); err != nil {
		return err
	}
	if isNilProvider(provider) {
		return ErrNilProvider
	}
	name := provider.Name()
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("%w: provider name must not be empty or whitespace", ErrProviderInvalid)
	}
	if validator, ok := provider.(ProviderValidator); ok {
		if err := validator.Validate(); err != nil {
			if r.debug != nil {
				gai.EmitObservation(ctx, r.debug, gai.Observation{
					Name:   "provider_validation_failed",
					Source: "ai:ModelRepository.RegisterProvider",
					Fields: map[string]any{
						"provider_name": name,
						"error":         err.Error(),
					},
					Err: err,
				})
			}
			return err
		}
	}

	_, exists := r.providers[name]
	if exists {
		if r.debug != nil {
			gai.EmitObservation(ctx, r.debug, gai.Observation{
				Name:   "provider_already_registered",
				Source: "ai:ModelRepository.RegisterProvider",
				Fields: map[string]any{
					"provider_name": name,
				},
			})
		}
		return ErrProviderAlreadyExists
	}
	r.providers[name] = provider
	if r.debug != nil {
		gai.EmitObservation(ctx, r.debug, gai.Observation{
			Name:   "provider_registered",
			Source: "ai:ModelRepository.RegisterProvider",
			Fields: map[string]any{
				"provider_name": name,
			},
		})
	}
	return nil
}

// UnregisterProvider removes the named provider.
// It returns ErrProviderNotFound when no such provider is registered.
func (r *ModelRepository) UnregisterProvider(ctx context.Context, providerName string) error {
	if err := r.Validate(); err != nil {
		return err
	}

	_, exists := r.providers[providerName]
	if !exists {
		if r.debug != nil {
			gai.EmitObservation(ctx, r.debug, gai.Observation{
				Name:   "provider_not_found_for_unregister",
				Source: "ai:ModelRepository.UnregisterProvider",
				Fields: map[string]any{
					"provider_name": providerName,
				},
			})
		}
		return ErrProviderNotFound
	}
	delete(r.providers, providerName)
	if r.debug != nil {
		gai.EmitObservation(ctx, r.debug, gai.Observation{
			Name:   "provider_unregistered",
			Source: "ai:ModelRepository.UnregisterProvider",
			Fields: map[string]any{
				"provider_name": providerName,
			},
		})
	}
	return nil
}

// GetModel resolves modelName through the named provider.
// It calls Model directly and never invokes optional discovery capabilities.
func (r *ModelRepository) GetModel(ctx context.Context, providerName, modelName string) (Model, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}

	provider, ok := r.providers[providerName]
	if !ok {
		if r.debug != nil {
			gai.EmitObservation(ctx, r.debug, gai.Observation{
				Name:   "provider_not_found_for_model",
				Source: "ai:ModelRepository.GetModel",
				Fields: map[string]any{
					"provider_name": providerName,
					"model_name":    modelName,
				},
			})
		}
		return nil, ErrProviderNotFound
	}
	if r.debug != nil {
		gai.EmitObservation(ctx, r.debug, gai.Observation{
			Name:   "getting_model",
			Source: "ai:ModelRepository.GetModel",
			Fields: map[string]any{
				"provider_name": providerName,
				"model_name":    modelName,
			},
		})
	}
	return provider.Model(modelName)
}

// ListModels returns all registered models as sorted "provider:model" names.
// It prefers ModelCatalogProvider, deriving names from nonempty descriptor
// Model fields, and otherwise uses ModelLister. If any provider supports
// neither capability, it returns UnsupportedModelDiscoveryError and no partial
// results. An empty successful catalog or list is valid.
//
// Providers are queried in name order. Cancellation is checked between calls;
// only ModelCatalogProvider can honor cancellation during discovery itself.
func (r *ModelRepository) ListModels(ctx context.Context) ([]string, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	var models []string
	for _, name := range r.providerNames() {
		providerModels, err := listProviderModels(ctx, name, r.providers[name])
		if err != nil {
			if r.debug != nil {
				gai.EmitObservation(ctx, r.debug, gai.Observation{
					Name:   "list_provider_models_failed",
					Source: "ai:ModelRepository.ListModels",
					Fields: map[string]any{
						"provider_name": name,
						"error":         err.Error(),
					},
					Err: err,
				})
			}
			return nil, err
		}
		for _, model := range providerModels {
			models = append(models, name+":"+model)
		}
	}
	sort.Strings(models)
	if r.debug != nil {
		gai.EmitObservation(ctx, r.debug, gai.Observation{
			Name:   "models_listed",
			Source: "ai:ModelRepository.ListModels",
			Fields: map[string]any{
				"model_count": len(models),
			},
		})
	}
	return models, nil
}

// ListModelDescriptors returns descriptors sorted by provider and model name.
// It prefers ModelCatalogProvider, copying descriptors with nonempty Model
// fields and assigning their registered provider name. Otherwise it uses
// ModelLister, resolves each listed model, and copies its optional ModelDescriber
// descriptor. Listed models without descriptors are skipped.
//
// If any provider supports neither discovery capability, it returns
// UnsupportedModelDiscoveryError and no partial results. An empty successful
// catalog or list is valid. Providers are queried in name order, and cancellation
// is checked between discovery and model-resolution calls. Only
// ModelCatalogProvider can honor cancellation during discovery itself.
func (r *ModelRepository) ListModelDescriptors(ctx context.Context) ([]ModelDescriptor, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	var descriptors []ModelDescriptor
	for _, providerName := range r.providerNames() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		provider := r.providers[providerName]
		if catalog, ok := provider.(ModelCatalogProvider); ok {
			providerDescriptors, err := catalog.ListModelDescriptors(ctx)
			if err != nil {
				return nil, err
			}
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			for _, providerDescriptor := range providerDescriptors {
				descriptor := providerDescriptor.Copy()
				if descriptor.Model == "" {
					continue
				}
				descriptor.Provider = providerName
				descriptors = append(descriptors, descriptor)
			}
			continue
		}
		models, err := listProviderModels(ctx, providerName, provider)
		if err != nil {
			return nil, err
		}
		for _, name := range models {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			model, err := provider.Model(name)
			if err != nil {
				return nil, err
			}
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			describer, ok := model.(ModelDescriber)
			if !ok {
				continue
			}
			descriptor := describer.Descriptor().Copy()
			descriptor.Provider = providerName
			if descriptor.Model == "" {
				descriptor.Model = name
			}
			descriptors = append(descriptors, descriptor)
		}
	}
	sort.Slice(descriptors, func(i, j int) bool {
		if descriptors[i].Provider == descriptors[j].Provider {
			return descriptors[i].Model < descriptors[j].Model
		}
		return descriptors[i].Provider < descriptors[j].Provider
	})
	return descriptors, nil
}

func (r *ModelRepository) providerNames() []string {
	names := make([]string, 0, len(r.providers))
	for name := range r.providers {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func listProviderModels(ctx context.Context, name string, provider Provider) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var names []string
	if catalog, ok := provider.(ModelCatalogProvider); ok {
		descriptors, err := catalog.ListModelDescriptors(ctx)
		if err != nil {
			return nil, err
		}
		for _, descriptor := range descriptors {
			if descriptor.Model != "" {
				names = append(names, descriptor.Model)
			}
		}
	} else if lister, ok := provider.(ModelLister); ok {
		var err error
		names, err = lister.ListModels()
		if err != nil {
			return nil, err
		}
	} else {
		return nil, &UnsupportedModelDiscoveryError{Provider: name}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return names, nil
}

func isNilProvider(provider Provider) bool {
	if provider == nil {
		return true
	}
	value := reflect.ValueOf(provider)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
