package ai

import "fmt"

// Provider resolves models made available by an AI service.
//
// Named model lookup does not require discovery or validation methods.
// Providers can implement ProviderValidator, ModelLister, and
// ModelCatalogProvider to opt into those capabilities. Built-in providers
// retain these additional methods on their concrete types.
type Provider interface {
	// Name returns the stable name used to identify the provider.
	Name() string
	// Model returns the model with the given provider-specific name.
	// It must not depend on model discovery being available.
	Model(name string) (Model, error)
}

// ProviderValidator is an optional provider capability used by
// ModelRepository.RegisterProvider. An error prevents registration.
//
// Providers without this capability are responsible for reporting invalid
// configuration when Model or generation is called.
type ProviderValidator interface {
	// Validate checks whether the provider is configured and usable.
	Validate() error
}

// ModelLister is an optional provider capability for model-name discovery.
// ModelRepository prefers ModelCatalogProvider when both are implemented,
// so discovery can use the caller's context and expose model descriptors.
//
// An empty successful list means discovery is supported but found no models.
// ModelLister has no context parameter; repositories can check cancellation
// before and after ListModels, but cannot interrupt a call in progress.
type ModelLister interface {
	// ListModels returns the provider-specific names of available models.
	ListModels() ([]string, error)
}

// UnsupportedModelDiscoveryError identifies a registered provider that has
// neither ModelCatalogProvider nor ModelLister. Repository listing methods
// return this error instead of silently omitting the provider. Model lookup
// remains available through ModelRepository.GetModel.
//
// Use errors.As to inspect Provider and errors.Is with ErrUnsupportedCapability
// to identify unsupported discovery without depending on the error message.
type UnsupportedModelDiscoveryError struct {
	Provider string
}

func (e *UnsupportedModelDiscoveryError) Error() string {
	return fmt.Sprintf("%v: provider %q does not support model discovery", ErrUnsupportedCapability, e.Provider)
}

func (e *UnsupportedModelDiscoveryError) Unwrap() error { return ErrUnsupportedCapability }
