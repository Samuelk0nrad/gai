package modelcatalog

import (
	"sync"

	"github.com/lace-ai/gai/ai"
)

// ModelCatalogCache stores the last successful provider catalog snapshot.
// Replacement and reads deep-copy descriptors so callers cannot mutate the
// cached snapshot. Its zero value is ready for use.
type ModelCatalogCache struct {
	mu          sync.RWMutex
	loaded      bool
	descriptors map[string]ai.ModelDescriptor
	ordered     []ai.ModelDescriptor
}

// Replace atomically replaces the cache with descriptors, including an empty
// successful snapshot.
func (c *ModelCatalogCache) Replace(descriptors []ai.ModelDescriptor) {
	snapshot := make(map[string]ai.ModelDescriptor, len(descriptors))
	ordered := make([]ai.ModelDescriptor, 0, len(descriptors))
	for _, descriptor := range descriptors {
		if descriptor.Model == "" {
			continue
		}
		copy := descriptor.Copy()
		snapshot[copy.Model] = copy
		ordered = append(ordered, copy)
	}
	c.mu.Lock()
	c.descriptors = snapshot
	c.ordered = ordered
	c.loaded = true
	c.mu.Unlock()
}

// Load returns an independent copy of the cached snapshot.
func (c *ModelCatalogCache) Load() ([]ai.ModelDescriptor, bool) {
	c.mu.RLock()
	if !c.loaded {
		c.mu.RUnlock()
		return nil, false
	}
	descriptors := make([]ai.ModelDescriptor, len(c.ordered))
	for i, descriptor := range c.ordered {
		descriptors[i] = descriptor.Copy()
	}
	c.mu.RUnlock()
	return descriptors, true
}

// Lookup returns an independent copy of the cached descriptor for model.
func (c *ModelCatalogCache) Lookup(model string) (ai.ModelDescriptor, bool) {
	c.mu.RLock()
	descriptor, ok := c.descriptors[model]
	c.mu.RUnlock()
	return descriptor.Copy(), ok
}

// IntersectModelDescriptors returns the capabilities supported by both the
// adapter and the provider catalog. Unsupported dominates, Supported requires
// agreement from both inputs, and all other combinations remain Unknown.
func IntersectModelDescriptors(adapter, catalog ai.ModelDescriptor) ai.ModelDescriptor {
	result := ai.ModelDescriptor{
		Provider:         firstDescriptorIdentity(adapter.Provider, catalog.Provider),
		Model:            firstDescriptorIdentity(adapter.Model, catalog.Model),
		NativeMessages:   intersectFeatureSupport(adapter.NativeMessages, catalog.NativeMessages),
		NativeTools:      intersectFeatureSupport(adapter.NativeTools, catalog.NativeTools),
		Usage:            intersectFeatureSupport(adapter.Usage, catalog.Usage),
		FinishReason:     intersectFeatureSupport(adapter.FinishReason, catalog.FinishReason),
		StreamingUsage:   intersectFeatureSupport(adapter.StreamingUsage, catalog.StreamingUsage),
		JSONOutput:       intersectFeatureSupport(adapter.JSONOutput, catalog.JSONOutput),
		JSONSchemaOutput: intersectFeatureSupport(adapter.JSONSchemaOutput, catalog.JSONSchemaOutput),
		Reasoning:        intersectFeatureSupport(adapter.Reasoning, catalog.Reasoning),
		ReasoningEffort:  intersectFeatureSupport(adapter.ReasoningEffort, catalog.ReasoningEffort),
	}
	if result.NativeTools == ai.FeatureSupportSupported {
		result.ToolChoiceModes = intersectKnownValues(adapter.ToolChoiceModes, catalog.ToolChoiceModes)
	}
	if result.ReasoningEffort == ai.FeatureSupportSupported {
		result.ReasoningEfforts = intersectKnownValues(adapter.ReasoningEfforts, catalog.ReasoningEfforts)
	}
	return result
}

// OverrideModelDescriptor replaces facts explicitly supplied by override.
// Unknown values and empty lists leave the corresponding base facts unchanged.
// Intersect the result with an adapter descriptor before enforcing it so an
// override cannot enable behavior the adapter does not implement.
func OverrideModelDescriptor(base, override ai.ModelDescriptor) ai.ModelDescriptor {
	result := base.Copy()
	if override.Provider != "" {
		result.Provider = override.Provider
	}
	if override.Model != "" {
		result.Model = override.Model
	}
	overrideFeatureSupport(&result.NativeMessages, override.NativeMessages)
	overrideFeatureSupport(&result.NativeTools, override.NativeTools)
	overrideFeatureSupport(&result.Usage, override.Usage)
	overrideFeatureSupport(&result.FinishReason, override.FinishReason)
	overrideFeatureSupport(&result.StreamingUsage, override.StreamingUsage)
	overrideFeatureSupport(&result.JSONOutput, override.JSONOutput)
	overrideFeatureSupport(&result.JSONSchemaOutput, override.JSONSchemaOutput)
	overrideFeatureSupport(&result.Reasoning, override.Reasoning)
	overrideFeatureSupport(&result.ReasoningEffort, override.ReasoningEffort)
	if len(override.ToolChoiceModes) > 0 {
		result.ToolChoiceModes = append([]ai.ToolChoiceMode(nil), override.ToolChoiceModes...)
	}
	if len(override.ReasoningEfforts) > 0 {
		result.ReasoningEfforts = append([]ai.ReasoningEffort(nil), override.ReasoningEfforts...)
	}
	return result
}

func intersectFeatureSupport(left, right ai.FeatureSupport) ai.FeatureSupport {
	if left == ai.FeatureSupportUnsupported || right == ai.FeatureSupportUnsupported {
		return ai.FeatureSupportUnsupported
	}
	if left == ai.FeatureSupportSupported && right == ai.FeatureSupportSupported {
		return ai.FeatureSupportSupported
	}
	return ai.FeatureSupportUnknown
}

func intersectKnownValues[T comparable](left, right []T) []T {
	if len(left) == 0 {
		return append([]T(nil), right...)
	}
	if len(right) == 0 {
		return append([]T(nil), left...)
	}
	available := make(map[T]struct{}, len(right))
	for _, value := range right {
		available[value] = struct{}{}
	}
	result := make([]T, 0)
	seen := make(map[T]struct{}, len(left))
	for _, value := range left {
		if _, ok := available[value]; !ok {
			continue
		}
		if _, duplicate := seen[value]; duplicate {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	return result
}

func overrideFeatureSupport(target *ai.FeatureSupport, override ai.FeatureSupport) {
	if override != ai.FeatureSupportUnknown {
		*target = override
	}
}

func firstDescriptorIdentity(primary, secondary string) string {
	if primary != "" {
		return primary
	}
	return secondary
}
