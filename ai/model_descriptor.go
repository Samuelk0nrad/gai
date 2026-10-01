package ai

import (
	"context"
	"fmt"
	"sync"
)

// FeatureSupport describes whether a model supports a feature. The zero value
// is Unknown, so descriptors can safely omit information they do not know.
type FeatureSupport uint8

const (
	FeatureSupportUnknown FeatureSupport = iota
	FeatureSupportSupported
	FeatureSupportUnsupported
)

// TokenizerFidelity describes how closely tokenizer results match provider
// billing and generation tokenization.
type TokenizerFidelity uint8

const (
	TokenizerFidelityUnknown TokenizerFidelity = iota
	TokenizerFidelityExact
	TokenizerFidelityEstimated
)

// TokenizerDescriptor describes the legacy concrete tokenizer without
// constructing it. It does not describe the automatic local TokenCounter; use
// TokenCounter.Fidelity for that algorithm.
type TokenizerDescriptor struct {
	Available FeatureSupport
	Fidelity  TokenizerFidelity
}

// ModelDescriptor describes capabilities known for a model. Unknown features
// are deliberately not rejected during local preflight.
type ModelDescriptor struct {
	// Provider is the provider's stable name when the descriptor is obtained
	// through ModelRepository. Direct model descriptors may leave it empty.
	Provider string
	Model    string
	// NativeMessages reports native role/part transport support. Canonical
	// AIRequest.Messages may also be lowered to a rendered compatibility prompt.
	NativeMessages FeatureSupport
	// NativeTools reports support for AIRequest.Tools and tool-call responses.
	NativeTools FeatureSupport
	// ToolChoiceModes lists the tool-choice modes supported by the adapter.
	// An empty list means the adapter does not know the supported modes.
	ToolChoiceModes []ToolChoiceMode
	// Multimodal reports support for non-text request or response content.
	Multimodal FeatureSupport
	// Usage and FinishReason report whether those AIResponse metadata fields
	// are populated when the provider exposes them.
	Usage        FeatureSupport
	FinishReason FeatureSupport
	// StreamingUsage reports whether usage is emitted through Token.TokenUsage.
	StreamingUsage   FeatureSupport
	ToolCalling      FeatureSupport
	JSONOutput       FeatureSupport
	JSONSchemaOutput FeatureSupport
	Reasoning        FeatureSupport
	// ReasoningEfforts enumerates supported values. An empty list means the
	// supported values are not known. ReasoningEffort remains the compatibility
	// summary for callers that only need a yes/no/unknown answer.
	ReasoningEfforts []ReasoningEffort
	ReasoningEffort  FeatureSupport
	Tokenizer        TokenizerDescriptor
}

// SupportsNativeTools reports whether native tool calling is known to be
// supported. Unknown capability is intentionally treated as unsupported so
// callers can retain the text-based tool protocol as a compatibility fallback.
func (d ModelDescriptor) SupportsNativeTools() bool {
	return d.NativeTools == FeatureSupportSupported
}

// Copy returns an independent copy of d. It is provided so callers need not
// rely on the descriptor's current value-only representation.
func (d ModelDescriptor) Copy() ModelDescriptor {
	d.ToolChoiceModes = append([]ToolChoiceMode(nil), d.ToolChoiceModes...)
	d.ReasoningEfforts = append([]ReasoningEffort(nil), d.ReasoningEfforts...)
	return d
}

// ModelDescriber is an optional extension to Model. Model intentionally does
// not include Descriptor so existing custom Model implementations keep source
// compatibility.
type ModelDescriber interface {
	Descriptor() ModelDescriptor
}

// ModelCatalogProvider is an optional context-aware extension to Provider.
// Implementations may perform discovery while building the snapshot. Returned
// descriptors must be independent copies, and model request paths must not call
// this method.
type ModelCatalogProvider interface {
	ListModelDescriptors(context.Context) ([]ModelDescriptor, error)
}

// ContextMutex is a zero-value-ready mutex that honors context cancellation
// while waiting for ownership. It is useful for serialized, optional discovery
// paths where a canceled caller must not wait behind network I/O.
type ContextMutex struct {
	once sync.Once
	ch   chan struct{}
}

func (m *ContextMutex) Lock(ctx context.Context) error {
	m.once.Do(func() {
		m.ch = make(chan struct{}, 1)
		m.ch <- struct{}{}
	})
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-m.ch:
		return nil
	}
}

func (m *ContextMutex) Unlock() {
	m.ch <- struct{}{}
}

// ModelCatalogCache stores the last successful provider catalog snapshot.
// Replacement and reads deep-copy descriptors so callers cannot mutate the
// cached snapshot. Its zero value is ready for use.
type ModelCatalogCache struct {
	mu          sync.RWMutex
	loaded      bool
	descriptors map[string]ModelDescriptor
	ordered     []ModelDescriptor
}

// Replace atomically replaces the cache with descriptors, including an empty
// successful snapshot.
func (c *ModelCatalogCache) Replace(descriptors []ModelDescriptor) {
	snapshot := make(map[string]ModelDescriptor, len(descriptors))
	ordered := make([]ModelDescriptor, 0, len(descriptors))
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
func (c *ModelCatalogCache) Load() ([]ModelDescriptor, bool) {
	c.mu.RLock()
	if !c.loaded {
		c.mu.RUnlock()
		return nil, false
	}
	descriptors := make([]ModelDescriptor, len(c.ordered))
	for i, descriptor := range c.ordered {
		descriptors[i] = descriptor.Copy()
	}
	c.mu.RUnlock()
	return descriptors, true
}

// Lookup returns an independent copy of the cached descriptor for model.
func (c *ModelCatalogCache) Lookup(model string) (ModelDescriptor, bool) {
	c.mu.RLock()
	descriptor, ok := c.descriptors[model]
	c.mu.RUnlock()
	return descriptor.Copy(), ok
}

// IntersectModelDescriptors returns the capabilities supported by both the
// adapter and the provider catalog. Unsupported dominates, Supported requires
// agreement from both inputs, and all other combinations remain Unknown.
func IntersectModelDescriptors(adapter, catalog ModelDescriptor) ModelDescriptor {
	result := ModelDescriptor{
		Provider:         firstDescriptorIdentity(adapter.Provider, catalog.Provider),
		Model:            firstDescriptorIdentity(adapter.Model, catalog.Model),
		NativeMessages:   intersectFeatureSupport(adapter.NativeMessages, catalog.NativeMessages),
		NativeTools:      intersectFeatureSupport(adapter.NativeTools, catalog.NativeTools),
		Multimodal:       intersectFeatureSupport(adapter.Multimodal, catalog.Multimodal),
		Usage:            intersectFeatureSupport(adapter.Usage, catalog.Usage),
		FinishReason:     intersectFeatureSupport(adapter.FinishReason, catalog.FinishReason),
		StreamingUsage:   intersectFeatureSupport(adapter.StreamingUsage, catalog.StreamingUsage),
		ToolCalling:      intersectFeatureSupport(adapter.ToolCalling, catalog.ToolCalling),
		JSONOutput:       intersectFeatureSupport(adapter.JSONOutput, catalog.JSONOutput),
		JSONSchemaOutput: intersectFeatureSupport(adapter.JSONSchemaOutput, catalog.JSONSchemaOutput),
		Reasoning:        intersectFeatureSupport(adapter.Reasoning, catalog.Reasoning),
		ReasoningEffort:  intersectFeatureSupport(adapter.ReasoningEffort, catalog.ReasoningEffort),
	}
	result.Tokenizer.Available = intersectFeatureSupport(adapter.Tokenizer.Available, catalog.Tokenizer.Available)
	if result.Tokenizer.Available == FeatureSupportSupported {
		result.Tokenizer.Fidelity = intersectTokenizerFidelity(adapter.Tokenizer.Fidelity, catalog.Tokenizer.Fidelity)
	}
	if result.ToolCalling == FeatureSupportSupported && result.NativeTools == FeatureSupportSupported {
		result.ToolChoiceModes = intersectKnownValues(adapter.ToolChoiceModes, catalog.ToolChoiceModes)
	}
	if result.ReasoningEffort == FeatureSupportSupported {
		result.ReasoningEfforts = intersectKnownValues(adapter.ReasoningEfforts, catalog.ReasoningEfforts)
	}
	return result
}

// OverrideModelDescriptor replaces facts explicitly supplied by override.
// Unknown values and empty lists leave the corresponding base facts unchanged.
// Intersect the result with an adapter descriptor before enforcing it so an
// override cannot enable behavior the adapter does not implement.
func OverrideModelDescriptor(base, override ModelDescriptor) ModelDescriptor {
	result := base.Copy()
	if override.Provider != "" {
		result.Provider = override.Provider
	}
	if override.Model != "" {
		result.Model = override.Model
	}
	overrideFeatureSupport(&result.NativeMessages, override.NativeMessages)
	overrideFeatureSupport(&result.NativeTools, override.NativeTools)
	overrideFeatureSupport(&result.Multimodal, override.Multimodal)
	overrideFeatureSupport(&result.Usage, override.Usage)
	overrideFeatureSupport(&result.FinishReason, override.FinishReason)
	overrideFeatureSupport(&result.StreamingUsage, override.StreamingUsage)
	overrideFeatureSupport(&result.ToolCalling, override.ToolCalling)
	overrideFeatureSupport(&result.JSONOutput, override.JSONOutput)
	overrideFeatureSupport(&result.JSONSchemaOutput, override.JSONSchemaOutput)
	overrideFeatureSupport(&result.Reasoning, override.Reasoning)
	overrideFeatureSupport(&result.ReasoningEffort, override.ReasoningEffort)
	overrideFeatureSupport(&result.Tokenizer.Available, override.Tokenizer.Available)
	if override.Tokenizer.Fidelity != TokenizerFidelityUnknown {
		result.Tokenizer.Fidelity = override.Tokenizer.Fidelity
	}
	if len(override.ToolChoiceModes) > 0 {
		result.ToolChoiceModes = append([]ToolChoiceMode(nil), override.ToolChoiceModes...)
	}
	if len(override.ReasoningEfforts) > 0 {
		result.ReasoningEfforts = append([]ReasoningEffort(nil), override.ReasoningEfforts...)
	}
	return result
}

func intersectFeatureSupport(left, right FeatureSupport) FeatureSupport {
	if left == FeatureSupportUnsupported || right == FeatureSupportUnsupported {
		return FeatureSupportUnsupported
	}
	if left == FeatureSupportSupported && right == FeatureSupportSupported {
		return FeatureSupportSupported
	}
	return FeatureSupportUnknown
}

func intersectTokenizerFidelity(left, right TokenizerFidelity) TokenizerFidelity {
	if left == TokenizerFidelityUnknown {
		return right
	}
	if right == TokenizerFidelityUnknown {
		return left
	}
	if left == right {
		return left
	}
	return TokenizerFidelityUnknown
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

func overrideFeatureSupport(target *FeatureSupport, override FeatureSupport) {
	if override != FeatureSupportUnknown {
		*target = override
	}
}

func firstDescriptorIdentity(primary, secondary string) string {
	if primary != "" {
		return primary
	}
	return secondary
}

// UnsupportedCapabilityError reports a request feature known to be unsupported
// by a model before a provider request is made.
type UnsupportedCapabilityError struct {
	Model      string
	Capability string
}

func (e *UnsupportedCapabilityError) Error() string {
	if e.Model == "" {
		return fmt.Sprintf("%v: %s", ErrUnsupportedCapability, e.Capability)
	}
	return fmt.Sprintf("%v: model %q does not support %s", ErrUnsupportedCapability, e.Model, e.Capability)
}

func (e *UnsupportedCapabilityError) Unwrap() error { return ErrUnsupportedCapability }

// ValidateRequest validates req and rejects only features explicitly marked
// unsupported by d.
func (d ModelDescriptor) ValidateRequest(req AIRequest) error {
	if err := req.Validate(); err != nil {
		return err
	}
	// Messages describes semantic input even for rendered transports. Native
	// capability flags constrain native tool definitions, not conversation data.
	if len(req.Tools) > 0 && d.ToolCalling == FeatureSupportUnsupported {
		return d.unsupported("tool calling")
	}
	if len(req.Tools) > 0 && d.NativeTools == FeatureSupportUnsupported {
		return d.unsupported("native tools")
	}
	if len(req.Tools) > 0 && req.ToolChoice.Mode != "" && len(d.ToolChoiceModes) > 0 && !containsToolChoiceMode(d.ToolChoiceModes, req.ToolChoice.Mode) {
		return d.unsupported("tool choice mode " + string(req.ToolChoice.Mode))
	}
	switch req.ResponseFormat.Type {
	case ResponseFormatJSONObject:
		if d.JSONOutput == FeatureSupportUnsupported {
			return d.unsupported("JSON output")
		}
	case ResponseFormatJSONSchema:
		if d.JSONSchemaOutput == FeatureSupportUnsupported {
			return d.unsupported("JSON schema output")
		}
	}
	if req.Reasoning.Enabled || req.Reasoning.IncludeThoughts || req.Reasoning.BudgetTokens > 0 {
		if d.Reasoning == FeatureSupportUnsupported {
			return d.unsupported("reasoning")
		}
	}
	if req.Reasoning.Effort != "" && d.ReasoningEffort == FeatureSupportUnsupported {
		return d.unsupported("reasoning effort")
	}
	if req.Reasoning.Effort != "" && len(d.ReasoningEfforts) > 0 && !containsReasoningEffort(d.ReasoningEfforts, req.Reasoning.Effort) {
		return d.unsupported("reasoning effort " + string(req.Reasoning.Effort))
	}
	return nil
}

func containsToolChoiceMode(modes []ToolChoiceMode, want ToolChoiceMode) bool {
	for _, mode := range modes {
		if mode == want {
			return true
		}
	}
	return false
}

func containsReasoningEffort(efforts []ReasoningEffort, want ReasoningEffort) bool {
	for _, effort := range efforts {
		if effort == want {
			return true
		}
	}
	return false
}

func (d ModelDescriptor) unsupported(capability string) error {
	return &UnsupportedCapabilityError{Model: d.Model, Capability: capability}
}

// ValidateModelRequest applies common request validation and, when available,
// the optional model descriptor's local preflight checks.
func ValidateModelRequest(model Model, req AIRequest) error {
	if describer, ok := model.(ModelDescriber); ok {
		return describer.Descriptor().ValidateRequest(req)
	}
	return req.Validate()
}
