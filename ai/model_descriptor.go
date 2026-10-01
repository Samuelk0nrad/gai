package ai

import (
	"context"
	"fmt"
)

// FeatureSupport describes whether a model supports a feature. The zero value
// is Unknown, so descriptors can safely omit information they do not know.
type FeatureSupport uint8

const (
	FeatureSupportUnknown FeatureSupport = iota
	FeatureSupportSupported
	FeatureSupportUnsupported
)

// ModelDescriptor describes effective portable adapter capabilities, not all
// native service features. Unknown facts do not reject requests during preflight.
type ModelDescriptor struct {
	// Provider is the provider's stable name when the descriptor is obtained
	// through ModelRepository. Direct model descriptors may leave it empty.
	Provider string
	Model    string
	// NativeMessages reports native role/part transport support. Canonical
	// AIRequest.Messages may also be lowered with the explicit RenderMessages helper.
	NativeMessages FeatureSupport
	// NativeTools reports support for provider-native AIRequest.Tools.
	// Canonical call/result history remains semantic conversation content.
	NativeTools FeatureSupport
	// ToolChoiceModes lists the tool-choice modes supported by the adapter.
	// An empty list means the adapter does not know the supported modes.
	ToolChoiceModes []ToolChoiceMode
	// Usage and FinishReason report whether those AIResponse metadata fields
	// are populated when the provider exposes them.
	Usage        FeatureSupport
	FinishReason FeatureSupport
	// StreamingUsage reports whether usage is emitted through Token.TokenUsage.
	StreamingUsage   FeatureSupport
	JSONOutput       FeatureSupport
	JSONSchemaOutput FeatureSupport
	Reasoning        FeatureSupport
	// ReasoningEfforts enumerates supported values. An empty list means the
	// supported values are not known. ReasoningEffort reports whether effort
	// selection is available.
	ReasoningEfforts []ReasoningEffort
	ReasoningEffort  FeatureSupport
}

// SupportsNativeTools reports whether native tool calling is known to be
// supported. Unknown capability is intentionally treated as unsupported so
// callers use the text tool protocol when native support is unavailable.
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

// ModelDescriber supplies optional model capability information.
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
	// Messages are canonical semantic input, including for text rendering.
	// NativeTools constrains explicit native tool definitions.
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
