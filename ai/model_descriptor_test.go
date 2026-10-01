package ai

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

func TestModelDescriptorZeroValueIsUnknownAndPermitsRequests(t *testing.T) {
	if FeatureSupportUnknown != 0 {
		t.Fatalf("FeatureSupportUnknown = %d, want zero", FeatureSupportUnknown)
	}
	if err := (ModelDescriptor{}).ValidateRequest(AIRequest{Messages: []Message{TextMessage(RoleUser, "question")}, Reasoning: ReasoningConfig{Effort: ReasoningEffortHigh}}); err != nil {
		t.Fatalf("zero descriptor rejected unknown capability: %v", err)
	}
}

func TestModelDescriptorSupportsNativeTools(t *testing.T) {
	tests := []struct {
		name    string
		support FeatureSupport
		want    bool
	}{
		{name: "supported", support: FeatureSupportSupported, want: true},
		{name: "unknown", support: FeatureSupportUnknown},
		{name: "unsupported", support: FeatureSupportUnsupported},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := (ModelDescriptor{NativeTools: tt.support}).SupportsNativeTools(); got != tt.want {
				t.Fatalf("SupportsNativeTools() = %t, want %t", got, tt.want)
			}
		})
	}
}

func TestModelDescriptorCopy(t *testing.T) {
	d := ModelDescriptor{Model: "test", NativeMessages: FeatureSupportSupported, ToolChoiceModes: []ToolChoiceMode{ToolChoiceAuto}, ReasoningEfforts: []ReasoningEffort{ReasoningEffortLow}}
	copy := d.Copy()
	copy.Model = "other"

	copy.ToolChoiceModes[0] = ToolChoiceNone
	copy.ReasoningEfforts[0] = ReasoningEffortHigh
	if d.Model != "test" || d.ToolChoiceModes[0] != ToolChoiceAuto || d.ReasoningEfforts[0] != ReasoningEffortLow {
		t.Fatalf("Copy mutated original: %#v", d)
	}
}

func TestModelDescriptorRejectsUnsupportedWithTypedError(t *testing.T) {
	err := (ModelDescriptor{Model: "test", ReasoningEffort: FeatureSupportUnsupported}).ValidateRequest(AIRequest{Messages: []Message{TextMessage(RoleUser, "question")}, Reasoning: ReasoningConfig{Effort: ReasoningEffortHigh}})
	if !errors.Is(err, ErrUnsupportedCapability) {
		t.Fatalf("error = %v, want ErrUnsupportedCapability", err)
	}
	var unsupported *UnsupportedCapabilityError
	if !errors.As(err, &unsupported) || unsupported.Capability != "reasoning effort" || unsupported.Model != "test" {
		t.Fatalf("error = %#v, want typed reasoning effort error", err)
	}
}

func TestModelDescriptorRejectsUnadvertisedReasoningEffort(t *testing.T) {
	d := ModelDescriptor{Model: "test", ReasoningEffort: FeatureSupportSupported, ReasoningEfforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortHigh}}
	err := d.ValidateRequest(AIRequest{Messages: []Message{TextMessage(RoleUser, "question")}, Reasoning: ReasoningConfig{Effort: ReasoningEffortMedium}})
	if !errors.Is(err, ErrUnsupportedCapability) {
		t.Fatalf("error = %v, want unsupported capability", err)
	}
}

func TestModelDescriptorRejectsUnsupportedNativeCapabilities(t *testing.T) {
	messageErr := (ModelDescriptor{Model: "test", NativeMessages: FeatureSupportUnsupported}).ValidateRequest(AIRequest{Messages: []Message{TextMessage(RoleUser, "hello")}})
	if messageErr != nil {
		t.Fatalf("canonical text must not require native messages: %v", messageErr)
	}
	toolErr := (ModelDescriptor{Model: "test", NativeTools: FeatureSupportUnsupported}).ValidateRequest(AIRequest{Messages: []Message{TextMessage(RoleUser, "question")}, Tools: []ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: []byte(`{"type":"object"}`)}}})
	if !errors.Is(toolErr, ErrUnsupportedCapability) {
		t.Fatalf("native tool error = %v, want unsupported capability", toolErr)
	}
}

func TestModelDescriptorPermitsSemanticToolHistoryWithoutNativeTools(t *testing.T) {
	d := ModelDescriptor{
		Model:          "test",
		NativeMessages: FeatureSupportSupported,
		NativeTools:    FeatureSupportUnsupported,
	}
	req := AIRequest{Messages: []Message{
		{Role: RoleAssistant, Parts: []ContentPart{{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "call_1", Type: "function", Name: "search", Args: json.RawMessage(`{"q":"x"}`)}}}},
		{Role: RoleTool, Parts: []ContentPart{{Kind: ContentToolResult, ToolResult: &ToolResult{ToolCallID: "call_1", Name: "search", Parts: TextParts("ok")}}}},
	}}

	if err := d.ValidateRequest(req); err != nil {
		t.Fatalf("semantic tool history must permit text fallback: %v", err)
	}
	if _, err := RenderMessages(t.Context(), req.Messages); err != nil {
		t.Fatalf("fallback: %v", err)
	}

}

func TestValidateModelRequestSupportsMinimalModel(t *testing.T) {
	if err := ValidateModelRequest(minimalModel{}, AIRequest{Messages: []Message{TextMessage(RoleUser, "question")}, Reasoning: ReasoningConfig{Effort: ReasoningEffortHigh}}); err != nil {
		t.Fatalf("minimal model request rejected: %v", err)
	}
}

type minimalModel struct{}

func (minimalModel) GenerateStream(context.Context, AIRequest) <-chan Token { return nil }
