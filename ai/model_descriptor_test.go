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
	d := ModelDescriptor{Model: "test", ToolCalling: FeatureSupportSupported, NativeMessages: FeatureSupportSupported, ToolChoiceModes: []ToolChoiceMode{ToolChoiceAuto}, ReasoningEfforts: []ReasoningEffort{ReasoningEffortLow}, Tokenizer: TokenizerDescriptor{Available: FeatureSupportSupported, Fidelity: TokenizerFidelityExact}}
	copy := d.Copy()
	copy.Model = "other"
	copy.Tokenizer.Fidelity = TokenizerFidelityEstimated
	copy.ToolChoiceModes[0] = ToolChoiceNone
	copy.ReasoningEfforts[0] = ReasoningEffortHigh
	if d.Model != "test" || d.Tokenizer.Fidelity != TokenizerFidelityExact || d.ToolChoiceModes[0] != ToolChoiceAuto || d.ReasoningEfforts[0] != ReasoningEffortLow {
		t.Fatalf("Copy mutated original: %#v", d)
	}
}

func TestIntersectModelDescriptorsUsesStrictTriStateIntersection(t *testing.T) {
	supports := []FeatureSupport{
		FeatureSupportUnknown,
		FeatureSupportSupported,
		FeatureSupportUnsupported,
	}
	for _, adapter := range supports {
		for _, catalog := range supports {
			want := FeatureSupportUnknown
			if adapter == FeatureSupportUnsupported || catalog == FeatureSupportUnsupported {
				want = FeatureSupportUnsupported
			} else if adapter == FeatureSupportSupported && catalog == FeatureSupportSupported {
				want = FeatureSupportSupported
			}
			got := IntersectModelDescriptors(
				ModelDescriptor{NativeMessages: adapter},
				ModelDescriptor{NativeMessages: catalog},
			)
			if got.NativeMessages != want {
				t.Fatalf("%v intersect %v = %v, want %v", adapter, catalog, got.NativeMessages, want)
			}
		}
	}
}

func TestIntersectModelDescriptorsIntersectsKnownValuesWithoutAliasing(t *testing.T) {
	adapter := ModelDescriptor{
		Provider:         "provider",
		Model:            "model",
		NativeTools:      FeatureSupportSupported,
		ToolCalling:      FeatureSupportSupported,
		ToolChoiceModes:  []ToolChoiceMode{ToolChoiceAuto, ToolChoiceRequired},
		ReasoningEffort:  FeatureSupportSupported,
		ReasoningEfforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortHigh},
		Tokenizer:        TokenizerDescriptor{Available: FeatureSupportSupported, Fidelity: TokenizerFidelityExact},
	}
	catalog := ModelDescriptor{
		NativeTools:      FeatureSupportSupported,
		ToolCalling:      FeatureSupportSupported,
		ToolChoiceModes:  []ToolChoiceMode{ToolChoiceRequired, ToolChoiceNone},
		ReasoningEffort:  FeatureSupportSupported,
		ReasoningEfforts: []ReasoningEffort{ReasoningEffortHigh},
		Tokenizer:        TokenizerDescriptor{Available: FeatureSupportSupported, Fidelity: TokenizerFidelityEstimated},
	}

	got := IntersectModelDescriptors(adapter, catalog)
	if got.Provider != "provider" || got.Model != "model" {
		t.Fatalf("identity = %q:%q", got.Provider, got.Model)
	}
	if len(got.ToolChoiceModes) != 1 || got.ToolChoiceModes[0] != ToolChoiceRequired {
		t.Fatalf("tool choice modes = %#v", got.ToolChoiceModes)
	}
	if len(got.ReasoningEfforts) != 1 || got.ReasoningEfforts[0] != ReasoningEffortHigh {
		t.Fatalf("reasoning efforts = %#v", got.ReasoningEfforts)
	}
	if got.Tokenizer.Fidelity != TokenizerFidelityUnknown {
		t.Fatalf("tokenizer fidelity = %v, want unknown conflict", got.Tokenizer.Fidelity)
	}
	got.ToolChoiceModes[0] = ToolChoiceNone
	got.ReasoningEfforts[0] = ReasoningEffortLow
	if adapter.ToolChoiceModes[0] != ToolChoiceAuto || catalog.ReasoningEfforts[0] != ReasoningEffortHigh {
		t.Fatal("intersection aliases an input descriptor")
	}
}

func TestOverrideModelDescriptorCannotDefeatAdapterUnsupported(t *testing.T) {
	adapter := ModelDescriptor{Model: "model", Reasoning: FeatureSupportUnsupported}
	facts := ModelDescriptor{}
	override := ModelDescriptor{Reasoning: FeatureSupportSupported}

	got := IntersectModelDescriptors(adapter, OverrideModelDescriptor(facts, override))
	if got.Reasoning != FeatureSupportUnsupported {
		t.Fatalf("reasoning = %v, want unsupported", got.Reasoning)
	}
}

func TestModelCatalogCacheSnapshotsAreImmutable(t *testing.T) {
	var cache ModelCatalogCache
	original := []ModelDescriptor{{
		Model:           "model",
		ToolChoiceModes: []ToolChoiceMode{ToolChoiceAuto},
	}}
	cache.Replace(original)
	original[0].Model = "changed"
	original[0].ToolChoiceModes[0] = ToolChoiceNone

	loaded, ok := cache.Load()
	if !ok || len(loaded) != 1 || loaded[0].Model != "model" || loaded[0].ToolChoiceModes[0] != ToolChoiceAuto {
		t.Fatalf("loaded snapshot = %#v, %v", loaded, ok)
	}
	loaded[0].ToolChoiceModes[0] = ToolChoiceRequired
	lookup, ok := cache.Lookup("model")
	if !ok || lookup.ToolChoiceModes[0] != ToolChoiceAuto {
		t.Fatalf("lookup = %#v, %v", lookup, ok)
	}

	cache.Replace(nil)
	empty, ok := cache.Load()
	if !ok || len(empty) != 0 {
		t.Fatalf("empty successful snapshot = %#v, %v", empty, ok)
	}
}

func TestModelCatalogCachePreservesSnapshotOrder(t *testing.T) {
	var cache ModelCatalogCache
	cache.Replace([]ModelDescriptor{{Model: "z"}, {Model: "a"}, {Model: "m"}})
	for i := 0; i < 10; i++ {
		got, ok := cache.Load()
		if !ok || len(got) != 3 || got[0].Model != "z" || got[1].Model != "a" || got[2].Model != "m" {
			t.Fatalf("load %d order = %#v, %v", i, got, ok)
		}
	}
}

func TestContextMutexHonorsCanceledContext(t *testing.T) {
	var mu ContextMutex
	if err := mu.Lock(context.Background()); err != nil {
		t.Fatalf("first lock: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := mu.Lock(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled lock error = %v, want context.Canceled", err)
	}
	mu.Unlock()
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
