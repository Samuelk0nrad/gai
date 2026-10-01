package modelcatalog

import (
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestIntersectModelDescriptorsUsesStrictTriStateIntersection(t *testing.T) {
	supports := []ai.FeatureSupport{
		ai.FeatureSupportUnknown,
		ai.FeatureSupportSupported,
		ai.FeatureSupportUnsupported,
	}
	for _, adapter := range supports {
		for _, catalog := range supports {
			want := ai.FeatureSupportUnknown
			if adapter == ai.FeatureSupportUnsupported || catalog == ai.FeatureSupportUnsupported {
				want = ai.FeatureSupportUnsupported
			} else if adapter == ai.FeatureSupportSupported && catalog == ai.FeatureSupportSupported {
				want = ai.FeatureSupportSupported
			}
			got := IntersectModelDescriptors(
				ai.ModelDescriptor{NativeMessages: adapter},
				ai.ModelDescriptor{NativeMessages: catalog},
			)
			if got.NativeMessages != want {
				t.Fatalf("%v intersect %v = %v, want %v", adapter, catalog, got.NativeMessages, want)
			}
		}
	}
}

func TestIntersectModelDescriptorsIntersectsKnownValuesWithoutAliasing(t *testing.T) {
	adapter := ai.ModelDescriptor{
		Provider:         "provider",
		Model:            "model",
		NativeTools:      ai.FeatureSupportSupported,
		ToolChoiceModes:  []ai.ToolChoiceMode{ai.ToolChoiceAuto, ai.ToolChoiceRequired},
		ReasoningEffort:  ai.FeatureSupportSupported,
		ReasoningEfforts: []ai.ReasoningEffort{ai.ReasoningEffortLow, ai.ReasoningEffortHigh},
	}
	catalog := ai.ModelDescriptor{
		NativeTools:      ai.FeatureSupportSupported,
		ToolChoiceModes:  []ai.ToolChoiceMode{ai.ToolChoiceRequired, ai.ToolChoiceNone},
		ReasoningEffort:  ai.FeatureSupportSupported,
		ReasoningEfforts: []ai.ReasoningEffort{ai.ReasoningEffortHigh},
	}

	got := IntersectModelDescriptors(adapter, catalog)
	if got.Provider != "provider" || got.Model != "model" {
		t.Fatalf("identity = %q:%q", got.Provider, got.Model)
	}
	if len(got.ToolChoiceModes) != 1 || got.ToolChoiceModes[0] != ai.ToolChoiceRequired {
		t.Fatalf("tool choice modes = %#v", got.ToolChoiceModes)
	}
	if len(got.ReasoningEfforts) != 1 || got.ReasoningEfforts[0] != ai.ReasoningEffortHigh {
		t.Fatalf("reasoning efforts = %#v", got.ReasoningEfforts)
	}
	got.ToolChoiceModes[0] = ai.ToolChoiceNone
	got.ReasoningEfforts[0] = ai.ReasoningEffortLow
	if adapter.ToolChoiceModes[0] != ai.ToolChoiceAuto || catalog.ReasoningEfforts[0] != ai.ReasoningEffortHigh {
		t.Fatal("intersection aliases an input descriptor")
	}
}

func TestOverrideModelDescriptorCannotDefeatAdapterUnsupported(t *testing.T) {
	adapter := ai.ModelDescriptor{Model: "model", Reasoning: ai.FeatureSupportUnsupported}
	facts := ai.ModelDescriptor{}
	override := ai.ModelDescriptor{Reasoning: ai.FeatureSupportSupported}

	got := IntersectModelDescriptors(adapter, OverrideModelDescriptor(facts, override))
	if got.Reasoning != ai.FeatureSupportUnsupported {
		t.Fatalf("reasoning = %v, want unsupported", got.Reasoning)
	}
}

func TestModelCatalogCacheSnapshotsAreImmutable(t *testing.T) {
	var cache ModelCatalogCache
	original := []ai.ModelDescriptor{{
		Model:           "model",
		ToolChoiceModes: []ai.ToolChoiceMode{ai.ToolChoiceAuto},
	}}
	cache.Replace(original)
	original[0].Model = "changed"
	original[0].ToolChoiceModes[0] = ai.ToolChoiceNone

	loaded, ok := cache.Load()
	if !ok || len(loaded) != 1 || loaded[0].Model != "model" || loaded[0].ToolChoiceModes[0] != ai.ToolChoiceAuto {
		t.Fatalf("loaded snapshot = %#v, %v", loaded, ok)
	}
	loaded[0].ToolChoiceModes[0] = ai.ToolChoiceRequired
	lookup, ok := cache.Lookup("model")
	if !ok || lookup.ToolChoiceModes[0] != ai.ToolChoiceAuto {
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
	cache.Replace([]ai.ModelDescriptor{{Model: "z"}, {Model: "a"}, {Model: "m"}})
	for i := 0; i < 10; i++ {
		got, ok := cache.Load()
		if !ok || len(got) != 3 || got[0].Model != "z" || got[1].Model != "a" || got[2].Model != "m" {
			t.Fatalf("load %d order = %#v, %v", i, got, ok)
		}
	}
}
