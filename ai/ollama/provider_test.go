package ollama

import (
	"errors"
	"net/http"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestProviderResolvesArbitraryModelsAndSnapshotsConfiguration(t *testing.T) {
	client := &http.Client{}
	provider := New(nil, WithBaseURL("http://ollama.test/root/"), WithHTTPClient(client), WithBearerToken("secret"))

	if got := provider.Name(); got != "ollama" {
		t.Fatalf("Name() = %q, want ollama", got)
	}
	model, err := provider.TypedModel("  custom/model:latest  ", WithOptions(Options{NumCtx: intPtr(8192)}))
	if err != nil {
		t.Fatal(err)
	}
	if got := model.Name(); got != "custom/model:latest" {
		t.Fatalf("Name() = %q", got)
	}
	if _, err := provider.Model(""); !errors.Is(err, ai.ErrModelNotFound) {
		t.Fatalf("empty model error = %v", err)
	}
	if provider.baseURL != "http://ollama.test/root" {
		t.Fatalf("baseURL = %q", provider.baseURL)
	}
	if provider.httpClient == client {
		t.Fatal("caller HTTP client was not shallow-copied")
	}
}

func TestProviderDefaultsAndDescriptor(t *testing.T) {
	provider := New(nil)
	if provider.baseURL != "http://localhost:11434" {
		t.Fatalf("baseURL = %q", provider.baseURL)
	}
	model, err := provider.TypedModel("qwen3:8b")
	if err != nil {
		t.Fatal(err)
	}
	d := model.Descriptor()
	if d.Model != "qwen3:8b" || d.NativeMessages != ai.FeatureSupportSupported || d.NativeTools != ai.FeatureSupportUnknown {
		t.Fatalf("descriptor = %#v", d)
	}
	if d.JSONOutput != ai.FeatureSupportUnsupported || d.JSONSchemaOutput != ai.FeatureSupportUnsupported || d.Reasoning != ai.FeatureSupportUnsupported {
		t.Fatalf("unsupported capabilities not explicit: %#v", d)
	}
}

func intPtr(v int) *int { return &v }
