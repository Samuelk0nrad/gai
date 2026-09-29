package openai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
)

type errorRoundTripper struct{}

func (errorRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("model discovery unavailable")
}

type handlerRoundTripper func(*http.Request) (*http.Response, error)

func (h handlerRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return h(r) }

func response(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Status: http.StatusText(status), Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
}

func TestProviderDynamicallyListsModelsAndAcceptsThem(t *testing.T) {
	var requests []*http.Request
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(r *http.Request) (*http.Response, error) {
		requests = append(requests, r.Clone(r.Context()))
		return response(http.StatusOK, `{"data":[{"id":"gpt-dynamic"}]}`), nil
	})}
	p.baseURL = "https://models.test/v1"

	models, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if len(models) != 1 || models[0] != "gpt-dynamic" {
		t.Fatalf("unexpected models: %#v", models)
	}

	model, err := p.Model("gpt-dynamic")
	if err != nil {
		t.Fatalf("Model returned error: %v", err)
	}
	if ai.ModelName(model) != "gpt-dynamic" {
		t.Fatalf("unexpected model name: %q", ai.ModelName(model))
	}
	if len(requests) != 1 {
		t.Fatalf("expected one discovery request, got %d", len(requests))
	}
	request := requests[0]
	if request.Method != http.MethodGet || request.URL.Path != "/v1/models" {
		t.Fatalf("unexpected request: %s %s", request.Method, request.URL.String())
	}
	if request.Header.Get("Authorization") != "Bearer test-key" {
		t.Fatalf("unexpected authorization: %q", request.Header.Get("Authorization"))
	}
}

func TestProviderAggregatesDynamicReasoningDescriptors(t *testing.T) {
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(*http.Request) (*http.Response, error) {
		return response(http.StatusOK, `{"data":[{"id":"gpt-5.6-terra"},{"id":"future-chat-model"}]}`), nil
	})}
	repo := ai.NewModelRepository(nil)
	if err := repo.RegisterProvider(context.Background(), p); err != nil {
		t.Fatal(err)
	}
	descriptors, err := repo.ListModelDescriptors(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(descriptors) != 2 || descriptors[0].Provider != "openai" || descriptors[0].Model != "future-chat-model" {
		t.Fatalf("descriptors = %#v", descriptors)
	}
	if descriptors[1].Model != "gpt-5.6-terra" || descriptors[1].ReasoningEffort != ai.FeatureSupportSupported || len(descriptors[1].ReasoningEfforts) != 6 || descriptors[1].ReasoningEfforts[0] != ai.ReasoningEffortNone || descriptors[1].ReasoningEfforts[5] != ai.ReasoningEffortMax {
		t.Fatalf("dynamic reasoning descriptor = %#v", descriptors[1])
	}
	if descriptors[0].ReasoningEffort != ai.FeatureSupportUnknown {
		t.Fatalf("unrelated dynamic descriptor = %#v, want unknown reasoning effort", descriptors[0])
	}
}

func TestModelDescriptorUsesCatalogSnapshotWithoutNetwork(t *testing.T) {
	hits := 0
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(*http.Request) (*http.Response, error) {
		hits++
		return response(http.StatusOK, `{"data":[{"id":"gpt-5.6-terra"}]}`), nil
	})}
	if _, err := p.ListModelDescriptors(context.Background()); err != nil {
		t.Fatal(err)
	}
	model, err := p.Model("gpt-5.6-terra")
	if err != nil {
		t.Fatal(err)
	}
	if got := model.(ai.ModelDescriber).Descriptor(); got.ReasoningEffort != ai.FeatureSupportSupported {
		t.Fatalf("descriptor = %#v", got)
	}
	if hits != 1 {
		t.Fatalf("descriptor caused catalog I/O: got %d requests, want 1", hits)
	}
}

func TestProviderExcludesNonChatModelsFromDiscovery(t *testing.T) {
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(*http.Request) (*http.Response, error) {
		return response(http.StatusOK, `{"data":[{"id":"gpt-dynamic"},{"id":"gpt-image-1"},{"id":"text-embedding-3-large"},{"id":"omni-moderation-latest"}]}`), nil
	})}

	models, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if len(models) != 1 || models[0] != "gpt-dynamic" {
		t.Fatalf("expected only chat-capable models, got %#v", models)
	}
}

func TestProviderFallsBackWhenModelDiscoveryFails(t *testing.T) {
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: errorRoundTripper{}}

	models, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	found := false
	for _, model := range models {
		if model == GPT56Sol {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected fallback models to include %q, got %#v", GPT56Sol, models)
	}
	if _, err := p.Model(GPT56Sol); err != nil {
		t.Fatalf("Model did not accept fallback model: %v", err)
	}
}

func TestProviderFallsBackWhenModelDiscoveryTimesOut(t *testing.T) {
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(r *http.Request) (*http.Response, error) {
		<-r.Context().Done()
		return nil, r.Context().Err()
	})}

	descriptors, err := p.ListModelDescriptors(context.Background())
	if err != nil {
		t.Fatalf("ListModelDescriptors returned error: %v", err)
	}
	found := false
	for _, descriptor := range descriptors {
		if descriptor.Model == GPT56Sol {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected hard-coded fallback descriptors, got %#v", descriptors)
	}
}

func TestProviderBoundsModelDiscovery(t *testing.T) {
	p := New("test-key", nil)
	var hasDeadline bool
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(r *http.Request) (*http.Response, error) {
		_, hasDeadline = r.Context().Deadline()
		return nil, errors.New("model discovery unavailable")
	})}

	models, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if !hasDeadline {
		t.Fatal("expected model discovery request to have a deadline")
	}
	if len(models) == 0 {
		t.Fatal("expected fallback models after discovery failure")
	}
}

func TestProviderModelValidation(t *testing.T) {
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: errorRoundTripper{}}

	model, err := p.Model("   ")
	if !errors.Is(err, ai.ErrModelNotFound) {
		t.Fatalf("expected ErrModelNotFound, got %v", err)
	}
	if model != nil {
		t.Fatalf("expected nil model on empty name")
	}

	model, err = p.Model("unknown-model")
	if err != nil {
		t.Fatalf("expected dynamically discoverable model name to be accepted, got %v", err)
	}
	if model == nil || ai.ModelName(model) != "unknown-model" {
		t.Fatalf("unexpected model for dynamic name: %#v", model)
	}
}

func TestProviderModelAndFallbackList(t *testing.T) {
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: errorRoundTripper{}}

	model, err := p.Model(GPT41Mini)
	if err != nil {
		t.Fatalf("Model returned error: %v", err)
	}
	if model == nil || ai.ModelName(model) != GPT41Mini {
		t.Fatalf("unexpected model: %#v", model)
	}

	models, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("expected static models")
	}
	models[0] = "mutated"
	again, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if again[0] == "mutated" {
		t.Fatal("ListModels must return a copy")
	}
}

func TestProviderExposesTextModelsOnly(t *testing.T) {
	p := New("test-key", nil)
	models, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}

	for _, want := range []string{GPT56Terra, GPT56Sol, GPT56Luna} {
		if !isKnownModel(want) {
			t.Errorf("expected text model %q to be supported", want)
		}
	}
	for _, excluded := range []string{"gpt-image-1", "gpt-5.6-auto"} {
		if isKnownModel(excluded) {
			t.Errorf("did not expect non-text model %q to be supported", excluded)
		}
	}
	if len(models) == 0 {
		t.Fatal("expected text models")
	}
}

func TestProviderValidation(t *testing.T) {
	if err := (*Provider)(nil).Validate(); !errors.Is(err, ai.ErrNilProvider) {
		t.Fatalf("expected ErrNilProvider, got %v", err)
	}
	if err := New("   ", nil).Validate(); !errors.Is(err, ErrInvalidAPIKey) {
		t.Fatalf("expected ErrInvalidAPIKey, got %v", err)
	}
	if model, err := New("test-key", nil).Model("unknown"); err != nil || model == nil || ai.ModelName(model) != "unknown" {
		t.Fatalf("expected dynamic model acceptance, got model=%#v err=%v", model, err)
	}
}

func TestStreamingHTTPClientDoesNotApplyProviderTimeout(t *testing.T) {
	p := New("test-key", nil)
	if p.httpClient.Timeout != 0 {
		t.Fatalf("expected no provider client timeout, got %s", p.httpClient.Timeout)
	}
	p.httpClient.Timeout = time.Millisecond

	streaming := p.streamingHTTPClient()
	if streaming == p.httpClient {
		t.Fatal("streamingHTTPClient must return a copy")
	}
	if streaming.Timeout != 0 {
		t.Fatalf("expected no streaming client timeout, got %s", streaming.Timeout)
	}
	if p.httpClient.Timeout != time.Millisecond {
		t.Fatalf("streamingHTTPClient mutated provider timeout: %s", p.httpClient.Timeout)
	}
}
