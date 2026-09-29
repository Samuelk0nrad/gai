package mistral

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

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
		return response(http.StatusOK, `{"data":[
			{"id":"mistral-chat","capabilities":{"completion_chat":true}},
			{"id":"mistral-fim","capabilities":{"completion_chat":false}},
			{"id":"mistral-without-capability"}
		]}`), nil
	})}
	p.baseURL = "https://models.test"

	models, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if len(models) != 2 || models[0] != "mistral-chat" || models[1] != "mistral-without-capability" {
		t.Fatalf("unexpected models: %#v", models)
	}

	model, err := p.Model("mistral-chat")
	if err != nil {
		t.Fatalf("Model returned error: %v", err)
	}
	if ai.ModelName(model) != "mistral-chat" {
		t.Fatalf("unexpected model name: %q", ai.ModelName(model))
	}
	if len(requests) != 1 {
		t.Fatalf("expected one discovery request, got %d", len(requests))
	}
	for _, request := range requests {
		if request.Method != http.MethodGet || request.URL.Path != "/v1/models" {
			t.Fatalf("unexpected request: %s %s", request.Method, request.URL.String())
		}
		if request.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("unexpected authorization: %q", request.Header.Get("Authorization"))
		}
	}
}

func TestProviderCatalogCachesExplicitCapabilitiesForLocalValidation(t *testing.T) {
	hits := 0
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(*http.Request) (*http.Response, error) {
		hits++
		return response(http.StatusOK, `{"data":[{"id":"no-tools","capabilities":{"completion_chat":true,"function_calling":false}}]}`), nil
	})}

	descriptors, err := p.ListModelDescriptors(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(descriptors) != 1 || descriptors[0].ToolCalling != ai.FeatureSupportUnsupported {
		t.Fatalf("descriptors = %#v", descriptors)
	}
	descriptors[0].ToolCalling = ai.FeatureSupportSupported

	model, err := p.Model("no-tools")
	if err != nil {
		t.Fatal(err)
	}
	req := ai.AIRequest{Tools: []ai.ToolDefinition{{
		Type: "function", Name: "search", Description: "Search", Parameters: []byte(`{"type":"object"}`),
	}}}
	if err := ai.ValidateModelRequest(model, req); !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("ValidateModelRequest error = %v, want unsupported capability", err)
	}
	if hits != 1 {
		t.Fatalf("descriptor validation made a discovery request; hits = %d", hits)
	}
	again, err := p.ListModelDescriptors(context.Background())
	if err != nil || again[0].ToolCalling != ai.FeatureSupportUnsupported || hits != 1 {
		t.Fatalf("cached snapshot = %#v, err = %v, hits = %d", again, err, hits)
	}
}

func TestProviderFallsBackWhenModelDiscoveryFails(t *testing.T) {
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(*http.Request) (*http.Response, error) {
		return response(http.StatusServiceUnavailable, "unavailable"), nil
	})}

	models, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if !containsModel(models, MistralSmallLatest) {
		t.Fatalf("expected hard-coded fallback models, got %#v", models)
	}
	if _, err := p.Model(MistralSmallLatest); err != nil {
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
	if len(descriptors) == 0 || descriptors[0].Model != MistralSmallLatest {
		t.Fatalf("expected hard-coded fallback descriptors, got %#v", descriptors)
	}
}

func TestProviderDoesNotDiscoverModelsWithInvalidAPIKey(t *testing.T) {
	hits := 0
	p := New("   ", nil)
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(*http.Request) (*http.Response, error) {
		hits++
		return response(http.StatusOK, `{}`), nil
	})}
	if _, err := p.ListModels(); !errors.Is(err, ErrInvalidAPIKey) {
		t.Fatalf("expected ErrInvalidAPIKey, got %v", err)
	}
	if hits != 0 {
		t.Fatalf("expected no discovery requests, got %d", hits)
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

func TestProviderModelAndListModels(t *testing.T) {
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: errorRoundTripper{}}

	model, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model returned error: %v", err)
	}
	if model == nil {
		t.Fatalf("expected non-nil model")
	}
	if ai.ModelName(model) != MistralSmallLatest {
		t.Fatalf("unexpected model name: %q", ai.ModelName(model))
	}

	models, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if len(models) == 0 {
		t.Fatalf("expected at least one model")
	}
}

func TestProviderValidate(t *testing.T) {
	if err := (*Provider)(nil).Validate(); !errors.Is(err, ai.ErrNilProvider) {
		t.Fatalf("expected ErrNilProvider, got %v", err)
	}

	if err := New("   ", nil).Validate(); !errors.Is(err, ErrInvalidAPIKey) {
		t.Fatalf("expected ErrInvalidAPIKey, got %v", err)
	}

	if err := New("test-key", nil).Validate(); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
}
