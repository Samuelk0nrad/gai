package anthropic

import (
	"context"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
)

type handlerRoundTripper func(*http.Request) (*http.Response, error)

func (h handlerRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return h(r) }

func response(status int, body string) *http.Response {
	header := make(http.Header)
	header.Set("Content-Type", "application/json")
	return &http.Response{StatusCode: status, Status: http.StatusText(status), Header: header, Body: io.NopCloser(strings.NewReader(body))}
}

func TestProviderDynamicallyListsModelsAndAcceptsThem(t *testing.T) {
	var requests []*http.Request
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(r *http.Request) (*http.Response, error) {
		requests = append(requests, r.Clone(r.Context()))
		return response(http.StatusOK, `{"data":[{"id":" claude-dynamic ","type":"model","display_name":"Dynamic","created_at":"2026-01-01T00:00:00Z","max_input_tokens":200000,"max_tokens":8192,"capabilities":{"completion":true}}],"has_more":false,"first_id":"claude-dynamic","last_id":"claude-dynamic"}`), nil
	})}
	p.baseURL = "https://models.test"

	got, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if !reflect.DeepEqual(got, []string{"claude-dynamic"}) {
		t.Fatalf("ListModels() = %#v, want dynamic API result", got)
	}
	if _, err := p.Model("claude-dynamic"); err != nil {
		t.Fatalf("Model(dynamic) returned error: %v", err)
	}
	if len(requests) != 1 {
		t.Fatalf("expected one discovery request, got %d", len(requests))
	}
	request := requests[0]
	if request.Method != http.MethodGet || request.URL.Path != "/v1/models" {
		t.Fatalf("unexpected request: %s %s", request.Method, request.URL.String())
	}
	if request.Header.Get("x-api-key") != "test-key" {
		t.Fatalf("unexpected API key header: %q", request.Header.Get("x-api-key"))
	}
}

func TestModelDescriptorUsesCatalogSnapshotWithoutNetwork(t *testing.T) {
	hits := 0
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(*http.Request) (*http.Response, error) {
		hits++
		return response(http.StatusOK, `{"data":[{"id":"claude-dynamic","type":"model","display_name":"Dynamic","created_at":"2026-01-01T00:00:00Z","max_input_tokens":200000,"max_tokens":8192,"capabilities":{"thinking":{"supported":true},"effort":{"supported":true}}}],"has_more":false,"first_id":"claude-dynamic","last_id":"claude-dynamic"}`), nil
	})}
	if _, err := p.ListModelDescriptors(context.Background()); err != nil {
		t.Fatal(err)
	}
	model, err := p.Model("claude-dynamic")
	if err != nil {
		t.Fatal(err)
	}
	if got := model.(ai.ModelDescriber).Descriptor(); got.Reasoning != ai.FeatureSupportSupported || got.ReasoningEffort != ai.FeatureSupportSupported {
		t.Fatalf("descriptor = %#v; catalog thinking and effort facts must enable the dynamic model", got)
	}
	if hits != 1 {
		t.Fatalf("descriptor caused catalog I/O: got %d requests, want 1", hits)
	}
}

func TestProviderValidationAndModels(t *testing.T) {
	if err := (*Provider)(nil).Validate(); !errors.Is(err, ai.ErrNilProvider) {
		t.Fatalf("Validate(nil) error = %v, want ErrNilProvider", err)
	}
	if err := New("   ", nil).Validate(); !errors.Is(err, ErrInvalidAPIKey) {
		t.Fatalf("Validate(empty) error = %v, want ErrInvalidAPIKey", err)
	}

	p := New(" test-key ", nil)
	if p.apiKey != "test-key" {
		t.Fatalf("New() stored API key %q, want trimmed key", p.apiKey)
	}
	if p.httpClient.Timeout != 0 {
		t.Fatalf("client timeout = %s, want context-controlled lifetime", p.httpClient.Timeout)
	}
	if p.Name() != "anthropic" {
		t.Fatalf("Name() = %q", p.Name())
	}
	if _, err := p.Model("unknown"); err != nil {
		t.Fatalf("Model(unknown) error = %v, want dynamically selectable model", err)
	}
	got, err := p.ListModels()
	if err != nil || !reflect.DeepEqual(got, models) {
		t.Fatalf("ListModels() = %v, %v; want %v", got, err, models)
	}
	for _, name := range models {
		model, err := p.Model(name)
		if err != nil || ai.ModelName(model) != name {
			t.Fatalf("Model(%q) = %v, %v", name, model, err)
		}
	}
}

func TestAnthropicDescriptorKeepsUnknownModelFactsUnknown(t *testing.T) {
	got := (&Model{name: "claude-unfamiliar", client: New("test-key", nil)}).Descriptor()
	if got.Reasoning != ai.FeatureSupportUnknown || got.ReasoningEffort != ai.FeatureSupportUnknown {
		t.Fatalf("descriptor = %#v; unfamiliar model facts must remain unknown", got)
	}
}

func TestProviderFallsBackWhenModelDiscoveryFails(t *testing.T) {
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("model discovery unavailable")
	})}

	got, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if !reflect.DeepEqual(got, models) {
		t.Fatalf("ListModels() = %#v, want fallback %#v", got, models)
	}
	got[0] = "mutated"
	if models[0] == "mutated" {
		t.Fatal("ListModels returned the mutable fallback catalog")
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
	if len(descriptors) == 0 || descriptors[0].Model != models[0] {
		t.Fatalf("expected hard-coded fallback descriptors, got %#v", descriptors)
	}
}

func TestProviderDiscoveryUsesDeadlineAndRejectsInvalidKey(t *testing.T) {
	var deadline time.Time
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(r *http.Request) (*http.Response, error) {
		var ok bool
		deadline, ok = r.Context().Deadline()
		if !ok {
			t.Fatal("expected discovery request deadline")
		}
		return response(http.StatusOK, `{"data":[],"has_more":false,"first_id":"","last_id":""}`), nil
	})}
	p.baseURL = "https://models.test"
	if _, err := p.ListModels(); err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if time.Until(deadline) <= 0 {
		t.Fatalf("discovery deadline %s is already expired", deadline)
	}

	hits := 0
	invalid := New(" ", nil)
	invalid.httpClient = &http.Client{Transport: handlerRoundTripper(func(*http.Request) (*http.Response, error) {
		hits++
		return response(http.StatusOK, `{}`), nil
	})}
	if _, err := invalid.ListModels(); !errors.Is(err, ErrInvalidAPIKey) {
		t.Fatalf("ListModels invalid key error = %v, want ErrInvalidAPIKey", err)
	}
	if hits != 0 {
		t.Fatalf("invalid key made %d discovery requests, want 0", hits)
	}
}
