package langfuse

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Samuelk0nrad/gai"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestNewTracerProviderRejectsMissingCredentials(t *testing.T) {
	_, err := NewTracerProvider(context.Background(), Config{Endpoint: "https://langfuse.example/api/public/otel"})
	if err == nil {
		t.Fatal("expected missing credentials error")
	}
}

func TestNewTracerProviderRejectsInvalidEndpoint(t *testing.T) {
	_, err := NewTracerProvider(context.Background(), Config{PublicKey: "pk", SecretKey: "sk", Endpoint: "://invalid"})
	if err == nil {
		t.Fatal("expected invalid endpoint error")
	}
}

func TestNewTracerProviderSendsLangfuseIngestionVersion(t *testing.T) {
	headers := make(chan http.Header, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headers <- r.Header
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	provider, err := NewTracerProvider(context.Background(), Config{
		Endpoint:  server.URL,
		PublicKey: "pk",
		SecretKey: "sk",
	})
	if err != nil {
		t.Fatalf("NewTracerProvider() error = %v", err)
	}
	_, span := provider.Tracer("test").Start(context.Background(), "test-span")
	span.End()
	if err := provider.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown() error = %v", err)
	}

	select {
	case got := <-headers:
		if got.Get("x-langfuse-ingestion-version") != "4" {
			t.Fatalf("x-langfuse-ingestion-version = %q, want %q", got.Get("x-langfuse-ingestion-version"), "4")
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for exported span")
	}
}

func TestTraceContextSpanProcessorMapsEveryLangfuseField(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(NewTraceContextSpanProcessor(recorder)))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })

	ctx := gai.WithTraceContext(t.Context(), gai.TraceContext{
		Name:        "chat",
		UserID:      "user-1",
		SessionID:   "session-1",
		Tags:        []string{"mobile", "dogfood"},
		Release:     "2026.08",
		Environment: "staging",
		Metadata: map[string]string{
			"feature":     "lace-chat",
			"tier":        "pro",
			"invalid_key": "hidden",
			"invalid-key": "hidden",
			"invalid.key": "hidden",
		},
	})
	_, span := provider.Tracer("langfuse-test").Start(ctx, "child")
	span.End()

	ended := recorder.Ended()
	if len(ended) != 1 {
		t.Fatalf("ended spans = %d, want 1", len(ended))
	}
	attributes := map[string]any{}
	for _, attr := range ended[0].Attributes() {
		attributes[string(attr.Key)] = attr.Value.AsInterface()
	}
	want := map[string]string{
		"langfuse.trace.name":             "chat",
		"langfuse.user.id":                "user-1",
		"langfuse.session.id":             "session-1",
		"langfuse.release":                "2026.08",
		"langfuse.environment":            "staging",
		"langfuse.trace.metadata.feature": "lace-chat",
		"langfuse.trace.metadata.tier":    "pro",
	}
	for key, value := range want {
		if attributes[key] != value {
			t.Errorf("attribute %q = %#v, want %#v", key, attributes[key], value)
		}
	}
	if _, exists := attributes["langfuse.trace.release"]; exists {
		t.Fatalf("unexpected legacy release attribute in %#v", attributes)
	}
	for _, key := range []string{"langfuse.trace.metadata.invalid_key", "langfuse.trace.metadata.invalid-key", "langfuse.trace.metadata.invalid.key"} {
		if _, exists := attributes[key]; exists {
			t.Fatalf("invalid metadata key %q was mapped: %#v", key, attributes)
		}
	}
	tags, ok := attributes["langfuse.trace.tags"].([]string)
	if !ok || len(tags) != 2 || tags[0] != "mobile" || tags[1] != "dogfood" {
		t.Fatalf("langfuse tags = %#v", attributes["langfuse.trace.tags"])
	}
}

func TestNewTracerProviderExportsToLangfuseTracesPath(t *testing.T) {
	tests := []struct {
		name      string
		localhost bool
		path      string
		wantPath  string
	}{
		{name: "bare base", path: "/api/public/otel", wantPath: "/api/public/otel/v1/traces"},
		{name: "base with trailing slash", path: "/api/public/otel/", wantPath: "/api/public/otel/v1/traces"},
		{name: "full traces URL", path: "/api/public/otel/v1/traces", wantPath: "/api/public/otel/v1/traces"},
		{name: "self-hosted localhost base", localhost: true, path: "/api/public/otel", wantPath: "/api/public/otel/v1/traces"},
		{name: "base under a prefix", path: "/langfuse/api/public/otel", wantPath: "/langfuse/api/public/otel/v1/traces"},
		{name: "unrelated path", path: "/collector/v1/traces", wantPath: "/collector/v1/traces"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := exportedPath(t, tt.localhost, tt.path); got != tt.wantPath {
				t.Fatalf("export path = %q, want %q", got, tt.wantPath)
			}
		})
	}
}

// exportedPath exports one span to an httptest server at the given endpoint
// path and returns the request path the exporter sent it to. With localhost
// set, the endpoint names the server as http://localhost:<port>.
func exportedPath(t *testing.T, localhost bool, endpointPath string) string {
	t.Helper()
	paths := make(chan string, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case paths <- r.URL.Path:
		default:
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	base := server.URL
	if localhost {
		base = strings.Replace(base, "127.0.0.1", "localhost", 1)
	}
	provider, err := NewTracerProvider(context.Background(), Config{
		Endpoint:  base + endpointPath,
		PublicKey: "pk",
		SecretKey: "sk",
	})
	if err != nil {
		t.Fatalf("NewTracerProvider() error = %v", err)
	}
	defer func() { _ = provider.Shutdown(context.Background()) }()

	_, span := provider.Tracer("test").Start(context.Background(), "test-span")
	span.End()
	if err := provider.ForceFlush(context.Background()); err != nil {
		t.Fatalf("ForceFlush() error = %v", err)
	}

	select {
	case got := <-paths:
		return got
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for exported span")
		return ""
	}
}
