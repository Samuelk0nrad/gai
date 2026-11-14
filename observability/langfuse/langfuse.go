// Package langfuse configures an OpenTelemetry tracer provider for Langfuse's
// OTLP ingestion endpoint. It keeps Langfuse optional: applications that do not
// import this package retain the default OpenTelemetry behavior.
package langfuse

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/url"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// Config configures Langfuse's OTLP trace exporter.
type Config struct {
	// Endpoint is the Langfuse OTLP traces endpoint, for example
	// https://cloud.langfuse.com/api/public/otel/v1/traces. The bare
	// /api/public/otel base is also accepted; /v1/traces is appended to it.
	// Any other path is used as given.
	Endpoint string
	// PublicKey and SecretKey are the Langfuse project credentials.
	PublicKey string
	SecretKey string
	// ServiceName optionally identifies the emitting application.
	ServiceName string
}

// NewTracerProvider creates a batched OpenTelemetry tracer provider that exports
// traces to Langfuse. Export is asynchronous, so exporter outages do not affect
// agent execution. Call Shutdown during application shutdown to flush buffered
// spans and observe a final exporter error, if any.
func NewTracerProvider(ctx context.Context, config Config) (*sdktrace.TracerProvider, error) {
	endpoint := strings.TrimSpace(config.Endpoint)
	if endpoint == "" {
		return nil, fmt.Errorf("langfuse endpoint is required")
	}
	parsed, err := url.ParseRequestURI(endpoint)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, fmt.Errorf("invalid langfuse endpoint %q", endpoint)
	}
	endpoint = tracesEndpoint(parsed)
	publicKey := strings.TrimSpace(config.PublicKey)
	secretKey := strings.TrimSpace(config.SecretKey)
	if publicKey == "" || secretKey == "" {
		return nil, fmt.Errorf("langfuse public and secret keys are required")
	}

	exporter, err := otlptracehttp.New(ctx,
		otlptracehttp.WithEndpointURL(endpoint),
		otlptracehttp.WithHeaders(map[string]string{
			"Authorization":                "Basic " + base64.StdEncoding.EncodeToString([]byte(publicKey+":"+secretKey)),
			"x-langfuse-ingestion-version": "4",
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("create langfuse trace exporter: %w", err)
	}
	batcher := sdktrace.NewBatchSpanProcessor(exporter)
	options := []sdktrace.TracerProviderOption{sdktrace.WithSpanProcessor(NewTraceContextSpanProcessor(batcher))}
	if serviceName := strings.TrimSpace(config.ServiceName); serviceName != "" {
		options = append(options, sdktrace.WithResource(resource.NewWithAttributes("", attribute.String("service.name", serviceName))))
	}
	return sdktrace.NewTracerProvider(options...), nil
}

// langfuseOTLPBase is the path of Langfuse's OTLP API. Langfuse accepts traces
// only under its /v1/traces subpath.
const langfuseOTLPBase = "/api/public/otel"

// tracesEndpoint returns the URL to export traces to. otlptracehttp uses an
// endpoint URL's path verbatim, so a Langfuse OTLP base path gets /v1/traces
// appended; every other path is returned unchanged.
func tracesEndpoint(endpoint *url.URL) string {
	path := strings.TrimSuffix(endpoint.Path, "/")
	if !strings.HasSuffix(path, langfuseOTLPBase) {
		return endpoint.String()
	}
	resolved := *endpoint
	resolved.Path = path + "/v1/traces"
	resolved.RawPath = ""
	return resolved.String()
}
