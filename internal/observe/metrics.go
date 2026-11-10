package observe

import (
	"reflect"
	"sync/atomic"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/noop"
)

// DurationBuckets are the OpenTelemetry GenAI advisory histogram boundaries
// for operation durations, in seconds.
var DurationBuckets = []float64{0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92}

// ErrorTypeOther is the OpenTelemetry fallback for error classes that are not
// part of a documented, bounded set.
const ErrorTypeOther = "_OTHER"

// Instruments lazily builds one domain's metric instruments from the global
// OpenTelemetry MeterProvider. Domain packages own their instrument names and
// recording logic; this type only owns provider binding.
//
// Instruments created once from otel.Meter would stay bound to the first
// provider passed to otel.SetMeterProvider, because the global delegating
// provider forwards only once. Get rebuilds the set whenever the global
// provider changes, so later SetMeterProvider calls (and per-test readers)
// receive data. With the default no-op provider no setup is required.
type Instruments[T any] struct {
	scope   string
	build   func(metric.Meter) T
	current atomic.Pointer[instrumentSet[T]]
}

type instrumentSet[T any] struct {
	provider metric.MeterProvider
	value    T
}

// NewInstruments returns a lazily bound instrument set for an instrumentation
// scope. build is called with a meter from the current global provider.
func NewInstruments[T any](scope string, build func(metric.Meter) T) *Instruments[T] {
	return &Instruments[T]{scope: scope, build: build}
}

// Get returns instruments bound to the current global MeterProvider.
func (i *Instruments[T]) Get() T {
	provider := otel.GetMeterProvider()
	cacheable := cacheableProvider(provider)
	if cacheable {
		if set := i.current.Load(); set != nil && set.provider == provider {
			return set.value
		}
	}
	value := i.build(provider.Meter(i.scope))
	if cacheable {
		i.current.Store(&instrumentSet[T]{provider: provider, value: value})
	}
	return value
}

// cacheableProvider reports whether provider identity can be compared without
// panicking. Pointer providers (the SDK and the global delegate) are cached;
// value providers such as noop.MeterProvider are cheap to rebuild.
func cacheableProvider(provider metric.MeterProvider) bool {
	return provider != nil && reflect.TypeOf(provider).Kind() == reflect.Pointer
}

// Float64Histogram creates a histogram, reporting creation errors through the
// global OpenTelemetry error handler and never returning nil.
func Float64Histogram(meter metric.Meter, name, unit, description string, buckets []float64) metric.Float64Histogram {
	options := []metric.Float64HistogramOption{metric.WithUnit(unit), metric.WithDescription(description)}
	if len(buckets) > 0 {
		options = append(options, metric.WithExplicitBucketBoundaries(buckets...))
	}
	histogram, err := meter.Float64Histogram(name, options...)
	if err != nil {
		otel.Handle(err)
	}
	if histogram == nil {
		return noop.Float64Histogram{}
	}
	return histogram
}

// Int64Histogram creates a histogram, reporting creation errors through the
// global OpenTelemetry error handler and never returning nil.
func Int64Histogram(meter metric.Meter, name, unit, description string, buckets []float64) metric.Int64Histogram {
	options := []metric.Int64HistogramOption{metric.WithUnit(unit), metric.WithDescription(description)}
	if len(buckets) > 0 {
		options = append(options, metric.WithExplicitBucketBoundaries(buckets...))
	}
	histogram, err := meter.Int64Histogram(name, options...)
	if err != nil {
		otel.Handle(err)
	}
	if histogram == nil {
		return noop.Int64Histogram{}
	}
	return histogram
}

// Int64Counter creates a counter, reporting creation errors through the global
// OpenTelemetry error handler and never returning nil.
func Int64Counter(meter metric.Meter, name, unit, description string) metric.Int64Counter {
	counter, err := meter.Int64Counter(name, metric.WithUnit(unit), metric.WithDescription(description))
	if err != nil {
		otel.Handle(err)
	}
	if counter == nil {
		return noop.Int64Counter{}
	}
	return counter
}
