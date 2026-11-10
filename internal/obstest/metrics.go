package obstest

import (
	"context"
	"slices"
	"strings"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
)

// InstallMetrics installs an in-memory metric reader for the duration of the
// test. It replaces the global OpenTelemetry meter provider and must not be
// used with t.Parallel() or concurrently with another InstallMetrics call.
func InstallMetrics(t testing.TB) *sdkmetric.ManualReader {
	t.Helper()
	previous := otel.GetMeterProvider()
	reader := sdkmetric.NewManualReader()
	provider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	otel.SetMeterProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetMeterProvider(previous)
	})
	return reader
}

// CollectMetrics returns every metric currently aggregated by reader.
func CollectMetrics(t testing.TB, reader *sdkmetric.ManualReader) []metricdata.Metrics {
	t.Helper()
	var resourceMetrics metricdata.ResourceMetrics
	if err := reader.Collect(context.Background(), &resourceMetrics); err != nil {
		t.Fatalf("collect metrics: %v", err)
	}
	var metrics []metricdata.Metrics
	for _, scope := range resourceMetrics.ScopeMetrics {
		metrics = append(metrics, scope.Metrics...)
	}
	return metrics
}

// FindMetric returns the metric named name, if it was recorded.
func FindMetric(metrics []metricdata.Metrics, name string) (metricdata.Metrics, bool) {
	for _, metric := range metrics {
		if metric.Name == name {
			return metric, true
		}
	}
	return metricdata.Metrics{}, false
}

// HistogramPoints returns the data points of a histogram named name. It fails
// the test when the metric exists with another aggregation.
func HistogramPoints[N int64 | float64](t testing.TB, metrics []metricdata.Metrics, name string) []metricdata.HistogramDataPoint[N] {
	t.Helper()
	metric, ok := FindMetric(metrics, name)
	if !ok {
		return nil
	}
	histogram, ok := metric.Data.(metricdata.Histogram[N])
	if !ok {
		t.Fatalf("metric %s data = %T, want histogram", name, metric.Data)
	}
	return histogram.DataPoints
}

// SumPoints returns the data points of a sum named name. It fails the test when
// the metric exists with another aggregation.
func SumPoints[N int64 | float64](t testing.TB, metrics []metricdata.Metrics, name string) []metricdata.DataPoint[N] {
	t.Helper()
	metric, ok := FindMetric(metrics, name)
	if !ok {
		return nil
	}
	sum, ok := metric.Data.(metricdata.Sum[N])
	if !ok {
		t.Fatalf("metric %s data = %T, want sum", name, metric.Data)
	}
	return sum.DataPoints
}

// AttributeStrings returns a data-point attribute set as key/value strings.
func AttributeStrings(set attribute.Set) map[string]string {
	result := make(map[string]string, set.Len())
	for _, attr := range set.ToSlice() {
		result[string(attr.Key)] = attr.Value.String()
	}
	return result
}

// RequireMetricAttributesAllowed fails when any data point carries an
// attribute key outside allowed or an attribute value containing a forbidden
// substring. It guards metric cardinality and content leaks without asserting
// exact values.
func RequireMetricAttributesAllowed(t testing.TB, metrics []metricdata.Metrics, allowed []string, forbidden ...string) {
	t.Helper()
	for _, metric := range metrics {
		for _, set := range dataPointAttributes(metric) {
			for _, attr := range set.ToSlice() {
				if !slices.Contains(allowed, string(attr.Key)) {
					t.Errorf("metric %s has unexpected attribute %s", metric.Name, attr.Key)
				}
				value := attr.Value.String()
				for _, secret := range forbidden {
					if strings.Contains(value, secret) {
						t.Errorf("metric %s attribute %s contains forbidden value %q", metric.Name, attr.Key, secret)
					}
				}
			}
		}
	}
}

func dataPointAttributes(metric metricdata.Metrics) []attribute.Set {
	var sets []attribute.Set
	switch data := metric.Data.(type) {
	case metricdata.Histogram[float64]:
		for _, point := range data.DataPoints {
			sets = append(sets, point.Attributes)
		}
	case metricdata.Histogram[int64]:
		for _, point := range data.DataPoints {
			sets = append(sets, point.Attributes)
		}
	case metricdata.Sum[int64]:
		for _, point := range data.DataPoints {
			sets = append(sets, point.Attributes)
		}
	case metricdata.Sum[float64]:
		for _, point := range data.DataPoints {
			sets = append(sets, point.Attributes)
		}
	}
	return sets
}
