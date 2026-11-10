package observe

import (
	"context"
	"testing"

	"github.com/Samuelk0nrad/gai/internal/obstest"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/noop"
)

type testInstruments struct {
	counter metric.Int64Counter
}

// newTestInstruments returns an instrument set that counts how often it is
// rebuilt.
func newTestInstruments(builds *int) *Instruments[testInstruments] {
	return NewInstruments("github.com/Samuelk0nrad/gai/internal/observe/test", func(meter metric.Meter) testInstruments {
		*builds++
		return testInstruments{counter: Int64Counter(meter, "gai.test.count", "{test}", "Test counter.")}
	})
}

// TestInstrumentsRebindWhenGlobalProviderChanges checks that instruments
// follow later SetMeterProvider calls instead of the first provider only.
func TestInstrumentsRebindWhenGlobalProviderChanges(t *testing.T) {
	builds := 0
	instruments := newTestInstruments(&builds)

	first := obstest.InstallMetrics(t)
	instruments.Get().counter.Add(context.Background(), 1)
	instruments.Get().counter.Add(context.Background(), 1)
	if builds != 1 {
		t.Fatalf("builds = %d, want 1 while the provider is unchanged", builds)
	}

	// A second provider must receive data even though the global delegate was
	// already bound to the first one.
	second := obstest.InstallMetrics(t)
	instruments.Get().counter.Add(context.Background(), 5)
	if builds != 2 {
		t.Fatalf("builds = %d, want rebuild after provider change", builds)
	}

	if points := obstest.SumPoints[int64](t, obstest.CollectMetrics(t, first), "gai.test.count"); len(points) != 1 || points[0].Value != 2 {
		t.Fatalf("first provider points = %#v, want one point with 2", points)
	}
	if points := obstest.SumPoints[int64](t, obstest.CollectMetrics(t, second), "gai.test.count"); len(points) != 1 || points[0].Value != 5 {
		t.Fatalf("second provider points = %#v, want one point with 5", points)
	}
}

// TestInstrumentsCacheComparableValueProviders checks that comparable
// non-pointer providers, such as noop.MeterProvider, are cached.
func TestInstrumentsCacheComparableValueProviders(t *testing.T) {
	setMeterProvider(t, noop.NewMeterProvider())

	builds := 0
	instruments := newTestInstruments(&builds)
	instruments.Get().counter.Add(context.Background(), 1)
	instruments.Get().counter.Add(context.Background(), 1)
	if builds != 1 {
		t.Fatalf("builds = %d, want one cached build for a comparable value provider", builds)
	}
}

// uncomparableProvider is a MeterProvider whose dynamic type cannot be
// compared with ==.
type uncomparableProvider struct {
	noop.MeterProvider
	tags []string
}

// TestInstrumentsRebuildUncomparableProviders checks that providers whose
// identity cannot be compared are rebuilt without panicking, including after
// a comparable provider was cached.
func TestInstrumentsRebuildUncomparableProviders(t *testing.T) {
	builds := 0
	instruments := newTestInstruments(&builds)

	setMeterProvider(t, noop.NewMeterProvider())
	instruments.Get()
	setMeterProvider(t, uncomparableProvider{tags: []string{"a"}})
	instruments.Get().counter.Add(context.Background(), 1)
	instruments.Get().counter.Add(context.Background(), 1)
	if builds != 3 {
		t.Fatalf("builds = %d, want a rebuild for each uncomparable provider call", builds)
	}
}

// setMeterProvider installs provider globally until the test ends.
func setMeterProvider(t *testing.T, provider metric.MeterProvider) {
	t.Helper()
	previous := otel.GetMeterProvider()
	t.Cleanup(func() { otel.SetMeterProvider(previous) })
	otel.SetMeterProvider(provider)
}
