package observe

import (
	"context"
	"testing"

	"github.com/lace-ai/gai/internal/obstest"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/noop"
)

type testInstruments struct {
	counter metric.Int64Counter
}

func newTestInstruments(builds *int) *Instruments[testInstruments] {
	return NewInstruments("github.com/lace-ai/gai/internal/observe/test", func(meter metric.Meter) testInstruments {
		*builds++
		return testInstruments{counter: Int64Counter(meter, "gai.test.count", "{test}", "Test counter.")}
	})
}

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

func TestInstrumentsWorkWithValueProviders(t *testing.T) {
	previous := otel.GetMeterProvider()
	t.Cleanup(func() { otel.SetMeterProvider(previous) })
	otel.SetMeterProvider(noop.NewMeterProvider())

	builds := 0
	instruments := newTestInstruments(&builds)
	instruments.Get().counter.Add(context.Background(), 1)
	instruments.Get().counter.Add(context.Background(), 1)
	if builds != 2 {
		t.Fatalf("builds = %d, want uncached rebuilds for value providers", builds)
	}
}
