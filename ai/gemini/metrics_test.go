package gemini

import (
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/obstest"
)

// TestGenerationMetricsIncludeThinkingInOutputTokens verifies provider wiring
// end to end: one inference duration per call and normalized usage counters
// where Gemini thinking tokens are part of output tokens.
func TestGenerationMetricsIncludeThinkingInOutputTokens(t *testing.T) {
	parts := []json.RawMessage{json.RawMessage(`{"text":"hello "}`), json.RawMessage(`{"text":"world"}`)}
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprintf("streaming=%t", streaming), func(t *testing.T) {
			reader := obstest.InstallMetrics(t)
			model := emptyPartModel(t, func(w http.ResponseWriter, _ *http.Request) {
				if !streaming {
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write(emptyPartResponse(t, parts, true))
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				for i, part := range parts {
					_, _ = fmt.Fprintf(w, "data: %s\n\n", emptyPartResponse(t, []json.RawMessage{part}, i == len(parts)-1))
				}
			})
			request := ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hello")}}
			if streaming {
				for token := range model.GenerateStream(t.Context(), request) {
					if token.Err != nil {
						t.Fatal(token.Err)
					}
				}
			} else if _, err := model.Generate(t.Context(), request); err != nil {
				t.Fatal(err)
			}

			metrics := obstest.CollectMetrics(t, reader)
			durations := obstest.HistogramPoints[float64](t, metrics, "gen_ai.client.inference.duration")
			if len(durations) != 1 || durations[0].Count != 1 {
				t.Fatalf("duration points = %#v, want one generation", durations)
			}
			if got := obstest.AttributeStrings(durations[0].Attributes)["gen_ai.provider.name"]; got != "gcp.gemini" {
				t.Fatalf("provider = %q, want gcp.gemini", got)
			}
			// Fixture: promptTokenCount=11, candidatesTokenCount=7, thoughtsTokenCount=3.
			for name, want := range map[string]int64{
				"gen_ai.client.inference.usage.input_tokens":  11,
				"gen_ai.client.inference.usage.output_tokens": 10,
			} {
				points := obstest.SumPoints[int64](t, metrics, name)
				if len(points) != 1 || points[0].Value != want {
					t.Fatalf("%s points = %#v, want %d", name, points, want)
				}
			}
		})
	}
}
