package agent

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/Samuelk0nrad/gai/modelinput/history"
)

func TestCheckpointOverridesKeepHistoryCompactionPolicy(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   Optional[*HistoryCompactionPolicy]
	}{
		{"inherit", Optional[*HistoryCompactionPolicy]{}},
		{"disabled", Optional[*HistoryCompactionPolicy]{Set: true}},
		{"policy", Optional[*HistoryCompactionPolicy]{Set: true, Value: &HistoryCompactionPolicy{OnFailure: HistoryCompactionContinue, Amount: .5, SummaryMaxTokens: 300, ResponseFormat: history.SummaryResponseFormatKeep}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(snapshotOverrides(&ExecutionOverrides{HistoryCompaction: tc.in}))
			if err != nil {
				t.Fatal(err)
			}
			var decoded checkpointOverrides
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatal(err)
			}
			if got := restoreOverrides(&decoded).HistoryCompaction; !reflect.DeepEqual(got, tc.in) {
				t.Fatalf("restored %+v, want %+v", got, tc.in)
			}
		})
	}
}

func TestCheckpointOverridesReadPrePolicyAutoCompaction(t *testing.T) {
	for raw, want := range map[string]Optional[*HistoryCompactionPolicy]{
		`{"limits":{},"request_budget":{"Set":false,"Value":null},"auto_compact_history":true}`:  {Set: true, Value: &HistoryCompactionPolicy{}},
		`{"limits":{},"request_budget":{"Set":false,"Value":null},"auto_compact_history":false}`: {Set: true},
	} {
		var decoded checkpointOverrides
		if err := json.Unmarshal([]byte(raw), &decoded); err != nil {
			t.Fatal(err)
		}
		if got := restoreOverrides(&decoded).HistoryCompaction; !reflect.DeepEqual(got, want) {
			t.Fatalf("%s: restored %+v", raw, got)
		}
	}
}

func TestDurableOverridesRejectHistorySummarizer(t *testing.T) {
	overrides := &ExecutionOverrides{HistoryCompaction: Optional[*HistoryCompactionPolicy]{Set: true, Value: &HistoryCompactionPolicy{Summarizer: history.NewContinuationSummarizer(nil)}}}
	if err := validateDurableOverrides(overrides); !errors.Is(err, ErrDurability) {
		t.Fatal(err)
	}
	overrides.HistoryCompaction.Value.Summarizer = nil
	if err := validateDurableOverrides(overrides); err != nil {
		t.Fatal(err)
	}
}
