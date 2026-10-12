package history

import (
	"testing"

	"github.com/lace-ai/gai/modelinput"
)

func TestSortTurnsByCount(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		input      []modelinput.Turn
		wantIDs    []string
		wantCounts []int
	}{
		{
			name:    "nil slice",
			input:   nil,
			wantIDs: nil,
		},
		{
			name:    "empty slice",
			input:   []modelinput.Turn{},
			wantIDs: []string{},
		},
		{
			name: "already sorted",
			input: []modelinput.Turn{
				{ID: "turn-1", Count: 1},
				{ID: "turn-2", Count: 2},
				{ID: "turn-3", Count: 3},
			},
			wantIDs:    []string{"turn-1", "turn-2", "turn-3"},
			wantCounts: []int{1, 2, 3},
		},
		{
			name: "unsorted counts",
			input: []modelinput.Turn{
				{ID: "turn-3", Count: 3},
				{ID: "turn-1", Count: 1},
				{ID: "turn-2", Count: 2},
			},
			wantIDs:    []string{"turn-1", "turn-2", "turn-3"},
			wantCounts: []int{1, 2, 3},
		},
		{
			name: "stable for duplicate counts",
			input: []modelinput.Turn{
				{ID: "turn-a", Count: 2},
				{ID: "turn-b", Count: 1},
				{ID: "turn-c", Count: 2},
				{ID: "turn-d", Count: 1},
			},
			wantIDs:    []string{"turn-b", "turn-d", "turn-a", "turn-c"},
			wantCounts: []int{1, 1, 2, 2},
		},
		{
			name: "negative and zero counts",
			input: []modelinput.Turn{
				{ID: "turn-pos", Count: 2},
				{ID: "turn-zero", Count: 0},
				{ID: "turn-neg", Count: -1},
			},
			wantIDs:    []string{"turn-neg", "turn-zero", "turn-pos"},
			wantCounts: []int{-1, 0, 2},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := sortTurnsByCount(tt.input)

			if len(got) != len(tt.wantIDs) {
				t.Fatalf("unexpected length: want %d got %d", len(tt.wantIDs), len(got))
			}
			for i := range got {
				if got[i].ID != tt.wantIDs[i] {
					t.Fatalf("unexpected turn id at %d: want %q got %q", i, tt.wantIDs[i], got[i].ID)
				}
				if len(tt.wantCounts) > 0 && got[i].Count != tt.wantCounts[i] {
					t.Fatalf("unexpected turn count at %d: want %d got %d", i, tt.wantCounts[i], got[i].Count)
				}
			}
		})
	}
}
