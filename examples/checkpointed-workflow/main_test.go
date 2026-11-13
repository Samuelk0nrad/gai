package main

import (
	"bytes"
	"testing"
)

func TestApprovalBeforeEffectAndFreshAgentResume(t *testing.T) {
	var output bytes.Buffer
	if err := run(t.Context(), &output); err != nil {
		t.Fatal(err)
	}
	want := "Paused for approval; effects: 0\nSaved after approval. Effects: 1\n"
	if output.String() != want {
		t.Fatalf("output = %q", output.String())
	}
}
