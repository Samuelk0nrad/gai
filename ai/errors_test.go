package ai_test

import (
	"errors"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestTerminalErrorIdentifiesUnsuccessfulProviderOutcome(t *testing.T) {
	err := &ai.TerminalError{Provider: "openai", Reason: "length"}
	if !errors.Is(err, ai.ErrUnsuccessfulGeneration) {
		t.Fatalf("errors.Is(%v, ErrUnsuccessfulGeneration) = false", err)
	}
	var terminal *ai.TerminalError
	if !errors.As(err, &terminal) || terminal.Provider != "openai" || terminal.Reason != "length" {
		t.Fatalf("terminal error = %#v", terminal)
	}
	if got := err.Error(); got != "openai generation ended unsuccessfully: length" {
		t.Fatalf("error = %q", got)
	}
}
