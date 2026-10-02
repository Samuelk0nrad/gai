package history

import (
	"github.com/lace-ai/gai/ai"
	"strings"
	"testing"

	gaictx "github.com/lace-ai/gai/context"
)

func TestWriteTurnHandlesNilMessageContent(t *testing.T) {
	t.Parallel()

	turn := gaictx.Turn{
		UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "")},
		Messages:    []gaictx.StoredMessage{{Message: ai.TextMessage(ai.RoleAssistant, "")}},
	}
	var builder strings.Builder

	if err := writeTurn(t.Context(), &builder, &turn); err != nil {
		t.Fatal(err)
	}

	if got, want := builder.String(), "<user>\n\n</user>\n<assistant>\n\n</assistant>\n\n"; got != want {
		t.Fatalf("unexpected serialized turn: want %q got %q", want, got)
	}
}
