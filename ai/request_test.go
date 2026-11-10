package ai_test

import (
	"github.com/Samuelk0nrad/gai/ai"
	"testing"
)

func TestAIRequestRequiresCanonicalMessages(t *testing.T) {
	if err := (ai.AIRequest{}).Validate(); err == nil {
		t.Fatal("empty request accepted")
	}
	if err := (ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "")}}).Validate(); err != nil {
		t.Fatal(err)
	}
}
