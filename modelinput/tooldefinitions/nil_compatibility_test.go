package tooldefinitions_test

import (
	"testing"

	"github.com/lace-ai/gai/modelinput/tooldefinitions"
)

func TestNewAcceptsUntypedNilTools(t *testing.T) {
	_, err := tooldefinitions.New(nil, nil, nil)
	if err != tooldefinitions.ErrToolsEmpty {
		t.Fatalf("New error = %v, want ErrToolsEmpty", err)
	}
}
