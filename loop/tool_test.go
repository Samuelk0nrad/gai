package loop_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
)

type namedTestTool struct {
	name string
}

func (t namedTestTool) Name() string            { return t.name }
func (namedTestTool) Description() string       { return "A test tool." }
func (namedTestTool) Params() ai.ToolParameters { return loop.NewEchoTool().Params() }
// Function returns a fixed successful result for tool registration and invocation fixtures.
func (namedTestTool) Function(context.Context, ai.ToolCall) (string, error) {
	return "ok", nil
}

// TestCallToolRejectsNilTool verifies that a nil registered tool returns a definition error
// instead of panicking.
func TestCallToolRejectsNilTool(t *testing.T) {
	t.Parallel()

	req := ai.ToolCall{
		ID:   "call_1",
		Type: "function",
		Name: "test_tool",
		Args: json.RawMessage(`{}`),
	}

	response := loop.CallTool(context.Background(), req, []loop.Tool{nil})

	if err := response.Err; !errors.Is(err, ai.ErrInvalidToolDefinition) {
		t.Fatalf("error = %v, want ErrInvalidToolDefinition", err)
	}
}

func TestToolDefinitionsRejectNonCanonicalToolName(t *testing.T) {
	t.Parallel()

	_, err := loop.ToolDefinitions([]loop.Tool{namedTestTool{name: " search "}})
	if !errors.Is(err, ai.ErrInvalidToolDefinition) {
		t.Fatalf("ToolDefinitions() error = %v, want ErrInvalidToolDefinition", err)
	}
}

// TestDecodeToolArgs checks typed argument decoding and rejects malformed JSON or values
// incompatible with the target fields.
func TestDecodeToolArgs(t *testing.T) {
	t.Parallel()

	type SampleArgs struct {
		Foo string `json:"foo"`
		Bar int    `json:"bar"`
	}

	ValidArgs := SampleArgs{
		Foo: "hello",
		Bar: 42,
	}

	jsonArgs, err := json.Marshal(ValidArgs)
	if err != nil {
		t.Fatalf("marshal sample args: %v", err)
	}

	req := ai.ToolCall{
		ID:   "call_1",
		Type: "function",
		Name: "test_tool",
	}

	tests := []struct {
		name     string
		input    json.RawMessage
		wantErr  bool
		wantArgs SampleArgs
	}{
		{
			name:     "valid args",
			input:    jsonArgs,
			wantErr:  false,
			wantArgs: ValidArgs,
		},
		{
			name:    "invalid json",
			input:   json.RawMessage(`{"foo": "hello", "bar": }`),
			wantErr: true,
		},
		{
			name:    "invalid Tool Args",
			input:   json.RawMessage(`{"foo": "hello", "bar": "not an int"}`),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotArgs SampleArgs

			tReq := req
			tReq.Args = tt.input
			err := loop.DecodeToolArgs(tReq, &gotArgs)

			if (err != nil) != tt.wantErr {
				t.Fatalf("DecodeToolArgs() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}

			if !reflect.DeepEqual(gotArgs, tt.wantArgs) {
				t.Errorf("Decoded args = %#v, want %#v", gotArgs, tt.wantArgs)
			}
		})
	}
}
