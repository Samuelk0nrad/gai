package loop_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/loop"
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

func TestNewRawSchemaToolPreservesSchemaInDefinition(t *testing.T) {
	t.Parallel()
	raw := json.RawMessage(`{
  "type": "object",
  "properties": {
    "choice": {
      "description": "Choose <a> or b & explain why",
      "oneOf": [{"const": "a"}, {"const": "b"}]
    }
  },
  "required": ["choice"],
  "additionalProperties": false
}`)

	tool, err := loop.NewRawSchemaTool("choose", "Chooses a value.", raw, func(context.Context, ai.ToolCall) (string, error) {
		return "ok", nil
	})
	if err != nil {
		t.Fatalf("NewRawSchemaTool: %v", err)
	}
	definitions, err := loop.ToolDefinitions([]loop.Tool{tool})
	if err != nil {
		t.Fatalf("ToolDefinitions: %v", err)
	}
	if len(definitions) != 1 || string(definitions[0].Parameters) != string(raw) {
		t.Fatalf("definitions = %#v, want exact raw schema %s", definitions, raw)
	}

	configured, err := loop.WithToolOptions(tool, loop.ToolOptions{Serial: true})
	if err != nil {
		t.Fatalf("WithToolOptions: %v", err)
	}
	definitions, err = loop.ToolDefinitions([]loop.Tool{configured})
	if err != nil {
		t.Fatalf("configured ToolDefinitions: %v", err)
	}
	if string(definitions[0].Parameters) != string(raw) {
		t.Fatalf("configured raw schema = %s, want %s", definitions[0].Parameters, raw)
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
