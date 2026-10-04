package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/internal/obstest"
)

func TestTextToolContract(t *testing.T) {
	sentinel := errors.New("unavailable")
	for _, tc := range []struct {
		name, text string
		err        error
	}{
		{name: "empty"}, {name: "text", text: "hello"},
		{name: "error", text: "must be ignored", err: fmt.Errorf("wrapped: %w", sentinel)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tool, err := NewTool("test", "test", ai.ToolParameters{}, func(context.Context, ai.ToolCall) (string, error) { return tc.text, tc.err })
			if err != nil {
				t.Fatal(err)
			}
			result := CallTool(t.Context(), ai.ToolCall{ID: "1", Type: "function", Name: "test", Args: json.RawMessage(`{}`)}, []Tool{tool})
			if !errors.Is(result.Err, tc.err) {
				t.Fatalf("result = %#v", result)
			}
			if tc.err != nil {
				if result.Text != "" || result.String() != tc.err.Error() || !errors.Is(result.Err, sentinel) {
					t.Fatalf("error precedence: %#v", result)
				}
			} else if result.Text != tc.text {
				t.Fatalf("result = %#v", result)
			}
		})
	}
}

func TestToolCallbacksOwnCallSnapshots(t *testing.T) {
	call := ai.ToolCall{ID: "1", Type: "function", Name: "test", Args: json.RawMessage(`{"x":1}`), Extensions: []ai.Extension{{Namespace: "test", Type: "state", Data: json.RawMessage(`"state"`)}}}
	original := call.Clone()
	tool, err := NewTool("test", "test", ai.ToolParameters{}, func(_ context.Context, c ai.ToolCall) (string, error) {
		c.Args[0] = '!'
		c.Extensions[0].Data[0] = '!'
		return "result", nil
	})
	if err != nil {
		t.Fatal(err)
	}
	processor := ToolResultProcessorFunc(func(_ context.Context, input ToolPolicyInput, result ToolResult) (ToolResult, error) {
		if string(input.Call.Args) != string(original.Args) || string(input.Call.Extensions[0].Data) != string(original.Extensions[0].Data) {
			t.Fatal("handler mutated processor input")
		}
		input.Call.Args[0] = '!'
		input.Call.Extensions[0].Data[0] = '!'
		return result, nil
	})
	_, _, err = processObservedTool(t.Context(), ToolPolicyInput{Call: call}, []Tool{tool}, processor)
	if err != nil {
		t.Fatal(err)
	}
	if string(call.Args) != string(original.Args) || string(call.Extensions[0].Data) != string(original.Extensions[0].Data) {
		t.Fatal("callback mutated retained call")
	}
}

func TestFunctionToolValidatesAndCopiesDeclaration(t *testing.T) {
	if _, err := NewTool("test", "test", ai.ToolParameters{}, nil); !errors.Is(err, ai.ErrInvalidToolDefinition) {
		t.Fatalf("nil function: %v", err)
	}
	params := ai.ToolParameters{Properties: []ai.ToolParameter{{Name: "value", Type: ai.ToolParameterString, Enum: []any{"original"}}}}
	tool, err := NewTool("test", "test", params, func(context.Context, ai.ToolCall) (string, error) { return "", nil })
	if err != nil {
		t.Fatal(err)
	}
	params.Properties[0].Enum[0] = "caller mutation"
	exposed := tool.Params()
	exposed.Properties[0].Enum[0] = "reader mutation"
	if got := tool.Params().Properties[0].Enum[0]; got != "original" {
		t.Fatalf("parameter alias: %v", got)
	}
	var nilEcho *EchoTool
	if _, err := ToolDefinitions([]Tool{nilEcho}); !errors.Is(err, ai.ErrInvalidToolDefinition) {
		t.Fatalf("typed nil: %v", err)
	}
	call := ai.ToolCall{ID: "1", Type: "function", Name: "echo", Args: json.RawMessage(`{}`)}
	if result := CallTool(t.Context(), call, []Tool{nilEcho}); !errors.Is(result.Err, ai.ErrInvalidToolDefinition) {
		t.Fatalf("typed nil invocation: %#v", result)
	}
	call.Args = json.RawMessage(`broken`)
	if result := CallTool(t.Context(), call, []Tool{NewEchoTool()}); !errors.Is(result.Err, ErrToolCallMalformed) {
		t.Fatalf("malformed: %#v", result)
	}
}

func TestResultProcessingPrecedesPublication(t *testing.T) {
	for _, toolError := range []bool{false, true} {
		for _, reject := range []bool{false, true} {
			t.Run(fmt.Sprintf("tool_error=%v/reject=%v", toolError, reject), func(t *testing.T) {
				recorder := obstest.Install(t)
				ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{ToolOutput: gai.CaptureEnabled})
				tool, _ := NewTool("test", "test", ai.ToolParameters{}, func(context.Context, ai.ToolCall) (string, error) {
					if toolError {
						return "", errors.New("secret-sentinel")
					}
					return "secret-sentinel", nil
				})
				l := &Loop{Tools: []Tool{tool}, ToolResultProcessor: ToolResultProcessorFunc(func(_ context.Context, _ ToolPolicyInput, result ToolResult) (ToolResult, error) {
					if !strings.Contains(result.String(), "secret-sentinel") {
						t.Fatal("processor did not receive original result")
					}
					if reject {
						return ToolResult{}, errors.New("filter unavailable")
					}
					return ToolResult{Err: errors.New("safe replacement")}, nil
				})}
				iteration := &Iteration{Parts: make([]IterationPart, 1)}
				events := make(chan Event, 2)
				err := l.executeToolCalls(ctx, iteration, []pendingToolCall{{partIndex: 0, call: ai.ToolCall{ID: "1", Type: "function", Name: "test", Args: json.RawMessage(`{}`)}}}, l.Tools, events, 1, 1, 0)
				if reject != (err != nil) {
					t.Fatalf("error=%v", err)
				}
				close(events)
				for event := range events {
					if strings.Contains(fmt.Sprint(event.ToolResult, event.Err), "secret-sentinel") {
						t.Fatal("raw result escaped in event")
					}
				}
				if reject && iteration.Parts[0].ToolResp != nil {
					t.Fatal("failed filter retained raw output")
				}
				if !reject && (len(iteration.Conversation) != 1 || !iteration.Conversation[0].Parts[0].ToolResult.IsError) {
					t.Fatal("processed error was not serialized")
				}
				for _, span := range requireToolSpans(t, recorder, 1) {
					if strings.Contains(toolSpanText(span), "secret-sentinel") {
						t.Fatal("raw result escaped in telemetry")
					}
				}
			})
		}
	}
}

func TestFunctionToolPreservesSchemaNumbers(t *testing.T) {
	params := ai.ToolParameters{Properties: []ai.ToolParameter{{Name: "number", Type: ai.ToolParameterInteger, Enum: []any{int64(9007199254740993), json.Number("9007199254740995")}, Default: int64(9007199254740993)}}}
	original, err := params.JSONSchema()
	if err != nil {
		t.Fatal(err)
	}
	tool, err := NewTool("test", "test", params, func(context.Context, ai.ToolCall) (string, error) { return "", nil })
	if err != nil {
		t.Fatal(err)
	}
	copied, err := tool.Params().JSONSchema()
	if err != nil {
		t.Fatal(err)
	}
	if string(original) != string(copied) {
		t.Fatalf("schema changed: %s -> %s", original, copied)
	}
}
