package loop_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
)

func configuredTerminalTool(t *testing.T, tool loop.Tool) loop.Tool {
	t.Helper()
	configured, err := loop.WithToolOptions(tool, loop.ToolOptions{Terminal: true})
	if err != nil {
		t.Fatal(err)
	}
	return configured
}

func namedTerminalTool(t *testing.T, name string, fn loop.ToolFunc) loop.Tool {
	t.Helper()
	tool, err := loop.NewTool(name, name, ai.ToolParameters{}, fn)
	if err != nil {
		t.Fatal(err)
	}
	return configuredTerminalTool(t, tool)
}

func namedOrdinaryTool(t *testing.T, name string, fn loop.ToolFunc) loop.Tool {
	t.Helper()
	tool, err := loop.NewTool(name, name, ai.ToolParameters{}, fn)
	if err != nil {
		t.Fatal(err)
	}
	return tool
}

func terminalCall(id, name string, args json.RawMessage) ai.Token {
	return ai.Token{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{
		ID: id, Type: "function", Name: name, Args: args,
	}}}
}

func TestTerminalToolCallFinishesWithoutAnotherGeneration(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{terminalCall("call-1", "echo", json.RawMessage(`{"text":"presented"}`))},
		{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "must not be requested"}}},
	}}
	l := loop.New(model, []loop.Tool{configuredTerminalTool(t, loop.NewEchoTool())}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("terminal call failed: %v", err)
	}
	if got := len(model.Requests()); got != 1 {
		t.Fatalf("model requests = %d, want 1", got)
	}
	requireEventTypes(t, events,
		loop.EventAttemptStart,
		loop.EventToken,
		loop.EventToolStart,
		loop.EventToolResult,
		loop.EventIterationDone,
		loop.EventDone,
	)
	if len(l.Iterations) != 1 || len(l.Iterations[0].Conversation) != 3 {
		t.Fatalf("terminal iteration = %#v", l.Iterations)
	}
	result := l.Iterations[0].Conversation[2].ToolResults()
	if len(result) != 1 || result[0].ToolCallID != "call-1" || result[0].Text() != "presented" {
		t.Fatalf("terminal canonical result = %#v", result)
	}
}

func TestTerminalToolBatchExecutesOnceAndPreservesRequestedOrder(t *testing.T) {
	t.Parallel()

	releaseFirst := make(chan struct{})
	var textCalls, productCalls atomic.Int32
	displayText := namedTerminalTool(t, "display_text", func(ctx context.Context, _ ai.ToolCall) (string, error) {
		textCalls.Add(1)
		select {
		case <-releaseFirst:
			return "text-result", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})
	displayProducts := namedTerminalTool(t, "display_products", func(context.Context, ai.ToolCall) (string, error) {
		productCalls.Add(1)
		return "products-result", nil
	})
	model := &scriptedStreamModel{sequences: [][]ai.Token{{
		terminalCall("text", "display_text", json.RawMessage(`{}`)),
		terminalCall("products", "display_products", json.RawMessage(`{}`)),
	}}}
	l := loop.New(model, []loop.Tool{displayText, displayProducts}, testPromptBuilder(), nil)
	l.ToolExecution.MaxConcurrent = 2

	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()
	var events []loop.Event
	var completionOrder []string
	for event := range l.Run(ctx) {
		events = append(events, event)
		if event.Type != loop.EventToolResult || event.ToolCall == nil {
			continue
		}
		completionOrder = append(completionOrder, event.ToolCall.ID)
		if event.ToolCall.ID == "products" {
			close(releaseFirst)
		}
	}
	if err := loopError(events); err != nil {
		t.Fatalf("terminal batch failed: %v", err)
	}
	if textCalls.Load() != 1 || productCalls.Load() != 1 || len(model.Requests()) != 1 {
		t.Fatalf("calls text=%d products=%d requests=%d", textCalls.Load(), productCalls.Load(), len(model.Requests()))
	}
	var ids []string
	for _, message := range l.Iterations[0].Conversation {
		for _, result := range message.ToolResults() {
			ids = append(ids, result.ToolCallID)
		}
	}
	if !reflect.DeepEqual(ids, []string{"text", "products"}) {
		t.Fatalf("canonical result order = %v", ids)
	}
	if !reflect.DeepEqual(completionOrder, []string{"products", "text"}) {
		t.Fatalf("completion order = %v, want out-of-order results", completionOrder)
	}
}

func TestEmptyOrdinaryResultContinuesButEmptyTerminalResultFinishes(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name         string
		terminal     bool
		wantRequests int
	}{
		{name: "ordinary", wantRequests: 2},
		{name: "terminal", terminal: true, wantRequests: 1},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			tool := namedOrdinaryTool(t, "empty", func(context.Context, ai.ToolCall) (string, error) { return "", nil })
			if test.terminal {
				tool = configuredTerminalTool(t, tool)
			}
			model := &scriptedStreamModel{sequences: [][]ai.Token{
				{terminalCall("empty", "empty", json.RawMessage(`{}`))},
				{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "done"}}},
			}}
			l := loop.New(model, []loop.Tool{tool}, testPromptBuilder(), nil)
			l.MaxLoopIterations = 2
			if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
				t.Fatal(err)
			}
			if got := len(model.Requests()); got != test.wantRequests {
				t.Fatalf("requests = %d, want %d", got, test.wantRequests)
			}
		})
	}
}

func TestTerminalBatchCompletesOnLastIterationAfterOrdinaryRound(t *testing.T) {
	t.Parallel()

	ordinary := namedOrdinaryTool(t, "lookup", func(context.Context, ai.ToolCall) (string, error) {
		return "found", nil
	})
	terminal := namedTerminalTool(t, "present", func(context.Context, ai.ToolCall) (string, error) {
		return "", nil
	})
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{terminalCall("lookup-1", "lookup", json.RawMessage(`{}`))},
		{terminalCall("present-1", "present", json.RawMessage(`{}`))},
		{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "must not be requested"}}},
	}}
	l := loop.New(model, []loop.Tool{ordinary, terminal}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 2

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("terminal batch on final iteration failed: %v", err)
	}
	if got := len(model.Requests()); got != 2 {
		t.Fatalf("model requests = %d, want 2", got)
	}
	if got := len(l.Iterations); got != 2 {
		t.Fatalf("accepted iterations = %d, want 2", got)
	}
	var done int
	for _, event := range events {
		if event.Type == loop.EventDone {
			done++
		}
	}
	if done != 1 {
		t.Fatalf("done events = %d, want 1", done)
	}
}

func TestMixedTerminalBatchIsRejectedBeforeHandlers(t *testing.T) {
	t.Parallel()

	var invoked atomic.Int32
	handler := func(context.Context, ai.ToolCall) (string, error) {
		invoked.Add(1)
		return "ok", nil
	}
	model := &scriptedStreamModel{sequences: [][]ai.Token{{
		terminalCall("terminal", "present", json.RawMessage(`{}`)),
		terminalCall("ordinary", "lookup", json.RawMessage(`{}`)),
	}}}
	l := loop.New(model, []loop.Tool{
		namedTerminalTool(t, "present", handler),
		namedOrdinaryTool(t, "lookup", handler),
	}, testPromptBuilder(), nil)
	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); !errors.Is(err, loop.ErrMixedTerminalToolBatch) {
		t.Fatalf("error = %v, want ErrMixedTerminalToolBatch", err)
	}
	if invoked.Load() != 0 || len(model.Requests()) != 1 {
		t.Fatalf("invoked=%d requests=%d", invoked.Load(), len(model.Requests()))
	}
	for _, event := range events {
		if event.Type == loop.EventToolStart {
			t.Fatalf("mixed batch emitted tool start: %#v", events)
		}
	}
}

func TestTerminalBatchRejectsDuplicateCallIDsBeforeHandlers(t *testing.T) {
	t.Parallel()

	var invoked atomic.Int32
	model := &scriptedStreamModel{sequences: [][]ai.Token{{
		terminalCall("duplicate", "display_text", json.RawMessage(`{}`)),
		terminalCall("duplicate", "display_products", json.RawMessage(`{}`)),
	}}}
	l := loop.New(model, []loop.Tool{
		namedTerminalTool(t, "display_text", func(context.Context, ai.ToolCall) (string, error) {
			invoked.Add(1)
			return "text", nil
		}),
		namedTerminalTool(t, "display_products", func(context.Context, ai.ToolCall) (string, error) {
			invoked.Add(1)
			return "products", nil
		}),
	}, testPromptBuilder(), nil)

	err := loopError(collectLoopEvents(t, l, context.Background()))
	if !errors.Is(err, ai.ErrInvalidToolCall) {
		t.Fatalf("error = %v, want ai.ErrInvalidToolCall", err)
	}
	if got := invoked.Load(); got != 0 {
		t.Fatalf("handler invocations = %d, want 0", got)
	}
	if got := len(model.Requests()); got != 1 {
		t.Fatalf("model requests = %d, want 1", got)
	}
}

func TestRequiredTerminalCandidateWithUnknownSiblingDoesNotRegenerate(t *testing.T) {
	t.Parallel()

	var invoked atomic.Int32
	model := &scriptedStreamModel{sequences: [][]ai.Token{
		{
			terminalCall("present", "present", json.RawMessage(`{}`)),
			terminalCall("unknown", "missing", json.RawMessage(`{}`)),
		},
		{{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "must not be requested"}}},
	}}
	l := loop.New(model, []loop.Tool{
		namedTerminalTool(t, "present", func(context.Context, ai.ToolCall) (string, error) {
			invoked.Add(1)
			return "presented", nil
		}),
	}, testPromptBuilder(), nil)
	l.ToolTransport = loop.ToolTransportText
	l.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"present"}}

	err := loopError(collectLoopEvents(t, l, context.Background()))
	if !errors.Is(err, loop.ErrTerminalToolBatch) {
		t.Fatalf("error = %v, want ErrTerminalToolBatch", err)
	}
	if got := len(model.Requests()); got != 1 {
		t.Fatalf("model requests = %d, want 1", got)
	}
	if got := invoked.Load(); got != 1 {
		t.Fatalf("terminal handler invocations = %d, want 1", got)
	}
}

func TestInvalidTerminalBatchesCannotSucceed(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name      string
		calls     []ai.Token
		wantErr   error
		configure func(*loop.Loop)
		tool      func(*testing.T, *atomic.Int32) loop.Tool
	}{
		{
			name:    "malformed",
			calls:   []ai.Token{terminalCall("bad", "present", json.RawMessage(`{`))},
			wantErr: ai.ErrInvalidToolCall,
			tool: func(t *testing.T, invoked *atomic.Int32) loop.Tool {
				return namedTerminalTool(t, "present", func(context.Context, ai.ToolCall) (string, error) { invoked.Add(1); return "ok", nil })
			},
		},
		{
			name: "unknown sibling",
			calls: []ai.Token{
				terminalCall("present", "present", json.RawMessage(`{}`)),
				terminalCall("unknown", "missing", json.RawMessage(`{}`)),
			},
			tool: func(t *testing.T, invoked *atomic.Int32) loop.Tool {
				return namedTerminalTool(t, "present", func(context.Context, ai.ToolCall) (string, error) { invoked.Add(1); return "ok", nil })
			},
		},
		{
			name:  "denied",
			calls: []ai.Token{terminalCall("denied", "present", json.RawMessage(`{}`))},
			tool: func(t *testing.T, invoked *atomic.Int32) loop.Tool {
				return namedTerminalTool(t, "present", func(context.Context, ai.ToolCall) (string, error) { invoked.Add(1); return "ok", nil })
			},
			configure: func(l *loop.Loop) {
				l.ToolPolicy = loop.ToolPolicyFunc(func(context.Context, loop.ToolPolicyInput) (loop.ToolDecision, error) {
					return loop.ToolDecision{Action: loop.ToolDeny}, nil
				})
			},
		},
		{
			name:  "handler failure",
			calls: []ai.Token{terminalCall("failed", "present", json.RawMessage(`{}`))},
			tool: func(t *testing.T, invoked *atomic.Int32) loop.Tool {
				return namedTerminalTool(t, "present", func(context.Context, ai.ToolCall) (string, error) { invoked.Add(1); return "", errors.New("failed") })
			},
		},
		{
			name:  "handler canceled",
			calls: []ai.Token{terminalCall("canceled", "present", json.RawMessage(`{}`))},
			tool: func(t *testing.T, invoked *atomic.Int32) loop.Tool {
				return namedTerminalTool(t, "present", func(context.Context, ai.ToolCall) (string, error) { invoked.Add(1); return "", context.Canceled })
			},
		},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			var invoked atomic.Int32
			model := &scriptedStreamModel{sequences: [][]ai.Token{test.calls, {{Part: &ai.ContentPart{Kind: ai.ContentText, Text: "must not retry"}}}}}
			l := loop.New(model, []loop.Tool{test.tool(t, &invoked)}, testPromptBuilder(), nil)
			if test.configure != nil {
				test.configure(l)
			}
			err := loopError(collectLoopEvents(t, l, context.Background()))
			wantErr := test.wantErr
			if wantErr == nil {
				wantErr = loop.ErrTerminalToolBatch
			}
			if !errors.Is(err, wantErr) {
				t.Fatalf("error = %v, want %v", err, wantErr)
			}
			if len(model.Requests()) != 1 {
				t.Fatalf("requests = %d, want 1", len(model.Requests()))
			}
			if (test.name == "malformed" || test.name == "denied") && invoked.Load() != 0 {
				t.Fatalf("invalid handler invoked %d times", invoked.Load())
			}
			if test.name == "unknown sibling" && invoked.Load() != 1 {
				t.Fatalf("known sibling invoked %d times, want 1", invoked.Load())
			}
		})
	}
}

func TestTerminalBatchJoinsStartedSiblingBeforeFailure(t *testing.T) {
	t.Parallel()

	started := make(chan struct{})
	exited := make(chan struct{})
	var once sync.Once
	blocking := namedTerminalTool(t, "blocking", func(ctx context.Context, _ ai.ToolCall) (string, error) {
		once.Do(func() { close(started) })
		<-ctx.Done()
		close(exited)
		return "", ctx.Err()
	})
	failing := namedTerminalTool(t, "failing", func(context.Context, ai.ToolCall) (string, error) {
		<-started
		panic("private")
	})
	model := &scriptedStreamModel{sequences: [][]ai.Token{{
		terminalCall("blocking", "blocking", json.RawMessage(`{}`)),
		terminalCall("failing", "failing", json.RawMessage(`{}`)),
	}}}
	l := loop.New(model, []loop.Tool{blocking, failing}, testPromptBuilder(), nil)
	l.ToolExecution.MaxConcurrent = 2
	err := loopError(collectLoopEvents(t, l, context.Background()))
	if !errors.Is(err, loop.ErrToolPanic) {
		t.Fatalf("error = %v, want ErrToolPanic", err)
	}
	select {
	case <-exited:
	default:
		t.Fatal("started terminal sibling was not joined")
	}
}
