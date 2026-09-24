package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/testutil/mocks"
)

func outputText(events []agent.Event) string {
	var text string
	for _, event := range events {
		if event.Type == agent.EventOutput && event.Output != nil && event.Output.Kind == agent.OutputText {
			text += event.Output.Text
		}
	}
	return text
}

func collectAgentEvents(stream <-chan agent.Event) []agent.Event {
	var events []agent.Event
	for event := range stream {
		events = append(events, event)
	}
	return events
}

func TestWorkflowUnifiedBlockingLifecycle(t *testing.T) {
	workflow, err := workflowAgent("main", "answer").NewRun(context.Background(), textRunInput("question"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if _, err := workflow.Wait(); !errors.Is(err, agent.ErrWorkflowNotStarted) {
		t.Fatalf("Wait before start error = %v", err)
	}

	result, err := workflow.Run(context.Background())
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if !result.Complete || result.Text != "answer" || result.Primary.Text != "answer" {
		t.Fatalf("unexpected result: %+v", result)
	}
	if got, want := result.Output, []agent.OutputPart{{Kind: agent.OutputText, Text: "answer"}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("output = %#v, want %#v", got, want)
	}

	for i := 0; i < 2; i++ {
		waited, err := workflow.Wait()
		if err != nil || !reflect.DeepEqual(waited.Output, result.Output) {
			t.Fatalf("Wait %d = (%+v, %v)", i, waited, err)
		}
	}
	if _, err := workflow.Run(context.Background()); !errors.Is(err, agent.ErrWorkflowAlreadyRun) {
		t.Fatalf("repeated Run error = %v", err)
	}
}

func TestWorkflowRunEventsUsesMiddlewarePipeline(t *testing.T) {
	post := workflowAgent("post", " polished")
	main := workflowAgent("main", "answer", agent.NewAgentMiddleware(post, agent.AgentMiddlewareConfig{
		Name:   "polisher",
		Output: agent.AppendOutput,
	}))
	workflow, err := main.NewRun(context.Background(), textRunInput("question"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}

	events := collectAgentEvents(workflow.RunEvents(context.Background()))
	if got := outputText(events); got != "answer polished" {
		t.Fatalf("visible output = %q", got)
	}
	if len(events) == 0 || events[len(events)-1].Type != agent.EventDone {
		t.Fatalf("last event = %#v", events)
	}
	terminalCount := 0
	middlewareOutput := false
	for _, event := range events {
		switch event.Type {
		case agent.EventDone, agent.EventError, agent.EventCanceled:
			terminalCount++
		}
		if event.Type == agent.EventOutput && event.Source.Kind == agent.SourceMiddleware {
			middlewareOutput = event.Source.Index == 1 && event.Source.Name == "polisher"
		}
	}
	if terminalCount != 1 || !middlewareOutput {
		t.Fatalf("terminalCount=%d middlewareOutput=%v events=%#v", terminalCount, middlewareOutput, events)
	}
	var stageOrder []agent.EventType
	for _, event := range events {
		if event.Source.Kind == agent.SourceMiddleware && event.Source.Index == 1 {
			stageOrder = append(stageOrder, event.Type)
		}
	}
	if want := []agent.EventType{agent.EventStageStart, agent.EventAttemptStart, agent.EventOutput, agent.EventIterationDone, agent.EventStageFinish}; !reflect.DeepEqual(stageOrder, want) {
		t.Fatalf("middleware event order = %v, want %v", stageOrder, want)
	}

	result, err := workflow.Wait()
	if err != nil || !result.Complete || result.Text != "answer polished" || len(result.Stages) != 1 {
		t.Fatalf("Wait = (%+v, %v)", result, err)
	}
}

func TestAgentMiddlewareReplaceOutputRestoresOnlyAcceptedUpstreamAttempts(t *testing.T) {
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{
		{{Type: ai.TokenTypeText, Text: "partial"}, {Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("retry")}}},
		{{Type: ai.TokenTypeText, Text: "final"}},
	}}
	main := agent.New(agent.Definition{
		Name:        "main",
		Model:       model,
		RetryPolicy: &loop.RetryPolicy{MaxRetries: 1},
		Limits:      agent.Limits{MaxLoopIterations: 1},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
		Middleware: []agent.Middleware{
			agent.NewAgentMiddleware(workflowAgent("post", "unused"), agent.AgentMiddlewareConfig{
				Output:    agent.ReplaceOutput,
				ShouldRun: func(agent.WorkflowResult) bool { return false },
			}),
		},
	})
	workflow, err := main.NewRun(context.Background(), textRunInput("question"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}

	events := collectAgentEvents(workflow.RunEvents(context.Background()))
	result, err := workflow.Wait()
	if err != nil {
		t.Fatalf("Wait failed: %v", err)
	}
	if got, want := outputText(events), result.Text; got != want {
		t.Fatalf("streamed output = %q, reduced result = %q", got, want)
	}
	if result.Text != "final" {
		t.Fatalf("result text = %q, want final", result.Text)
	}
}

func TestWorkflowCanceledAbandonedEventStreamStillCompletes(t *testing.T) {
	stopsOnCancellation := agent.MiddlewareFunc(func(ctx context.Context, _ *agent.MiddlewareContext, _ <-chan agent.Event) <-chan agent.Event {
		out := make(chan agent.Event)
		close(out)
		return out
	})
	workflow, err := workflowAgent("main", "answer", stopsOnCancellation).NewRun(context.Background(), textRunInput("question"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_ = workflow.RunEvents(ctx)
	waited := make(chan struct{})
	var result agent.WorkflowResult
	var waitErr error
	go func() {
		result, waitErr = workflow.Wait()
		close(waited)
	}()

	select {
	case <-waited:
	case <-time.After(time.Second):
		t.Fatal("Wait did not complete after cancellation with an abandoned event stream")
	}
	if !errors.Is(waitErr, context.Canceled) || !result.Complete || !result.Canceled {
		t.Fatalf("Wait = (%+v, %v), want complete canceled result", result, waitErr)
	}
}

func TestWorkflowCanceledDelayedConsumerReceivesTerminalEvent(t *testing.T) {
	emitsBeforeCancel := agent.MiddlewareFunc(func(_ context.Context, run *agent.MiddlewareContext, _ <-chan agent.Event) <-chan agent.Event {
		out := make(chan agent.Event, 1)
		out <- agent.Event{Type: agent.EventOutput, Source: run.Source(), Output: &agent.OutputPart{Kind: agent.OutputText, Text: "pending"}}
		close(out)
		return out
	})
	workflow, err := workflowAgent("main", "answer", emitsBeforeCancel).NewRun(context.Background(), textRunInput("question"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	stream := workflow.RunEvents(ctx)
	result, waitErr := workflow.Wait()
	if !errors.Is(waitErr, context.Canceled) || !result.Complete || !result.Canceled {
		t.Fatalf("Wait = (%+v, %v), want complete canceled result", result, waitErr)
	}

	events := collectAgentEvents(stream)
	if len(events) != 1 || events[0].Type != agent.EventCanceled || !errors.Is(events[0].Err, context.Canceled) {
		t.Fatalf("delayed events = %#v, want one canceled terminal event", events)
	}
}

func TestWorkflowWaitDoesNotConsumeEventStream(t *testing.T) {
	gate := make(chan struct{})
	middleware := agent.MiddlewareFunc(func(ctx context.Context, run *agent.MiddlewareContext, upstream <-chan agent.Event) <-chan agent.Event {
		out := make(chan agent.Event)
		go func() {
			defer close(out)
			for event := range upstream {
				out <- event
			}
			<-gate
			for i := 0; i < 64; i++ {
				out <- agent.Event{Type: agent.EventOutput, Source: run.Source(), Output: &agent.OutputPart{Kind: agent.OutputText, Text: "x"}}
			}
		}()
		return out
	})
	workflow, err := workflowAgent("main", "answer", middleware).NewRun(context.Background(), textRunInput("question"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	stream := workflow.RunEvents(context.Background())
	close(gate)

	waited := make(chan struct{})
	go func() {
		_, _ = workflow.Wait()
		close(waited)
	}()
	select {
	case <-waited:
		t.Fatal("Wait completed without the public event stream being drained")
	case <-time.After(20 * time.Millisecond):
	}

	events := collectAgentEvents(stream)
	select {
	case <-waited:
	case <-time.After(time.Second):
		t.Fatal("Wait did not complete after the stream was drained")
	}
	if got := outputText(events); len(got) != len("answer")+64 {
		t.Fatalf("unexpected output length: %d", len(got))
	}
}

func TestWorkflowConcurrentWaitersReceiveSameResult(t *testing.T) {
	workflow, err := workflowAgent("main", "answer").NewRun(context.Background(), textRunInput("question"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	stream := workflow.RunEvents(context.Background())

	const waiterCount = 8
	results := make([]agent.WorkflowResult, waiterCount)
	errs := make([]error, waiterCount)
	var wg sync.WaitGroup
	for i := range waiterCount {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results[i], errs[i] = workflow.Wait()
		}()
	}
	collectAgentEvents(stream)
	wg.Wait()
	for i := range waiterCount {
		if errs[i] != nil || results[i].Text != "answer" || !results[i].Complete {
			t.Fatalf("waiter %d = (%+v, %v)", i, results[i], errs[i])
		}
	}
}

func TestWorkflowRejectsCompetingStarts(t *testing.T) {
	workflow, err := workflowAgent("main", "answer").NewRun(context.Background(), textRunInput("question"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	errs := make(chan error, 2)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			_, err := workflow.Run(context.Background())
			errs <- err
		}()
	}
	close(start)
	wg.Wait()
	close(errs)
	var successes, rejected int
	for err := range errs {
		switch {
		case err == nil:
			successes++
		case errors.Is(err, agent.ErrWorkflowAlreadyRun):
			rejected++
		default:
			t.Fatalf("unexpected competing start error: %v", err)
		}
	}
	if successes != 1 || rejected != 1 {
		t.Fatalf("successes=%d rejected=%d", successes, rejected)
	}
}

func TestMiddlewareCanEmitStructuredOutputWithUpstreamIdentity(t *testing.T) {
	transform := agent.MiddlewareFunc(func(ctx context.Context, run *agent.MiddlewareContext, upstream <-chan agent.Event) <-chan agent.Event {
		out := make(chan agent.Event)
		go func() {
			defer close(out)
			for event := range upstream {
				if event.Type == agent.EventOutput && event.Output != nil && event.Output.Kind == agent.OutputText {
					identity := event
					identity.Output = &agent.OutputPart{Kind: agent.OutputData, Data: &agent.StructuredOutput{Name: "product", JSON: json.RawMessage(`{"id":"product-123"}`)}}
					out <- identity
					continue
				}
				out <- event
			}
		}()
		return out
	})
	workflow, err := workflowAgent("main", "product-123", transform).NewRun(context.Background(), agent.RunInput{Prompt: gaictx.PromptInput{User: gaictx.NewTextContent("question")}})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	events := collectAgentEvents(workflow.RunEvents(context.Background()))
	var dataEvent *agent.Event
	for i := range events {
		if events[i].Type == agent.EventOutput {
			dataEvent = &events[i]
			break
		}
	}
	if dataEvent == nil || dataEvent.Output == nil || dataEvent.Output.Data == nil || dataEvent.Source.Kind != agent.SourcePrimary || dataEvent.AttemptID == 0 {
		t.Fatalf("structured output lost upstream identity: %#v", dataEvent)
	}
	dataEvent.Output.Data.JSON[0] = 'X'
	result, err := workflow.Wait()
	if err != nil || len(result.Output) != 1 || result.Output[0].Data == nil || string(result.Output[0].Data.JSON) != `{"id":"product-123"}` {
		t.Fatalf("structured result = (%+v, %v)", result, err)
	}

	snapshot := workflow.Result()
	snapshot.Output[0].Data.JSON[0] = 'X'
	if got := string(workflow.Result().Output[0].Data.JSON); got != `{"id":"product-123"}` {
		t.Fatalf("Result returned shared structured JSON: %q", got)
	}
}

func TestStructuredOutputFromRetriedAttemptIsExcluded(t *testing.T) {
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{
		{{Type: ai.TokenTypeText, Text: "discarded"}, {Err: &ai.ProviderError{Kind: ai.ProviderErrorTransient, Err: errors.New("retry")}}},
		{{Type: ai.TokenTypeText, Text: "accepted"}},
	}}
	transform := agent.MiddlewareFunc(func(ctx context.Context, run *agent.MiddlewareContext, upstream <-chan agent.Event) <-chan agent.Event {
		out := make(chan agent.Event)
		go func() {
			defer close(out)
			for event := range upstream {
				if event.Type == agent.EventOutput && event.Output != nil {
					event.Output = &agent.OutputPart{Kind: agent.OutputData, Data: &agent.StructuredOutput{Name: "value", JSON: json.RawMessage(`{"value":"` + event.Output.Text + `"}`)}}
				}
				out <- event
			}
		}()
		return out
	})
	assistant := agent.New(agent.Definition{
		Name:        "retry-structured",
		Model:       model,
		RetryPolicy: &loop.RetryPolicy{MaxRetries: 1},
		Limits:      agent.Limits{MaxLoopIterations: 1},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
		Middleware: []agent.Middleware{transform},
	})
	workflow, err := assistant.NewRun(context.Background(), textRunInput("question"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	collectAgentEvents(workflow.RunEvents(context.Background()))
	result, err := workflow.Wait()
	if err != nil {
		t.Fatalf("Wait failed: %v", err)
	}
	if len(result.Output) != 1 || result.Output[0].Data == nil || string(result.Output[0].Data.JSON) != `{"value":"accepted"}` {
		t.Fatalf("retry output was not reduced by identity: %#v", result.Output)
	}
}

func TestPropagatedMiddlewareFailureSkipsFollowingDefaultStage(t *testing.T) {
	stageErr := errors.New("stage failed")
	failing := agent.New(agent.Definition{
		Name:   "failing",
		Model:  &mocks.MockModel{Responses: []mocks.MockModelResponse{{Err: stageErr}}},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return &testPromptBuilder{}, nil },
	})
	secondCalled := false
	second := agent.New(agent.Definition{
		Name:  "second",
		Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Text: "unexpected"}}}},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			secondCalled = true
			return &testPromptBuilder{}, nil
		},
	})
	main := workflowAgent("main", "answer",
		agent.NewAgentMiddleware(failing, agent.AgentMiddlewareConfig{ErrorPolicy: agent.PropagateError}),
		agent.NewAgentMiddleware(second, agent.AgentMiddlewareConfig{}),
	)
	workflow, err := main.NewRun(context.Background(), textRunInput("question"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	events := collectAgentEvents(workflow.RunEvents(context.Background()))
	result, err := workflow.Wait()
	if !errors.Is(err, stageErr) || secondCalled || len(result.Stages) != 1 {
		t.Fatalf("result=(%+v, %v) secondCalled=%v", result, err, secondCalled)
	}
	skipped := false
	for _, event := range events {
		if event.Type == agent.EventStageFinish && event.Source.Index == 2 && event.StageOutcome == agent.StageSkipped {
			skipped = true
		}
	}
	if !skipped {
		t.Fatalf("second stage was not observably skipped: %#v", events)
	}
}

func TestBlockingAndStreamingResultsAreEquivalent(t *testing.T) {
	newAgent := func() *agent.Agent {
		return workflowAgent("main", "answer", agent.NewAgentMiddleware(workflowAgent("post", " appended"), agent.AgentMiddlewareConfig{Output: agent.AppendOutput}))
	}
	input := textRunInput("question")
	input.ID = "equivalence"
	blocking, err := newAgent().NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("blocking NewRun failed: %v", err)
	}
	blockingResult, blockingErr := blocking.Run(context.Background())

	streaming, err := newAgent().NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("streaming NewRun failed: %v", err)
	}
	collectAgentEvents(streaming.RunEvents(context.Background()))
	streamingResult, streamingErr := streaming.Wait()
	if blockingErr != nil || streamingErr != nil {
		t.Fatalf("errors = (%v, %v)", blockingErr, streamingErr)
	}
	if !reflect.DeepEqual(blockingResult, streamingResult) {
		t.Fatalf("blocking and streaming differ:\nblocking=%+v\nstreaming=%+v", blockingResult, streamingResult)
	}
}

func productMarkerMiddleware() agent.Middleware {
	type key struct {
		source, iteration, attempt int
	}
	type pendingOutput struct {
		identity agent.Event
		events   []agent.Event
		text     string
	}
	return agent.MiddlewareFunc(func(ctx context.Context, run *agent.MiddlewareContext, upstream <-chan agent.Event) <-chan agent.Event {
		out := make(chan agent.Event)
		go func() {
			defer close(out)
			pending := map[key]*pendingOutput{}
			flush := func(item *pendingOutput) {
				if item == nil {
					return
				}
				const open = "```product\n"
				const closeMarker = "\n```"
				start := strings.Index(item.text, open)
				if start < 0 {
					for _, event := range item.events {
						out <- event
					}
					return
				}
				end := strings.Index(item.text[start+len(open):], closeMarker)
				if end < 0 {
					for _, event := range item.events {
						out <- event
					}
					return
				}
				end += start + len(open)
				emitText := func(text string) {
					if text == "" {
						return
					}
					event := item.identity
					event.Output = &agent.OutputPart{Kind: agent.OutputText, Text: text}
					out <- event
				}
				emitText(item.text[:start])
				data := item.identity
				data.Output = &agent.OutputPart{Kind: agent.OutputData, Data: &agent.StructuredOutput{Name: "product", JSON: json.RawMessage(`{"id":"` + item.text[start+len(open):end] + `"}`)}}
				out <- data
				emitText(item.text[end+len(closeMarker):])
			}
			for event := range upstream {
				k := key{event.Source.Index, event.IterationCount, event.AttemptID}
				switch event.Type {
				case agent.EventOutput:
					if event.Output != nil && event.Output.Kind == agent.OutputText {
						item := pending[k]
						if item == nil {
							item = &pendingOutput{identity: event}
							pending[k] = item
						}
						item.events = append(item.events, event)
						item.text += event.Output.Text
						continue
					}
				case agent.EventRetry, agent.EventDiscard:
					delete(pending, k)
				case agent.EventIterationDone:
					flush(pending[k])
					delete(pending, k)
				}
				out <- event
			}
			for _, item := range pending {
				flush(item)
			}
		}()
		return out
	})
}

func TestStructuredOutputMarkerAcrossTokenBoundariesAndMalformedFallback(t *testing.T) {
	for _, test := range []struct {
		name       string
		tokens     []ai.Token
		wantText   string
		wantDataID string
	}{
		{
			name:       "split marker",
			tokens:     []ai.Token{{Type: ai.TokenTypeText, Text: "Some ```pro"}, {Type: ai.TokenTypeText, Text: "duct\nproduct-123"}, {Type: ai.TokenTypeText, Text: "\n``` after"}},
			wantText:   "Some  after",
			wantDataID: `{"id":"product-123"}`,
		},
		{
			name:     "malformed fallback",
			tokens:   []ai.Token{{Type: ai.TokenTypeText, Text: "before ```product\n"}, {Type: ai.TokenTypeText, Text: "unfinished"}},
			wantText: "before ```product\nunfinished",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := &scriptedWorkflowModel{scripts: [][]ai.Token{test.tokens}}
			assistant := agent.New(agent.Definition{
				Name:  "structured-marker",
				Model: model,
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
					return &testPromptBuilder{}, nil
				},
				Middleware: []agent.Middleware{productMarkerMiddleware()},
			})
			workflow, err := assistant.NewRun(context.Background(), textRunInput("question"))
			if err != nil {
				t.Fatalf("NewRun failed: %v", err)
			}
			collectAgentEvents(workflow.RunEvents(context.Background()))
			result, err := workflow.Wait()
			if err != nil || result.Text != test.wantText {
				t.Fatalf("result=(%+v, %v)", result, err)
			}
			var gotData string
			for _, part := range result.Output {
				if part.Data != nil {
					gotData = string(part.Data.JSON)
				}
			}
			if gotData != test.wantDataID {
				t.Fatalf("structured JSON = %q, want %q", gotData, test.wantDataID)
			}
		})
	}
}
