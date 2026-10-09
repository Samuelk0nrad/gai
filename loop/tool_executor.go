package loop

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"time"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

type scheduledTool struct {
	execution ToolExecution
	pending   pendingToolCall
	tool      Tool
	options   ToolOptions
	result    *ToolResult
}

type toolCompletion struct {
	execution ToolExecution
	index     int
	result    *ToolResult
	duration  time.Duration
	err       error
}

// snapshotToolRegistrations captures trusted execution options once per run so
// classification and scheduling cannot observe different mutable provider values.
func snapshotToolRegistrations(tools []Tool) ([]Tool, error) {
	snapshots := make([]Tool, len(tools))
	registered := make(map[string]struct{}, len(tools))
	for i, tool := range tools {
		if nilImplementation(tool) {
			return nil, fmt.Errorf("%w: tool is nil", ai.ErrInvalidToolDefinition)
		}
		name := tool.Name()
		if _, exists := registered[name]; exists {
			return nil, fmt.Errorf("%w: duplicate tool %q", ai.ErrInvalidToolDefinition, name)
		}
		options, err := optionsForTool(tool)
		if err != nil {
			return nil, err
		}
		registered[name] = struct{}{}
		snapshots[i] = &configuredTool{Tool: tool, options: options.clone()}
	}
	return snapshots, nil
}

func validateToolCallBatch(calls []pendingToolCall) error {
	ids := make(map[string]struct{}, len(calls))
	for _, pending := range calls {
		id := pending.call.ID
		if _, exists := ids[id]; exists {
			return fmt.Errorf("%w: duplicate tool call ID %q", ai.ErrInvalidToolCall, id)
		}
		ids[id] = struct{}{}
	}
	return nil
}

// classifyTerminalBatch uses trusted registration metadata to classify the
// complete model-requested batch. Unknown calls do not make a known terminal
// batch ordinary; execution will retain their normal synthetic failures.
func classifyTerminalBatch(calls []pendingToolCall, tools []Tool) (terminal, mixed bool, err error) {
	registered := make(map[string]ToolOptions, len(tools))
	for _, tool := range tools {
		if nilImplementation(tool) {
			return false, false, fmt.Errorf("%w: tool is nil", ai.ErrInvalidToolDefinition)
		}
		name := tool.Name()
		if _, exists := registered[name]; exists {
			return false, false, fmt.Errorf("%w: duplicate tool %q", ai.ErrInvalidToolDefinition, name)
		}
		options, optionsErr := optionsForTool(tool)
		if optionsErr != nil {
			return false, false, optionsErr
		}
		registered[name] = options
	}
	var ordinary bool
	for _, pending := range calls {
		options, known := registered[pending.call.Name]
		if !known {
			continue
		}
		if options.Terminal {
			terminal = true
		} else {
			ordinary = true
		}
	}
	return terminal, terminal && ordinary, nil
}

func terminalBatchError(iteration Iteration, calls []pendingToolCall) error {
	for _, pending := range calls {
		if pending.partIndex < 0 || pending.partIndex >= len(iteration.Parts) {
			return fmt.Errorf("%w: tool %q has no execution result", ErrTerminalToolBatch, pending.call.Name)
		}
		part := iteration.Parts[pending.partIndex]
		if part.ToolExecution == nil || part.ToolExecution.State != ToolSucceeded ||
			part.ToolExecution.Output == ToolOutputRejected || part.ToolResp == nil || part.ToolResp.Err != nil {
			state := ToolNotStarted
			if part.ToolExecution != nil {
				state = part.ToolExecution.State
			}
			return fmt.Errorf("%w: tool %q did not succeed (state=%s)", ErrTerminalToolBatch, pending.call.Name, state)
		}
	}
	return nil
}

// reflect.Select accepts at most 65,536 cases; reserve two for completion/context.
const maxToolGuardWaits = 65534

// validateToolGuardCount rejects batches beyond the runtime select-case limit before dispatch.
func validateToolGuardCount(tasks []scheduledTool) error {
	guards := make(map[*ToolGuard]struct{})
	for _, task := range tasks {
		if task.result == nil && task.options.Guard != nil {
			guards[task.options.Guard] = struct{}{}
		}
	}
	if len(guards) > maxToolGuardWaits {
		return fmt.Errorf("%w: batch exceeds %d distinct tool guards", ErrToolExecutionConfig, maxToolGuardWaits)
	}
	return nil
}

// prepareToolCalls copies requests and snapshots registration settings.
func prepareToolCalls(calls []pendingToolCall, tools []Tool) ([]scheduledTool, error) {
	registry := make(map[string]Tool, len(tools))
	options := make(map[string]ToolOptions, len(tools))
	for _, tool := range tools {
		if nilImplementation(tool) {
			return nil, fmt.Errorf("%w: tool is nil", ai.ErrInvalidToolDefinition)
		}
		name := tool.Name()
		if _, exists := registry[name]; exists {
			return nil, fmt.Errorf("%w: duplicate tool %q", ai.ErrInvalidToolDefinition, name)
		}
		opts, err := optionsForTool(tool)
		if err != nil {
			return nil, err
		}
		registry[name], options[name] = tool, opts
	}
	tasks := make([]scheduledTool, len(calls))
	for i, pending := range calls {
		pending.call = pending.call.Clone()
		task := scheduledTool{execution: ToolExecution{State: ToolNotStarted}, pending: pending, tool: registry[pending.call.Name], options: options[pending.call.Name]}
		err := pending.call.Validate()
		if err == nil && !json.Valid(pending.call.Args) {
			err = ErrToolCallMalformed
		}
		if err == nil && task.tool == nil {
			err = fmt.Errorf("%w: %s", ErrToolNotFound, pending.call.Name)
		}
		if err != nil {
			task.result = &ToolResult{Err: err}
		}
		tasks[i] = task
	}
	return tasks, nil
}

// executeToolCalls owns all iteration writes. Only admitted work gets a worker;
// queued calls and shared-guard waiters never occupy goroutines or worker slots.
func (l *Loop) executeToolCalls(ctx context.Context, iteration *Iteration, calls []pendingToolCall, tools []Tool, events chan<- Event, iterationCount, attemptID, retryCount int) error {
	if err := l.ToolExecution.Validate(); err != nil {
		return err
	}
	tasks, err := prepareToolCalls(calls, tools)
	if err != nil {
		return err
	}
	preflightErr := l.preflightToolPolicies(ctx, tasks, events, iterationCount, attemptID, retryCount)
	if preflightErr == nil {
		preflightErr = l.preflightToolApprovals(ctx, tasks, events, iterationCount, attemptID, retryCount)
	}
	for _, task := range tasks {
		iteration.Parts[task.pending.partIndex].ToolExecution = cloneToolExecution(&task.execution)
	}
	if preflightErr != nil {
		return preflightErr
	}
	if err := validateToolGuardCount(tasks); err != nil {
		return err
	}
	workerCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	limit := l.ToolExecution.MaxConcurrent
	if limit == 0 || limit > len(tasks) {
		limit = len(tasks)
	}
	completions := make(chan toolCompletion, limit)
	pending := make([]int, len(tasks))
	for i := range pending {
		pending[i] = i
	}
	active := 0
	busy := map[string]bool{}
	var firstErr error
	fail := func(err error) {
		if firstErr == nil {
			firstErr = err
			cancel()
		}
	}
	publish := func(done toolCompletion) {
		task := tasks[done.index]
		iteration.Parts[task.pending.partIndex].ToolResp = done.result
		iteration.Parts[task.pending.partIndex].ToolExecution = cloneToolExecution(&done.execution)
		if task.result != nil {
			outcome, _, safeErr := toolResult(done.result)
			if done.err != nil {
				outcome, safeErr = toolOutcomeProcessing, errObservedToolProcessing
				if done.err == ErrToolPanic {
					outcome, safeErr = toolOutcomePanic, errObservedToolPanic
				}
			}
			status := "success"
			if safeErr != nil {
				status = "error"
			}
			fields := toolObservationFields(task.pending.call, &done.execution, outcome, status, 0)
			fields["invoked"] = false
			gai.EmitObservation(ctx, l.ObservationSink, gai.Observation{
				Name: "loop_tool_finished", Source: "loop:Tool", Err: safeErr, Fields: fields,
			})
		}
		if events != nil && firstErr == nil {
			var event Event
			if done.err != nil {
				event = ToolErrorEvent(iterationCount, attemptID, retryCount, task.pending.call, nil, done.duration, done.err)
			} else {
				event = ToolResultEvent(iterationCount, attemptID, retryCount, task.pending.call, done.result, done.duration)
			}
			event.ToolExecution = cloneToolExecution(&done.execution)
			if err := sendEvent(ctx, events, event); err != nil {
				fail(err)
			}
		}
		if done.err != nil {
			fail(done.err)
		}
	}
	for len(pending) > 0 || active > 0 {
		if err := ctx.Err(); err != nil {
			fail(err)
		}
		var guards []<-chan struct{}
		waitingGuards := map[*ToolGuard]bool{}
		blockedNames := map[string]bool{}
		if firstErr == nil {
			for position := 0; position < len(pending) && active < limit; {
				index := pending[position]
				task := tasks[index]
				name := task.pending.call.Name
				if task.options.Serial && (busy[name] || blockedNames[name]) {
					position++
					continue
				}
				// Synthetic refusals do not use the handler's shared resource.
				// Keep local Serial/capacity rules for their result processors.
				guard := task.options.Guard
				if task.result != nil {
					guard = nil
				}
				release, changed := guard.tryAcquire()
				if release == nil {
					if !waitingGuards[guard] {
						guards = append(guards, changed)
						waitingGuards[guard] = true
					}
					if task.options.Serial {
						blockedNames[name] = true
					}
					position++
					continue
				}
				if err := ctx.Err(); err != nil {
					release()
					fail(err)
					break
				}
				if events != nil && task.result == nil {
					if err := sendEvent(ctx, events, ToolStartEvent(iterationCount, attemptID, retryCount, task.pending.call)); err != nil {
						release()
						fail(err)
						break
					}
				}
				if err := ctx.Err(); err != nil {
					release()
					fail(err)
					break
				}
				pending = append(pending[:position], pending[position+1:]...)
				active++
				if task.options.Serial {
					busy[name] = true
				}
				go func(index int, task scheduledTool, release func()) {
					done := toolCompletion{index: index, execution: task.execution}
					defer func() {
						if recover() != nil {
							done.result = nil
							done.err = ErrToolPanic
							done.execution.Output = ToolOutputRejected
							if done.execution.State == ToolRunning {
								done.execution.State = ToolFailed
							}
						}
						release()
						completions <- done
					}()
					if task.result != nil {
						done.result, done.err = l.processUninvoked(workerCtx, task)
						done.execution.Output = outputState(*task.result, done.result)
						return
					}
					timeout := l.ToolExecution.DefaultTimeout
					if task.options.Timeout != nil {
						timeout = *task.options.Timeout
					}
					done.result, done.duration, done.err = processObservedToolExecution(workerCtx, ToolPolicyInput{Call: task.pending.call, Traits: task.options.Traits}, []Tool{task.tool}, l.ToolResultProcessor, timeout, &done.execution, l.ObservationSink)
				}(index, task, release)
			}
		}
		if firstErr != nil {
			pending = nil
		}
		if active == 0 && len(pending) == 0 {
			break
		}
		cases := []reflect.SelectCase{{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(completions)}}
		if firstErr == nil {
			cases = append(cases, reflect.SelectCase{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(ctx.Done())})
			for _, guard := range guards {
				cases = append(cases, reflect.SelectCase{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(guard)})
			}
		}
		selected, value, _ := reflect.Select(cases)
		if selected != 0 {
			continue
		}
		done := value.Interface().(toolCompletion)
		active--
		if tasks[done.index].options.Serial {
			delete(busy, tasks[done.index].pending.call.Name)
		}
		publish(done)
	}
	if firstErr != nil {
		return firstErr
	}
	for _, task := range tasks {
		result := iteration.Parts[task.pending.partIndex].ToolResp
		if result == nil {
			continue
		}
		output := ai.ToolResult{ToolCallID: task.pending.call.ID, Name: task.pending.call.Name, Parts: ai.TextParts(result.String()), IsError: result.Err != nil}
		iteration.Conversation = append(iteration.Conversation, ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &output}}})
	}
	return nil
}

// processUninvoked filters synthetic errors without invoking a handler.
func (l *Loop) processUninvoked(ctx context.Context, task scheduledTool) (response *ToolResult, processErr error) {
	defer func() {
		if recover() != nil {
			response = nil
			processErr = ErrToolPanic
		}
	}()
	result := *task.result
	if l.ToolResultProcessor != nil {
		processed, err := l.ToolResultProcessor.Process(ctx, ToolPolicyInput{Call: task.pending.call.Clone(), Traits: task.options.Traits}, result)
		if err != nil {
			return nil, fmt.Errorf("%w: %w", ErrToolResultProcess, err)
		}
		result = normalizeToolResult(processed.Text, processed.Err)
	}
	result = preserveToolRefusal(*task.result, result)
	return &result, nil
}

func (l *Loop) preflightToolPolicies(ctx context.Context, tasks []scheduledTool, events chan<- Event, iteration, attempt, retry int) error {
	for i := range tasks {
		task := &tasks[i]
		if err := ctx.Err(); err != nil {
			return err
		}
		if task.result != nil {
			continue
		}
		decision := ToolDecision{Action: ToolAllow}
		if l.ToolPolicy != nil {
			var err error
			decision, err = evaluateToolPolicy(ctx, l.ToolPolicy, ToolPolicyInput{Call: task.pending.call, Traits: task.options.Traits})
			if err != nil {
				return err
			}
		}
		task.execution.Decision = decision
		if l.ToolPolicy != nil && events != nil {
			event := Event{Type: EventToolDecision, IterationCount: iteration, AttemptID: attempt, RetryCount: retry, ToolCall: &task.pending.call, ToolExecution: cloneToolExecution(&task.execution)}
			if err := sendEvent(ctx, events, event); err != nil {
				return err
			}
		}
		switch decision.Action {
		case ToolDeny:
			task.result = &ToolResult{Err: decisionError(decision, ErrToolDenied)}
		case ToolRequireApproval:
			task.result = &ToolResult{Err: decisionError(decision, ErrToolApprovalRequired)}
		}
	}
	return nil
}
func decisionError(decision ToolDecision, kind error) error {
	if decision.Reason == "" {
		return kind
	}
	return safeToolError{text: decision.Reason, kind: kind}
}
