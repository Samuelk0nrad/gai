package agent

import (
	"context"
	"errors"
	"sort"

	"github.com/lace-ai/gai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/internal/observe"
	"go.opentelemetry.io/otel/attribute"
)

const agentTracerName = "github.com/lace-ai/gai/agent"

type agentRunObserver struct {
	operation *observe.Operation
}

func newAgentRunObserver(ctx context.Context, workflow *Workflow) (context.Context, *agentRunObserver) {
	name := ""
	var input RunInput
	if workflow != nil {
		name = workflow.name
		input = workflow.result.Input
	}
	attrs := []attribute.KeyValue{
		attribute.String("agent.name", name),
		attribute.Int("agent.meta_key_count", len(input.Meta)),
	}
	if input.ID != "" {
		attrs = append(attrs, attribute.String("agent.run_id", input.ID))
	}
	ctx, operation := observe.Start(ctx, nil, agentTracerName, "agent", "agent.operation", "run", "agent:Workflow.Run", attrs...)
	return ctx, &agentRunObserver{operation: operation}
}

func (o *agentRunObserver) Finished(result WorkflowResult) {
	if o == nil {
		return
	}
	o.operation.Set(
		attribute.Int("agent.token_count", len(result.Tokens)),
		attribute.Int("agent.error_count", len(result.Errors)),
		attribute.Bool("agent.complete", result.Complete),
	)
	o.operation.Finish(errors.Join(result.Errors...))
}

type runCreationObserver struct {
	debug           gai.ObservationSink
	operation       *observe.Operation
	agentName       string
	modelName       string
	toolCount       int
	middlewareCount int
	input           RunInput
}

func newRunCreationObserver(ctx context.Context, agent *Agent, input RunInput) (context.Context, *runCreationObserver) {
	name := ""
	var debug gai.ObservationSink
	modelName := ""
	toolCount := 0
	middlewareCount := 0
	if agent != nil {
		name = agent.name()
		debug = agent.debugSink()
		middlewareCount = len(agent.middleware())
		toolCount = len(agent.def.Tools)
		if agent.def.Model != nil {
			modelName = agent.def.Model.Name()
		}
	}
	ctx, operation := observe.Start(ctx, debug, agentTracerName, "agent.run", "agent.operation", "create", "agent:Agent.NewRun",
		attribute.String("agent.name", name),
		attribute.String("agent.model", modelName),
		attribute.Int("agent.tool_count", toolCount),
		attribute.Int("agent.middleware_count", middlewareCount),
		attribute.Int("agent.user_input_chars", promptUserChars(input.Prompt)),
		attribute.Int("agent.input_context_parts", len(input.Prompt.Context)),
		attribute.Int("agent.max_tokens", input.MaxTokens),
	)
	return ctx, &runCreationObserver{
		debug:           debug,
		operation:       operation,
		agentName:       name,
		modelName:       modelName,
		toolCount:       toolCount,
		middlewareCount: middlewareCount,
		input:           input,
	}
}

func (o *runCreationObserver) Created(ctx context.Context) {
	o.emit(ctx, "agent_run_created", o.fields(ctx), nil)
}

func (o *runCreationObserver) Failed(ctx context.Context, stage string, err error) {
	fields := o.fields(ctx)
	fields["stage"] = stage
	o.emit(ctx, "agent_run_creation_failed", fields, err)
}

func (o *runCreationObserver) Finish(err error) {
	if o == nil {
		return
	}
	o.operation.Finish(err)
}

func (o *runCreationObserver) fields(ctx context.Context) map[string]any {
	fields := map[string]any{}
	if o == nil {
		return fields
	}
	fields["agent_name"] = o.agentName
	fields["model"] = o.modelName
	fields["tool_count"] = o.toolCount
	fields["middleware_count"] = o.middlewareCount
	fields["user_input_chars"] = promptUserChars(o.input.Prompt)
	fields["input_context_parts"] = len(o.input.Prompt.Context)
	fields["max_tokens"] = o.input.MaxTokens
	fields["meta_keys"] = sortedMetaKeys(o.input.Meta)
	if o.input.ID != "" {
		fields["run_id"] = o.input.ID
	}
	if o.input.Prompt.User != nil {
		gai.AddObservationContent(ctx, o.debug, fields, "user_input", gai.ContentKindPrompt, o.input.Prompt.User.String())
	}
	return fields
}

func promptUserChars(input gaictx.PromptInput) int {
	if input.User == nil {
		return 0
	}
	return len(input.User.String())
}

func (o *runCreationObserver) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if o == nil {
		return
	}
	o.operation.Emit(ctx, name, fields, err)
}

type workflowObserver struct {
	debug           gai.ObservationSink
	operation       *observe.Operation
	agentName       string
	middlewareCount int
}

func newWorkflowObserver(ctx context.Context, workflow *Workflow) (context.Context, *workflowObserver) {
	name := ""
	middlewareCount := 0
	var debug gai.ObservationSink
	if workflow != nil {
		name = workflow.name
		middlewareCount = len(workflow.middleware)
		debug = workflow.debug
	}
	ctx, operation := observe.Start(ctx, debug, agentTracerName, "agent.workflow", "agent.operation", "run", "agent:Workflow.Run",
		attribute.String("agent.name", name),
		attribute.Int("agent.middleware_count", middlewareCount),
	)
	return ctx, &workflowObserver{debug: debug, operation: operation, agentName: name, middlewareCount: middlewareCount}
}

func (o *workflowObserver) Started(ctx context.Context) {
	o.emit(ctx, "agent_workflow_started", map[string]any{
		"agent_name":       o.agentName,
		"middleware_count": o.middlewareCount,
	}, nil)
}

func (o *workflowObserver) PrimaryFinished(ctx context.Context, result AgentResult) {
	fields := agentResultFields(result)
	fields["agent_name"] = o.agentName
	gai.AddObservationContent(ctx, o.debug, fields, "output_text", gai.ContentKindCompletion, result.Text)
	gai.AddObservationContent(ctx, o.debug, fields, "reasoning", gai.ContentKindReasoning, result.Reasoning)
	o.emit(ctx, "agent_primary_finished", fields, errors.Join(result.Errors...))
}

func (o *workflowObserver) Finished(ctx context.Context, result WorkflowResult) {
	if o == nil {
		return
	}
	fields := map[string]any{
		"agent_name":       o.agentName,
		"middleware_count": o.middlewareCount,
		"stage_count":      len(result.Stages),
		"token_count":      len(result.Tokens),
		"text_chars":       len(result.Text),
		"error_count":      len(result.Errors),
		"complete":         result.Complete,
	}
	o.operation.Set(
		attribute.Int("agent.stage_count", len(result.Stages)),
		attribute.Int("agent.token_count", len(result.Tokens)),
		attribute.Int("agent.text_chars", len(result.Text)),
		attribute.Int("agent.error_count", len(result.Errors)),
	)
	gai.AddObservationContent(ctx, o.debug, fields, "output_text", gai.ContentKindCompletion, result.Text)
	gai.AddObservationContent(ctx, o.debug, fields, "reasoning", gai.ContentKindReasoning, result.Reasoning)
	err := errors.Join(result.Errors...)
	o.emit(ctx, "agent_workflow_finished", fields, err)
	o.operation.Finish(err)
}

func (o *workflowObserver) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if o == nil {
		return
	}
	o.operation.Emit(ctx, name, fields, err)
}

type middlewareObserver struct {
	debug       gai.ObservationSink
	operation   *observe.Operation
	agentName   string
	stageName   string
	output      OutputPolicy
	errorPolicy ErrorPolicy
}

func newMiddlewareObserver(ctx context.Context, run *MiddlewareContext, middleware *AgentMiddleware, upstream WorkflowResult) (context.Context, *middlewareObserver) {
	agentName := ""
	var debug gai.ObservationSink
	if run != nil && run.workflow != nil {
		agentName = run.workflow.name
		debug = run.workflow.debug
	}
	stageName := middleware.name()
	ctx, operation := observe.Start(ctx, debug, agentTracerName, "agent.middleware", "agent.operation", "run", "agent:AgentMiddleware.Process",
		attribute.String("agent.name", agentName),
		attribute.String("agent.middleware.name", stageName),
		attribute.String("agent.output_policy", outputPolicyName(middleware.config.Output)),
		attribute.String("agent.error_policy", errorPolicyName(middleware.config.ErrorPolicy)),
		attribute.Int("agent.upstream_token_count", len(upstream.Tokens)),
		attribute.Int("agent.upstream_error_count", len(upstream.Errors)),
	)
	return ctx, &middlewareObserver{
		debug:       debug,
		operation:   operation,
		agentName:   agentName,
		stageName:   stageName,
		output:      middleware.config.Output,
		errorPolicy: middleware.config.ErrorPolicy,
	}
}

func (o *middlewareObserver) Started(ctx context.Context) {
	o.emit(ctx, "agent_middleware_started", o.fields(), nil)
}

func (o *middlewareObserver) Skipped(ctx context.Context, reason string) {
	fields := o.fields()
	fields["reason"] = reason
	o.emit(ctx, "agent_middleware_skipped", fields, nil)
	if o != nil {
		o.operation.Set(attribute.Bool("agent.middleware.skipped", true), attribute.String("agent.middleware.skip_reason", reason))
		o.operation.Finish(nil)
	}
}

func (o *middlewareObserver) Finished(ctx context.Context, result AgentResult, applied bool) {
	if o == nil {
		return
	}
	fields := o.fields()
	for key, value := range agentResultFields(result) {
		fields[key] = value
	}
	fields["output_applied"] = applied
	o.operation.Set(
		attribute.Int("agent.middleware.token_count", len(result.Tokens)),
		attribute.Int("agent.middleware.error_count", len(result.Errors)),
		attribute.Bool("agent.middleware.output_applied", applied),
	)
	gai.AddObservationContent(ctx, o.debug, fields, "output_text", gai.ContentKindCompletion, result.Text)
	gai.AddObservationContent(ctx, o.debug, fields, "reasoning", gai.ContentKindReasoning, result.Reasoning)
	err := errors.Join(result.Errors...)
	name := "agent_middleware_finished"
	if err != nil {
		name = "agent_middleware_failed"
	}
	o.emit(ctx, name, fields, err)
	o.operation.Finish(err)
}

func (o *middlewareObserver) fields() map[string]any {
	if o == nil {
		return map[string]any{}
	}
	return map[string]any{
		"agent_name":    o.agentName,
		"middleware":    o.stageName,
		"output_policy": outputPolicyName(o.output),
		"error_policy":  errorPolicyName(o.errorPolicy),
	}
}

func (o *middlewareObserver) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if o == nil {
		return
	}
	o.operation.Emit(ctx, name, fields, err)
}

func agentResultFields(result AgentResult) map[string]any {
	return map[string]any{
		"token_count":     len(result.Tokens),
		"text_chars":      len(result.Text),
		"reasoning_chars": len(result.Reasoning),
		"message_count":   len(result.Messages),
		"iteration_count": len(result.Iterations),
		"error_count":     len(result.Errors),
	}
}

func sortedMetaKeys(meta map[string]any) []string {
	keys := make([]string, 0, len(meta))
	for key := range meta {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func outputPolicyName(policy OutputPolicy) string {
	switch policy {
	case AppendOutput:
		return "append"
	case ReplaceOutput:
		return "replace"
	default:
		return "preserve"
	}
}

func errorPolicyName(policy ErrorPolicy) string {
	if policy == RecordError {
		return "record"
	}
	return "propagate"
}
