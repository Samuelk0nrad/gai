package agent

import (
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
)

func cloneRunInput(input RunInput) RunInput {
	cloned := input
	if input.TraceContext != nil {
		traceContext := *input.TraceContext
		traceContext.Tags = append([]string(nil), input.TraceContext.Tags...)
		if input.TraceContext.Metadata != nil {
			traceContext.Metadata = make(map[string]string, len(input.TraceContext.Metadata))
			for key, value := range input.TraceContext.Metadata {
				traceContext.Metadata[key] = value
			}
		}
		cloned.TraceContext = &traceContext
	}
	cloned.Prompt = input.Prompt.Clone()
	cloned.Execution = cloneExecution(input.Execution)
	if input.Meta != nil {
		cloned.Meta = make(map[string]any, len(input.Meta))
		for key, value := range input.Meta {
			cloned.Meta[key] = value
		}
	}
	return cloned
}

func cloneResponseFormat(format ai.ResponseFormat) ai.ResponseFormat {
	cloned := format
	cloned.Schema = append([]byte(nil), format.Schema...)
	return cloned
}

func cloneTokens(tokens []ai.Token) []ai.Token {
	if tokens == nil {
		return nil
	}
	out := make([]ai.Token, len(tokens))
	for i := range tokens {
		out[i] = tokens[i].Clone()
	}
	return out
}

func cloneToolCall(call *ai.ToolCall) *ai.ToolCall {
	if call == nil {
		return nil
	}
	cloned := call.Clone()
	return &cloned
}

func cloneMessages(messages []ai.Message) []ai.Message { return ai.CloneMessages(messages) }

func cloneIterationPtr(iteration *loop.Iteration) *loop.Iteration {
	if iteration == nil {
		return nil
	}
	cloned := cloneIterations([]loop.Iteration{*iteration})[0]
	return &cloned
}

func cloneIterations(iterations []loop.Iteration) []loop.Iteration {
	if iterations == nil {
		return nil
	}
	out := make([]loop.Iteration, len(iterations))
	for i := range iterations {
		out[i] = iterations[i].Clone()
	}
	return out
}

func cloneToolResult(response *loop.ToolResult) *loop.ToolResult {
	if response == nil {
		return nil
	}
	cloned := *response
	return &cloned
}

func cloneAgentResult(result AgentResult) AgentResult {
	result.Tokens = cloneTokens(result.Tokens)
	result.AttemptedTokens = cloneTokens(result.AttemptedTokens)
	result.Messages = cloneMessages(result.Messages)
	result.Iterations = cloneIterations(result.Iterations)
	result.Errors = append([]error(nil), result.Errors...)
	return result
}

func cloneStageResult(stage StageResult) StageResult {
	stage.Result = cloneAgentResult(stage.Result)
	return stage
}

func cloneWorkflowResult(result WorkflowResult) WorkflowResult {
	result.Input = cloneRunInput(result.Input)
	result.Output = cloneOutputParts(result.Output)
	result.Primary = cloneAgentResult(result.Primary)
	result.Tokens = cloneTokens(result.Tokens)
	result.AttemptedTokens = cloneTokens(result.AttemptedTokens)
	result.Errors = append([]error(nil), result.Errors...)
	result.Stages = append([]StageResult(nil), result.Stages...)
	for i := range result.Stages {
		result.Stages[i] = cloneStageResult(result.Stages[i])
	}
	return result
}
