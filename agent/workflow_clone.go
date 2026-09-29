package agent

import (
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
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
	cloned := make([]ai.Token, len(tokens))
	for i, token := range tokens {
		cloned[i] = token
		cloned[i].Data = append([]byte(nil), token.Data...)
		cloned[i].ToolCall = cloneToolCall(token.ToolCall)
		if token.Completion != nil {
			completion := *token.Completion
			completion.Raw = append([]byte(nil), token.Completion.Raw...)
			cloned[i].Completion = &completion
		}
	}
	return cloned
}

func cloneToolCall(call *ai.ToolCall) *ai.ToolCall {
	if call == nil {
		return nil
	}
	cloned := *call
	cloned.Args = append([]byte(nil), call.Args...)
	cloned.ThoughtSignature = append([]byte(nil), call.ThoughtSignature...)
	return &cloned
}

func cloneMessages(messages []gaictx.Message) []gaictx.Message {
	if messages == nil {
		return nil
	}
	cloned := make([]gaictx.Message, len(messages))
	for i, message := range messages {
		cloned[i] = message
		if message.TokenCount != nil {
			cloned[i].TokenCount = make(map[string]int, len(message.TokenCount))
			for counter, count := range message.TokenCount {
				cloned[i].TokenCount[counter] = count
			}
		}
	}
	return cloned
}

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
	cloned := make([]loop.Iteration, len(iterations))
	for i, iteration := range iterations {
		cloned[i] = iteration
		if iteration.UserMessage != nil {
			message := cloneMessages([]gaictx.Message{*iteration.UserMessage})[0]
			cloned[i].UserMessage = &message
		}
		cloned[i].Parts = make([]loop.IterationPart, len(iteration.Parts))
		for j, part := range iteration.Parts {
			cloned[i].Parts[j] = part
			if part.Response != nil {
				response := *part.Response
				response.Raw = append([]byte(nil), part.Response.Raw...)
				response.ToolCalls = make([]ai.ToolCall, len(part.Response.ToolCalls))
				for k := range part.Response.ToolCalls {
					response.ToolCalls[k] = *cloneToolCall(&part.Response.ToolCalls[k])
				}
				cloned[i].Parts[j].Response = &response
			}
			cloned[i].Parts[j].ToolReq = cloneToolCall(part.ToolReq)
			cloned[i].Parts[j].ToolResp = cloneToolResponse(part.ToolResp)
		}
	}
	return cloned
}

func cloneToolResponse(response *loop.ToolResponse) *loop.ToolResponse {
	if response == nil {
		return nil
	}
	cloned := *response
	if response.Text != nil {
		text := *response.Text
		cloned.Text = &text
	}
	if response.Err != nil {
		err := *response.Err
		cloned.Err = &err
	}
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
