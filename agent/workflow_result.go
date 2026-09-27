package agent

import (
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

// AgentResult contains canonical accepted output. AttemptedTokens and
// AttemptedText retain the raw real-time stream, including discarded retries.
type AgentResult struct {
	Tokens          []ai.Token
	Text            string
	Reasoning       string
	AttemptedTokens []ai.Token
	AttemptedText   string
	Messages        []gaictx.Message
	Iterations      []loop.Iteration
	Usage           ai.Usage
	BilledUsage     ai.Usage
	Errors          []error
	Canceled        bool
	CancellationErr error
}

// StageResult is the named result produced by agent middleware.
type StageResult struct {
	Name   string
	Output OutputPolicy
	Result AgentResult
}

// WorkflowResult is a concurrency-safe snapshot of complete workflow state.
type WorkflowResult struct {
	Input RunInput

	// Output is the canonical visible workflow output after middleware.
	Output []OutputPart
	// Tokens is retained as a convenience view of text and reasoning output.
	Tokens    []ai.Token
	Text      string
	Reasoning string
	// AttemptedTokens retains the raw primary stream, including non-text tokens.
	AttemptedTokens []ai.Token
	// AttemptedText retains attempted user-visible text after middleware.
	AttemptedText string

	Primary AgentResult
	Stages  []StageResult

	Usage           ai.Usage
	BilledUsage     ai.Usage
	Errors          []error
	Canceled        bool
	CancellationErr error
	Complete        bool
}

func (w *Workflow) setVisibleOutputLocked(output []OutputPart) {
	w.result.Output = cloneOutputParts(output)
	w.result.Text = outputTextOnly(output)
	w.result.Reasoning = outputReasoningOnly(output)
	w.result.Tokens = outputPartsToTokens(output)
}

func outputPartsFromTokens(tokens []ai.Token) []OutputPart {
	var output []OutputPart
	for _, token := range tokens {
		text := token.Text
		if text == "" {
			text = string(token.Data)
		}
		switch token.Type {
		case ai.TokenTypeText:
			output = append(output, OutputPart{Kind: OutputText, Text: text})
		case ai.TokenTypeThought:
			output = append(output, OutputPart{Kind: OutputReasoning, Text: text})
		}
	}
	return output
}

func outputPartsToTokens(output []OutputPart) []ai.Token {
	var tokens []ai.Token
	for _, part := range output {
		switch part.Kind {
		case OutputText:
			tokens = append(tokens, ai.Token{Type: ai.TokenTypeText, Text: part.Text})
		case OutputReasoning:
			tokens = append(tokens, ai.Token{Type: ai.TokenTypeThought, Text: part.Text})
		}
	}
	return tokens
}

func outputTextOnly(output []OutputPart) string {
	var text string
	for _, part := range output {
		if part.Kind == OutputText {
			text += part.Text
		}
	}
	return text
}

func outputReasoningOnly(output []OutputPart) string {
	var reasoning string
	for _, part := range output {
		if part.Kind == OutputReasoning {
			reasoning += part.Text
		}
	}
	return reasoning
}

func tokenText(tokens []ai.Token) string {
	return outputTextOnly(outputPartsFromTokens(tokens))
}

func tokenReasoning(tokens []ai.Token) string {
	return outputReasoningOnly(outputPartsFromTokens(tokens))
}

func iterationTokens(iterations []loop.Iteration) []ai.Token {
	var tokens []ai.Token
	for _, iteration := range iterations {
		for _, part := range iteration.Parts {
			if part.Response != nil {
				if part.Response.Reasoning != "" {
					tokens = append(tokens, ai.Token{Type: ai.TokenTypeThought, Text: part.Response.Reasoning})
				}
				if part.Response.Text != "" {
					tokens = append(tokens, ai.Token{Type: ai.TokenTypeText, Text: part.Response.Text})
				}
			}
			if part.ToolReq != nil {
				tokens = append(tokens, ai.Token{Type: ai.TokenTypeToolCall, ToolCall: cloneToolCall(part.ToolReq)})
			}
		}
	}
	return tokens
}

func iterationUsage(iterations []loop.Iteration) ai.Usage {
	var usage ai.Usage
	for _, iteration := range iterations {
		usage.Add(iteration.Usage)
	}
	return usage
}

func (w *Workflow) addStage(stage StageResult) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.result.Stages = append(w.result.Stages, cloneStageResult(stage))
}
