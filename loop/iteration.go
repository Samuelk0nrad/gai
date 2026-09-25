package loop

import (
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

// IterationType identifies the semantic kind of one iteration part.
type IterationType string

const (
	// IterationTypeToolCall identifies a model request to invoke a tool.
	IterationTypeToolCall IterationType = "tool_call"
	// IterationTypeResponse identifies generated assistant text.
	IterationTypeResponse IterationType = "response"
	// IterationTypeToolError identifies a failed tool operation.
	IterationTypeToolError IterationType = "tool_error"
)

// Iteration records one model generation and its tool interactions.
type Iteration struct {
	// Count is the one-based iteration number.
	Count int
	// Parts contains generated responses, tool calls, and tool errors in order.
	Parts []IterationPart
	// UserMessage is the structured user input retained by the first iteration.
	UserMessage *gaictx.Message
	// Usage is the provider-reported usage for the accepted generation attempt.
	Usage ai.Usage
}

// IterationPart contains one response, tool call, or tool result segment.
type IterationPart struct {
	// Type identifies which fields are meaningful.
	Type IterationType
	// Response contains generated text for IterationTypeResponse.
	Response *ai.AIResponse
	// ToolReq contains the requested call for tool-related parts.
	ToolReq *ai.ToolCall
	// ToolResp contains the result produced for ToolReq.
	ToolResp *ToolResponse
}

// Messages converts the iteration into ordered conversation messages.
// UserMessage is included when present on an accepted iteration.
func (i Iteration) Messages() []gaictx.Message {
	var msgs []gaictx.Message

	if i.UserMessage != nil {
		msgs = append(msgs, *i.UserMessage)
	}

	return append(msgs, i.partMessages()...)
}

// DeltaMessages converts only the iteration parts into conversation messages,
// excluding the original user request.
func (i *Iteration) DeltaMessages() []gaictx.Message {
	if i == nil {
		return nil
	}
	return i.partMessages()
}

func (i *Iteration) partMessages() []gaictx.Message {
	var msgs []gaictx.Message
	for _, part := range i.Parts {
		switch part.Type {
		case IterationTypeToolCall, IterationTypeToolError:
			if part.ToolReq != nil {
				msgs = append(msgs, gaictx.Message{
					Role:    gaictx.RoleAssistant,
					Content: gaictx.NewToolCallContent(part.ToolReq.Name, string(part.ToolReq.Args)),
				})
				if part.ToolResp != nil {
					if err := part.ToolResp.ErrorValue(); err != nil {
						msgs = append(msgs, gaictx.Message{
							Role:    gaictx.RoleTool,
							Content: gaictx.NewToolResultErrContent(part.ToolReq.Name, err.Error()),
						})
					} else {
						msgs = append(msgs, gaictx.Message{
							Role:    gaictx.RoleTool,
							Content: gaictx.NewToolResultContent(part.ToolReq.Name, part.ToolResp.TextValue(), false, ""),
						})
					}
				}
			}
		case IterationTypeResponse:
			if part.Response != nil {
				if part.Response.Text == "" {
					continue
				}
				msgs = append(msgs, gaictx.Message{
					Role:    gaictx.RoleAssistant,
					Content: gaictx.NewTextContent(part.Response.Text),
				})
			}
		}
	}
	return msgs
}

// CurrentPart returns the most recently appended part, or nil when empty.
func (i *Iteration) CurrentPart() *IterationPart {
	if len(i.Parts) == 0 {
		return nil
	}
	return &i.Parts[len(i.Parts)-1]
}

// AppendToken adds a streamed model token to the appropriate iteration part.
func (i *Iteration) AppendToken(t ai.Token) {
	text := t.Text
	if text == "" && len(t.Data) > 0 {
		text = string(t.Data)
	}

	var last *IterationPart
	if len(i.Parts) > 0 {
		last = &i.Parts[len(i.Parts)-1]
	}

	switch t.Type {
	case ai.TokenTypeCompletion:
		if t.Completion == nil {
			return
		}
		i.Usage = t.Completion.Usage
		if last != nil && last.Type == IterationTypeResponse {
			last.Response.AppendToken(t)
		} else {
			response := &ai.AIResponse{}
			response.AppendToken(t)
			i.Parts = append(i.Parts, IterationPart{Type: IterationTypeResponse, Response: response})
		}
	case ai.TokenTypeText:
		if last != nil && last.Type == IterationTypeResponse {
			last.Response.AppendToken(t)
		} else {
			i.Parts = append(i.Parts, IterationPart{
				Type:     IterationTypeResponse,
				Response: &ai.AIResponse{Text: text, OutputTokens: t.TokenUsage},
			})
		}
	case ai.TokenTypeErr:
		i.Parts = append(i.Parts, IterationPart{
			Type:     IterationTypeToolError,
			ToolResp: NewToolError(t.Err),
		})
	case ai.TokenTypeThought:
		if last != nil && last.Type == IterationTypeResponse {
			last.Response.AppendToken(t)
		} else {
			i.Parts = append(i.Parts, IterationPart{
				Type:     IterationTypeResponse,
				Response: &ai.AIResponse{Reasoning: text, OutputTokens: t.TokenUsage, ReasoningTokens: t.TokenUsage},
			})
		}
	case ai.TokenTypeToolCall:
		i.Parts = append(i.Parts, IterationPart{
			Type:    IterationTypeToolCall,
			ToolReq: t.ToolCall,
		})
	}
}
