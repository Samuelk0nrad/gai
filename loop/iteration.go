package loop

import (
	"github.com/lace-ai/gai/ai"
)

// IterationType identifies an execution diagnostic record.
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
	// Parts retains execution diagnostics for tools and generation accounting.
	// It is not a conversation representation; use Messages for semantic output.
	Parts []IterationPart
	// Conversation is the canonical snapshot accumulated during this attempt.
	// Parts below remain execution records and are never used to rebuild it.
	Conversation  []ai.Message
	inputMessages int
	// Usage is the provider-reported usage for the accepted generation attempt.
	Usage ai.Usage
	// UsageReported distinguishes absent usage from a reported zero.
	UsageReported bool
	// RequestBudget is the finalized pre-generation accounting decision. Reported
	// Usage supersedes its input estimate when establishing the next checkpoint.
	RequestBudget *ai.RequestBudgetResult
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

// Messages returns snapshots of the input and canonical output of this iteration.
func (i Iteration) Messages() []ai.Message { return ai.CloneMessages(i.Conversation) }

// InputMessage returns the original input snapshot on the first accepted attempt.
func (i Iteration) InputMessage() *ai.Message {
	if i.inputMessages == 0 || len(i.Conversation) == 0 {
		return nil
	}
	m := i.Conversation[0].Clone()
	return &m
}

// DeltaMessages returns canonical output without the original user input.
func (i *Iteration) DeltaMessages() []ai.Message {
	if i == nil {
		return nil
	}
	return ai.CloneMessages(i.Conversation[i.inputMessages:])
}

// Clone snapshots both execution diagnostics and semantic conversation data.
func (i Iteration) Clone() Iteration {
	if i.RequestBudget != nil {
		budget := *i.RequestBudget
		i.RequestBudget = &budget
	}
	i.Conversation = ai.CloneMessages(i.Conversation)
	i.Parts = append([]IterationPart(nil), i.Parts...)
	for n := range i.Parts {
		p := &i.Parts[n]
		if p.Response != nil {
			r := *p.Response
			r.Message = r.Message.Clone()
			r.Raw = append([]byte(nil), r.Raw...)
			p.Response = &r
		}
		if p.ToolReq != nil {
			c := p.ToolReq.Clone()
			p.ToolReq = &c
		}
		if p.ToolResp != nil {
			r := *p.ToolResp
			if r.Text != nil {
				v := *r.Text
				r.Text = &v
			}
			if r.Err != nil {
				v := *r.Err
				r.Err = &v
			}
			p.ToolResp = &r
		}
	}
	return i
}

func (i *Iteration) appendConversationToken(t ai.Token) {
	// Accumulate before tools execute; assistant part order comes from the stream.
	if t.Part == nil {
		return
	}
	n := len(i.Conversation)
	if n == 0 || i.Conversation[n-1].Role != ai.RoleAssistant {
		i.Conversation = append(i.Conversation, ai.Message{Role: ai.RoleAssistant})
		n++
	}
	i.Conversation[n-1].AppendToken(t)
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
	i.appendConversationToken(t)

	var last *IterationPart
	if len(i.Parts) > 0 {
		last = &i.Parts[len(i.Parts)-1]
	}

	switch t.Type() {
	case ai.TokenTypeCompletion:
		if t.Completion == nil {
			return
		}
		if t.Completion.UsageReported {
			i.Usage = t.Completion.Usage
			i.UsageReported = true
		}
		if last != nil && last.Type == IterationTypeResponse {
			last.Response.AppendToken(t)
		} else {
			response := &ai.AIResponse{}
			response.AppendToken(t)
			i.Parts = append(i.Parts, IterationPart{Type: IterationTypeResponse, Response: response})
		}
	case ai.TokenTypeText, ai.TokenTypePart:
		if last != nil && last.Type == IterationTypeResponse {
			last.Response.AppendToken(t)
		} else {
			i.Parts = append(i.Parts, IterationPart{
				Type:     IterationTypeResponse,
				Response: responseFromToken(t),
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
				Response: responseFromToken(t),
			})
		}
	case ai.TokenTypeToolCall:
		i.Parts = append(i.Parts, IterationPart{
			Type:    IterationTypeToolCall,
			ToolReq: t.ToolCall(),
		})
	}
}

func responseFromToken(t ai.Token) *ai.AIResponse { r := &ai.AIResponse{}; r.AppendToken(t); return r }
