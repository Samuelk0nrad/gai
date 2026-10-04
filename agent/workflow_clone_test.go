package agent

import (
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

// TestCloneWorkflowResultOwnsMutableExecutionData mutates a cloned workflow result to detect
// shared prompt slices, messages, call arguments, and tool results.
func TestCloneWorkflowResultOwnsMutableExecutionData(t *testing.T) {
	call := &ai.ToolCall{Name: "lookup", Args: []byte(`{"query":"original"}`)}
	result := WorkflowResult{
		Input: RunInput{Prompt: gaictx.PromptInput{Context: []gaictx.Part{gaictx.NewTextPart("original")}}},
		Primary: AgentResult{
			Tokens: []ai.Token{{Part: &ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: call}}},
			Messages: []ai.Message{{
				Role:  ai.RoleUser,
				Parts: ai.TextParts("original"),
			}},
			Iterations: []loop.Iteration{{
				Parts: []loop.IterationPart{{
					Response: &ai.AIResponse{Message: ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentText, Text: "original"}}}},
					ToolReq:  call,
					ToolResp: &loop.ToolResult{Text: "original"},
				}},
			}},
		},
	}

	cloned := cloneWorkflowResult(result)
	cloned.Input.Prompt.Context[0] = gaictx.NewTextPart("changed")
	cloned.Primary.Tokens[0].Part.ToolCall.Args[0] = 'X'
	cloned.Primary.Messages[0].Parts[0].Text = "changed"
	cloned.Primary.Iterations[0].Parts[0].Response.Message.Parts[0].Text = "changed"
	cloned.Primary.Iterations[0].Parts[0].ToolReq.Args[0] = 'X'
	cloned.Primary.Iterations[0].Parts[0].ToolResp.Text = "changed"
	originalPromptNode, err := result.Input.Prompt.Context[0].Render(t.Context())
	if err != nil || originalPromptNode.Value != "original" {
		t.Fatalf("prompt context slice was shared with clone: %+v", result.Input.Prompt.Context)
	}

	if string(result.Primary.Tokens[0].Part.ToolCall.Args) != `{"query":"original"}` {
		t.Fatalf("token data was shared with clone: %+v", result.Primary.Tokens[0])
	}
	if result.Primary.Messages[0].Text() != "original" {
		t.Fatalf("message token counts were shared with clone: %+v", result.Primary.Messages[0].Parts)
	}
	part := result.Primary.Iterations[0].Parts[0]
	if part.Response.Text() != "original" || string(part.ToolReq.Args) != `{"query":"original"}` || part.ToolResp.Text != "original" {
		t.Fatalf("iteration data was shared with clone: %+v", part)
	}
}
