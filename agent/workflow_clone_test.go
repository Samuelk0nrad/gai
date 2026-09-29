package agent

import (
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

func TestCloneWorkflowResultOwnsMutableExecutionData(t *testing.T) {
	call := &ai.ToolCall{Name: "lookup", Args: []byte(`{"query":"original"}`)}
	result := WorkflowResult{
		Input: RunInput{Prompt: gaictx.PromptInput{Context: []gaictx.Part{gaictx.NewTextPart("original")}}},
		Primary: AgentResult{
			Tokens: []ai.Token{{Type: ai.TokenTypeToolCall, Data: []byte("original"), ToolCall: call}},
			Messages: []gaictx.Message{{
				Role:       gaictx.RoleUser,
				Content:    gaictx.NewTextContent("original"),
				TokenCount: map[string]int{"counter": 1},
			}},
			Iterations: []loop.Iteration{{
				Parts: []loop.IterationPart{{
					Response: &ai.AIResponse{Text: "original"},
					ToolReq:  call,
					ToolResp: loop.NewToolSuccess("original"),
				}},
			}},
		},
	}

	cloned := cloneWorkflowResult(result)
	cloned.Input.Prompt.Context[0] = gaictx.NewTextPart("changed")
	cloned.Primary.Tokens[0].Data[0] = 'X'
	cloned.Primary.Tokens[0].ToolCall.Args[0] = 'X'
	cloned.Primary.Messages[0].TokenCount["counter"] = 99
	cloned.Primary.Iterations[0].Parts[0].Response.Text = "changed"
	cloned.Primary.Iterations[0].Parts[0].ToolReq.Args[0] = 'X'
	*cloned.Primary.Iterations[0].Parts[0].ToolResp.Text = "changed"
	originalPromptNode, err := result.Input.Prompt.Context[0].Render(t.Context())
	if err != nil || originalPromptNode.Value != "original" {
		t.Fatalf("prompt context slice was shared with clone: %+v", result.Input.Prompt.Context)
	}

	if string(result.Primary.Tokens[0].Data) != "original" || string(result.Primary.Tokens[0].ToolCall.Args) != `{"query":"original"}` {
		t.Fatalf("token data was shared with clone: %+v", result.Primary.Tokens[0])
	}
	if result.Primary.Messages[0].TokenCount["counter"] != 1 {
		t.Fatalf("message token counts were shared with clone: %+v", result.Primary.Messages[0].TokenCount)
	}
	part := result.Primary.Iterations[0].Parts[0]
	if part.Response.Text != "original" || string(part.ToolReq.Args) != `{"query":"original"}` || part.ToolResp.TextValue() != "original" {
		t.Fatalf("iteration data was shared with clone: %+v", part)
	}
}
