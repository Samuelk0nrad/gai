package loop

import (
	"context"
	"strings"

	"github.com/lace-ai/gai/ai"
)

// EchoTool is a small Tool implementation that returns its text argument.
type EchoTool struct{}

// NewEchoTool creates an EchoTool.
func NewEchoTool() *EchoTool {
	return &EchoTool{}
}

// Name returns the tool name exposed to models.
func (t *EchoTool) Name() string {
	return "echo"
}

// Description describes the echo operation.
func (t *EchoTool) Description() string {
	return "Returns the same text passed in arguments."
}

// Params returns the echo tool's argument schema.
func (t *EchoTool) Params() ai.ToolParameters {
	return ai.ToolParameters{
		Strict: true,
		Properties: []ai.ToolParameter{
			{
				Name:        "text",
				Type:        ai.ToolParameterString,
				Description: "Text to echo back",
				Required:    true,
			},
		},
	}
}

type echoArgs struct {
	Text string `json:"text"`
}

// Function decodes and returns the requested text.
func (t *EchoTool) Function(ctx context.Context, req ai.ToolCall) (string, error) {
	var args echoArgs
	if err := DecodeToolArgs(req, &args); err != nil {
		return "", err
	}

	text := strings.TrimSpace(args.Text)
	if text == "" {
		text = "(empty)"
	}
	return text, nil
}
