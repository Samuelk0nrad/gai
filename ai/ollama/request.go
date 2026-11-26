package ollama

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/lace-ai/gai/ai"
)

type chatRequest struct {
	Model    string           `json:"model"`
	Messages []chatMessage    `json:"messages"`
	Tools    []toolDefinition `json:"tools,omitempty"`
	Options  map[string]any   `json:"options,omitempty"`
	Stream   bool             `json:"stream"`
	Think    bool             `json:"think"`
}

type chatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolName   string     `json:"tool_name,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type toolCall struct {
	ID       string       `json:"id,omitempty"`
	Function toolFunction `json:"function"`
}

type toolFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type toolDefinition struct {
	Type     string                 `json:"type"`
	Function toolDefinitionFunction `json:"function"`
}

type toolDefinitionFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

func (m *Model) chatRequest(req ai.AIRequest) (*chatRequest, error) {
	req = req.Copy()
	if err := ai.ValidateModelRequest(m, req); err != nil {
		return nil, err
	}
	if req.MaxTokens < 0 {
		return nil, fmt.Errorf("max tokens must be non-negative")
	}
	if err := validateOptions(m.options); err != nil {
		return nil, err
	}
	messages, err := mapMessages(req.Messages)
	if err != nil {
		return nil, err
	}
	tools, err := mapTools(req.Tools)
	if err != nil {
		return nil, err
	}
	options := mapOptions(m.options)
	if req.MaxTokens > 0 {
		if options == nil {
			options = make(map[string]any)
		}
		options["num_predict"] = req.MaxTokens
	}
	return &chatRequest{
		Model:    m.name,
		Messages: messages,
		Stream:   true,
		Think:    false,
		Tools:    tools,
		Options:  options,
	}, nil
}

func mapMessages(messages []ai.Message) ([]chatMessage, error) {
	out := make([]chatMessage, 0, len(messages))
	for _, message := range messages {
		if err := rejectRequiredExtensions(message.Extensions); err != nil {
			return nil, err
		}
		wire := chatMessage{Role: string(message.Role)}
		for _, part := range message.Parts {
			if err := rejectRequiredExtensions(part.Extensions); err != nil {
				return nil, err
			}
			switch part.Kind {
			case ai.ContentText:
				if len(wire.ToolCalls) > 0 {
					return nil, fmt.Errorf("%w: Ollama text after tool calls", ai.ErrUnsupportedCapability)
				}
				wire.Content += part.Text
			case ai.ContentJSON:
				if len(wire.ToolCalls) > 0 {
					return nil, fmt.Errorf("%w: Ollama JSON after tool calls", ai.ErrUnsupportedCapability)
				}
				wire.Content += string(part.JSON)
			case ai.ContentToolCall:
				call := part.ToolCall
				if err := rejectRequiredExtensions(call.Extensions); err != nil {
					return nil, err
				}
				if !jsonObject(call.Args) {
					return nil, fmt.Errorf("%w: Ollama tool %q arguments must be a JSON object", ai.ErrInvalidToolCall, call.Name)
				}
				wire.ToolCalls = append(wire.ToolCalls, toolCall{
					ID: call.ID,
					Function: toolFunction{
						Name:      call.Name,
						Arguments: append(json.RawMessage(nil), call.Args...),
					},
				})
			case ai.ContentToolResult:
				content, err := toolResultContent(part.ToolResult)
				if err != nil {
					return nil, err
				}
				out = append(out, chatMessage{Role: "tool", Content: content, ToolName: part.ToolResult.Name, ToolCallID: part.ToolResult.ToolCallID})
			case ai.ContentExtension:
				// Optional extensions may be dropped. Required ones were rejected.
			default:
				return nil, &ai.UnsupportedContentError{Provider: "ollama", Kind: part.Kind, Detail: "no native representation"}
			}
		}
		if message.Role != ai.RoleTool {
			out = append(out, wire)
		}
	}
	return out, nil
}

func mapTools(definitions []ai.ToolDefinition) ([]toolDefinition, error) {
	if len(definitions) == 0 {
		return nil, nil
	}
	tools := make([]toolDefinition, 0, len(definitions))
	for _, definition := range definitions {
		if err := definition.Validate(); err != nil {
			return nil, err
		}
		if !jsonObject(definition.Parameters) {
			return nil, fmt.Errorf("%w: Ollama tool %q parameters must be a JSON object", ai.ErrInvalidToolDefinition, definition.Name)
		}
		tools = append(tools, toolDefinition{
			Type: "function",
			Function: toolDefinitionFunction{
				Name:        definition.Name,
				Description: definition.Description,
				Parameters:  append(json.RawMessage(nil), definition.Parameters...),
			},
		})
	}
	return tools, nil
}

func toolResultContent(result *ai.ToolResult) (string, error) {
	var content strings.Builder
	for _, part := range result.Parts {
		if err := rejectRequiredExtensions(part.Extensions); err != nil {
			return "", err
		}
		switch part.Kind {
		case ai.ContentText:
			content.WriteString(part.Text)
		case ai.ContentJSON:
			content.Write(part.JSON)
		default:
			return "", &ai.UnsupportedContentError{Provider: "ollama", Kind: part.Kind, Detail: "tool result content"}
		}
	}
	if result.IsError {
		encoded, _ := json.Marshal(map[string]string{"error": content.String()})
		return string(encoded), nil
	}
	return content.String(), nil
}

func rejectRequiredExtensions(extensions []ai.Extension) error {
	for _, extension := range extensions {
		if extension.Required {
			return fmt.Errorf("%w: Ollama extension %s/%s", ai.ErrUnsupportedCapability, extension.Namespace, extension.Type)
		}
	}
	return nil
}

func jsonObject(raw json.RawMessage) bool {
	var object map[string]json.RawMessage
	return json.Unmarshal(raw, &object) == nil && object != nil
}

func validateOptions(options Options) error {
	if options.NumCtx != nil && *options.NumCtx <= 0 {
		return fmt.Errorf("ollama num_ctx must be positive")
	}
	if options.Temperature != nil && (*options.Temperature < 0 || math.IsNaN(*options.Temperature) || math.IsInf(*options.Temperature, 0)) {
		return fmt.Errorf("ollama temperature must be finite and non-negative")
	}
	if options.TopP != nil && (*options.TopP < 0 || *options.TopP > 1 || math.IsNaN(*options.TopP) || math.IsInf(*options.TopP, 0)) {
		return fmt.Errorf("ollama top_p must be between 0 and 1")
	}
	return nil
}

func mapOptions(options Options) map[string]any {
	result := make(map[string]any)
	if options.NumCtx != nil {
		result["num_ctx"] = *options.NumCtx
	}
	if options.Temperature != nil {
		result["temperature"] = *options.Temperature
	}
	if options.TopP != nil {
		result["top_p"] = *options.TopP
	}
	if options.Seed != nil {
		result["seed"] = *options.Seed
	}
	if len(result) == 0 {
		return nil
	}
	return result
}
