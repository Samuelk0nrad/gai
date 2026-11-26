package ai

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ToolDefinition describes a function tool in a provider-neutral form.
type ToolDefinition struct {
	// Type identifies the tool kind. The only built-in kind is "function".
	Type string
	// Name is the function name exposed to the model.
	Name string
	// Description explains when and how the model should use the tool.
	Description string
	// Parameters contains the function parameters as JSON Schema.
	Parameters json.RawMessage
}

// ToolParameterType is a JSON Schema type supported by ToolParameters.
type ToolParameterType string

const (
	ToolParameterString  ToolParameterType = "string"
	ToolParameterNumber  ToolParameterType = "number"
	ToolParameterInteger ToolParameterType = "integer"
	ToolParameterBoolean ToolParameterType = "boolean"
	ToolParameterObject  ToolParameterType = "object"
	ToolParameterArray   ToolParameterType = "array"
)

// ToolParameters describes a function tool's object arguments.
type ToolParameters struct {
	Properties           []ToolParameter
	Strict               bool
	AdditionalProperties *bool
	// RawSchema is an exact JSON Schema escape hatch. When set, Properties,
	// Strict, and AdditionalProperties must remain unset.
	RawSchema json.RawMessage `json:"raw_schema,omitempty"`
}

// ToolParameter describes one property in a tool argument schema.
type ToolParameter struct {
	Name                 string
	Type                 ToolParameterType
	Description          string
	Required             bool
	Strict               bool
	Enum                 []any
	Default              any
	Items                *ToolParameter
	Properties           []ToolParameter
	AdditionalProperties *bool
}

type toolParameterSchema struct {
	Type                 ToolParameterType              `json:"type"`
	Description          string                         `json:"description,omitempty"`
	Required             []string                       `json:"required,omitempty"`
	Properties           map[string]toolParameterSchema `json:"properties,omitempty"`
	Enum                 []any                          `json:"enum,omitempty"`
	Default              any                            `json:"default,omitempty"`
	Items                *toolParameterSchema           `json:"items,omitempty"`
	AdditionalProperties *bool                          `json:"additionalProperties,omitempty"`
}

// JSONSchema validates and renders params as provider-facing JSON Schema.
func (params ToolParameters) JSONSchema() (json.RawMessage, error) {
	if len(params.RawSchema) > 0 {
		if len(params.Properties) > 0 || params.Strict || params.AdditionalProperties != nil {
			return nil, fmt.Errorf("%w: raw schema cannot be combined with typed parameters", ErrInvalidToolDefinition)
		}
		if !json.Valid(params.RawSchema) {
			return nil, fmt.Errorf("%w: raw parameters schema is invalid JSON", ErrInvalidToolDefinition)
		}
		var schema map[string]any
		if err := json.Unmarshal(params.RawSchema, &schema); err != nil {
			return nil, fmt.Errorf("%w: raw parameters schema: %w", ErrInvalidToolDefinition, err)
		}
		if typ, ok := schema["type"].(string); ok && typ != "object" {
			return nil, fmt.Errorf("%w: parameters type must be object", ErrInvalidToolDefinition)
		}
		return append(json.RawMessage(nil), params.RawSchema...), nil
	}
	schema, err := params.schema()
	if err != nil {
		return nil, err
	}
	raw, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("%w: marshal parameters schema: %w", ErrInvalidToolDefinition, err)
	}
	return json.RawMessage(raw), nil
}

func (params ToolParameters) schema() (toolParameterSchema, error) {
	result := toolParameterSchema{
		Type:                 ToolParameterObject,
		Properties:           map[string]toolParameterSchema{},
		AdditionalProperties: additionalPropertiesValue(params.Strict, params.AdditionalProperties),
	}
	required := make([]string, 0)
	for _, property := range params.Properties {
		name := strings.TrimSpace(property.Name)
		if name == "" {
			return toolParameterSchema{}, fmt.Errorf("%w: parameter name empty", ErrInvalidToolDefinition)
		}
		if _, exists := result.Properties[name]; exists {
			return toolParameterSchema{}, fmt.Errorf("%w: duplicate parameter %q", ErrInvalidToolDefinition, name)
		}
		child, err := property.schema(false)
		if err != nil {
			return toolParameterSchema{}, fmt.Errorf("parameter %q: %w", name, err)
		}
		result.Properties[name] = child
		if property.Required {
			required = append(required, name)
		}
	}
	if len(result.Properties) == 0 {
		result.Properties = nil
	}
	if len(required) > 0 {
		result.Required = required
	}
	return result, nil
}

func (param ToolParameter) schema(allowUnnamed bool) (toolParameterSchema, error) {
	if !allowUnnamed && strings.TrimSpace(param.Name) == "" {
		return toolParameterSchema{}, fmt.Errorf("%w: parameter name empty", ErrInvalidToolDefinition)
	}
	if !isSupportedToolParameterType(param.Type) {
		return toolParameterSchema{}, fmt.Errorf("%w: unsupported parameter type %q", ErrInvalidToolDefinition, param.Type)
	}
	result := toolParameterSchema{
		Type:                 param.Type,
		Description:          strings.TrimSpace(param.Description),
		Enum:                 param.Enum,
		Default:              param.Default,
		AdditionalProperties: additionalPropertiesValue(param.Strict, param.AdditionalProperties),
	}
	switch param.Type {
	case ToolParameterObject:
		properties := map[string]toolParameterSchema{}
		required := make([]string, 0)
		for _, property := range param.Properties {
			name := strings.TrimSpace(property.Name)
			if name == "" {
				return toolParameterSchema{}, fmt.Errorf("%w: nested parameter name empty", ErrInvalidToolDefinition)
			}
			if _, exists := properties[name]; exists {
				return toolParameterSchema{}, fmt.Errorf("%w: duplicate nested parameter %q", ErrInvalidToolDefinition, name)
			}
			child, err := property.schema(false)
			if err != nil {
				return toolParameterSchema{}, fmt.Errorf("nested parameter %q: %w", name, err)
			}
			properties[name] = child
			if property.Required {
				required = append(required, name)
			}
		}
		if len(properties) > 0 {
			result.Properties = properties
		}
		if len(required) > 0 {
			result.Required = required
		}
	case ToolParameterArray:
		if param.Items == nil {
			return toolParameterSchema{}, fmt.Errorf("%w: array parameter items empty", ErrInvalidToolDefinition)
		}
		itemSchema, err := param.Items.schema(true)
		if err != nil {
			return toolParameterSchema{}, fmt.Errorf("array items: %w", err)
		}
		result.Items = &itemSchema
	}
	return result, nil
}

func additionalPropertiesValue(strict bool, configured *bool) *bool {
	if configured != nil {
		return configured
	}
	if !strict {
		return nil
	}
	value := false
	return &value
}

func isSupportedToolParameterType(typ ToolParameterType) bool {
	switch typ {
	case ToolParameterString, ToolParameterNumber, ToolParameterInteger, ToolParameterBoolean, ToolParameterObject, ToolParameterArray:
		return true
	default:
		return false
	}
}

// ToolChoiceMode controls provider-side function calling.
type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceNone     ToolChoiceMode = "none"
)

// ToolChoice controls whether and how the model may call tools. Names may
// restrict calls only when Mode is ToolChoiceRequired.
type ToolChoice struct {
	Mode  ToolChoiceMode
	Names []string
}

// Validate checks that the tool-choice configuration is supported by the neutral API.
func (choice ToolChoice) Validate() error {
	switch choice.Mode {
	case "", ToolChoiceAuto, ToolChoiceNone:
		if len(choice.Names) != 0 {
			return fmt.Errorf("tool choice names require mode %q", ToolChoiceRequired)
		}
		return nil
	case ToolChoiceRequired:
		return nil
	default:
		return fmt.Errorf("unsupported tool choice mode %q", choice.Mode)
	}
}

// ResponseFormatType identifies the requested model response format.
type ResponseFormatType string

const (
	ResponseFormatText       ResponseFormatType = "text"
	ResponseFormatJSONObject ResponseFormatType = "json_object"
	ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)

// ResponseFormat requests text or structured JSON output.
type ResponseFormat struct {
	Type   ResponseFormatType
	Name   string
	Schema json.RawMessage
}

// ReasoningEffort is a provider-neutral reasoning intensity hint.
type ReasoningEffort string

const (
	ReasoningEffortNone    ReasoningEffort = "none"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
	ReasoningEffortMax     ReasoningEffort = "max"
)

// ReasoningConfig configures model reasoning/thinking behavior when supported.
type ReasoningConfig struct {
	Enabled         bool
	IncludeThoughts bool
	BudgetTokens    int
	Effort          ReasoningEffort
}

// NewToolDefinition validates and returns a provider-neutral function tool.
func NewToolDefinition(name, description string, parameters json.RawMessage) (ToolDefinition, error) {
	def := ToolDefinition{
		Type:        "function",
		Name:        strings.TrimSpace(name),
		Description: strings.TrimSpace(description),
		Parameters:  append(json.RawMessage(nil), parameters...),
	}
	if err := def.Validate(); err != nil {
		return ToolDefinition{}, err
	}
	return def, nil
}

// Validate checks that def is a usable function tool definition.
func (def ToolDefinition) Validate() error {
	if def.Type == "" {
		def.Type = "function"
	}
	if def.Type != "function" {
		return fmt.Errorf("%w: unsupported tool type %q", ErrInvalidToolDefinition, def.Type)
	}
	if strings.TrimSpace(def.Name) == "" {
		return fmt.Errorf("%w: name empty", ErrInvalidToolDefinition)
	}
	if strings.TrimSpace(def.Description) == "" {
		return fmt.Errorf("%w: description empty", ErrInvalidToolDefinition)
	}
	if len(def.Parameters) == 0 {
		return fmt.Errorf("%w: parameters empty", ErrInvalidToolDefinition)
	}
	if !json.Valid(def.Parameters) {
		return fmt.Errorf("%w: parameters invalid JSON", ErrInvalidToolDefinition)
	}
	var schema map[string]any
	if err := json.Unmarshal(def.Parameters, &schema); err != nil {
		return fmt.Errorf("%w: parameters schema: %w", ErrInvalidToolDefinition, err)
	}
	if typ, ok := schema["type"].(string); ok && typ != "object" {
		return fmt.Errorf("%w: parameters type must be object", ErrInvalidToolDefinition)
	}
	return nil
}

// Validate checks that format is internally consistent.
func (format ResponseFormat) Validate() error {
	switch format.Type {
	case "", ResponseFormatText, ResponseFormatJSONObject:
		return nil
	case ResponseFormatJSONSchema:
		if strings.TrimSpace(format.Name) == "" {
			return fmt.Errorf("%w: schema name empty", ErrInvalidResponseFormat)
		}
		if len(format.Schema) == 0 {
			return fmt.Errorf("%w: schema empty", ErrInvalidResponseFormat)
		}
		if !json.Valid(format.Schema) {
			return fmt.Errorf("%w: schema invalid JSON", ErrInvalidResponseFormat)
		}
		return nil
	default:
		return fmt.Errorf("%w: unsupported response format %q", ErrInvalidResponseFormat, format.Type)
	}
}
