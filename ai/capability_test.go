package ai_test

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
)

func TestNewToolDefinitionValidatesSchema(t *testing.T) {
	def, err := ai.NewToolDefinition(
		"search",
		"Searches documents.",
		json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
	)
	if err != nil {
		t.Fatalf("NewToolDefinition error: %v", err)
	}
	if def.Type != "function" {
		t.Fatalf("expected function type, got %q", def.Type)
	}
	if def.Name != "search" {
		t.Fatalf("unexpected name: %q", def.Name)
	}
}

func TestNewToolDefinitionRejectsInvalidSchema(t *testing.T) {
	_, err := ai.NewToolDefinition("search", "Searches documents.", json.RawMessage(`{"type":"string"}`))
	if !errors.Is(err, ai.ErrInvalidToolDefinition) {
		t.Fatalf("expected ErrInvalidToolDefinition, got %v", err)
	}
}

func TestToolParametersJSONSchema(t *testing.T) {
	params := ai.ToolParameters{
		Strict: true,
		Properties: []ai.ToolParameter{
			{
				Name:        "query",
				Type:        ai.ToolParameterString,
				Description: "Search query",
				Required:    true,
			},
			{
				Name:    "mode",
				Type:    ai.ToolParameterString,
				Enum:    []any{"fast", "deep"},
				Default: "fast",
			},
			{
				Name:     "filters",
				Type:     ai.ToolParameterArray,
				Required: true,
				Items: &ai.ToolParameter{
					Type: ai.ToolParameterString,
				},
			},
			{
				Name: "options",
				Type: ai.ToolParameterObject,
				Properties: []ai.ToolParameter{
					{Name: "include_archived", Type: ai.ToolParameterBoolean, Required: true},
					{Name: "limit", Type: ai.ToolParameterInteger},
				},
			},
		},
	}

	got, err := params.JSONSchema()
	if err != nil {
		t.Fatalf("JSONSchema error: %v", err)
	}
	want := `{"type":"object","required":["query","filters"],"properties":{"filters":{"type":"array","items":{"type":"string"}},"mode":{"type":"string","enum":["fast","deep"],"default":"fast"},"options":{"type":"object","required":["include_archived"],"properties":{"include_archived":{"type":"boolean"},"limit":{"type":"integer"}}},"query":{"type":"string","description":"Search query"}},"additionalProperties":false}`
	if string(got) != want {
		t.Fatalf("JSONSchema = %s, want %s", got, want)
	}
}

func TestToolParametersAllowsExplicitAdditionalPropertiesOverride(t *testing.T) {
	allowExtra := true
	params := ai.ToolParameters{
		Strict:               true,
		AdditionalProperties: &allowExtra,
	}
	got, err := params.JSONSchema()
	if err != nil {
		t.Fatalf("JSONSchema error: %v", err)
	}
	want := `{"type":"object","additionalProperties":true}`
	if string(got) != want {
		t.Fatalf("JSONSchema = %s, want %s", got, want)
	}
}

func TestToolParametersJSONSchemaValidation(t *testing.T) {
	tests := []struct {
		name   string
		params ai.ToolParameters
	}{
		{
			name: "missing name",
			params: ai.ToolParameters{Properties: []ai.ToolParameter{
				{Type: ai.ToolParameterString},
			}},
		},
		{
			name: "unsupported type",
			params: ai.ToolParameters{Properties: []ai.ToolParameter{
				{Name: "value", Type: ai.ToolParameterType("unknown")},
			}},
		},
		{
			name: "array without items",
			params: ai.ToolParameters{Properties: []ai.ToolParameter{
				{Name: "values", Type: ai.ToolParameterArray},
			}},
		},
		{
			name: "duplicate name",
			params: ai.ToolParameters{Properties: []ai.ToolParameter{
				{Name: "value", Type: ai.ToolParameterString},
				{Name: "value", Type: ai.ToolParameterString},
			}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.params.JSONSchema()
			if !errors.Is(err, ai.ErrInvalidToolDefinition) {
				t.Fatalf("expected ErrInvalidToolDefinition, got %v", err)
			}
		})
	}
}

func TestResponseFormatValidation(t *testing.T) {
	format := ai.ResponseFormat{
		Type:   ai.ResponseFormatJSONSchema,
		Name:   "answer",
		Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}}}`),
	}
	if err := format.Validate(); err != nil {
		t.Fatalf("Validate error: %v", err)
	}

	format.Schema = json.RawMessage(`{`)
	if err := format.Validate(); !errors.Is(err, ai.ErrInvalidResponseFormat) {
		t.Fatalf("expected ErrInvalidResponseFormat, got %v", err)
	}
}

func TestToolChoiceValidationRejectsNamesWithoutRequiredMode(t *testing.T) {
	tests := []struct {
		name   string
		choice ai.ToolChoice
	}{
		{
			name:   "default auto",
			choice: ai.ToolChoice{Names: []string{"search"}},
		},
		{
			name:   "explicit auto",
			choice: ai.ToolChoice{Mode: ai.ToolChoiceAuto, Names: []string{"search"}},
		},
		{
			name:   "none",
			choice: ai.ToolChoice{Mode: ai.ToolChoiceNone, Names: []string{"search"}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.choice.Validate(); err == nil {
				t.Fatal("Validate error = nil, want error")
			}
		})
	}
}
