package gemini

import (
	"context"

	"github.com/Samuelk0nrad/gai/ai"
	"google.golang.org/genai"
)

// ModelOption configures a concrete model at construction.
type ModelOption func(*Model)

// WithGenerateContentConfig appends a hook run on fresh config for synchronous
// and streaming calls. Hooks must be concurrency-safe and must not retain config
// pointers. Captured mutable data is caller-owned. Preserve portable tool/message
// semantics; use SDKClient for native contents, responses, or stream events.
func WithGenerateContentConfig(hook func(*genai.GenerateContentConfig) error) ModelOption {
	return func(m *Model) {
		if hook != nil {
			m.configHooks = append(m.configHooks, hook)
		}
	}
}

// SDKClient creates an independent configured SDK client. Calls retain native
// SDK types and bypass GAI preflight, observations, normalization, and retries.
// The SDK's transport/retry behavior and configured HTTP timeout apply.
func (p *Provider) SDKClient(ctx context.Context) (*genai.Client, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return p.getClient(ctx)
}

func (m *Model) generateContentConfig(req ai.AIRequest) (*genai.GenerateContentConfig, error) {
	config, err := buildGenerateContentConfig(req.Copy())
	if err != nil {
		return nil, err
	}
	if config == nil && len(m.configHooks) > 0 {
		config = &genai.GenerateContentConfig{}
	}
	for _, hook := range m.configHooks {
		if err := hook(config); err != nil {
			return nil, err
		}
	}
	preserveGenerateContentSchemaNumbers(config)
	return config, nil
}

func preserveGenerateContentSchemaNumbers(config *genai.GenerateContentConfig) {
	if config == nil {
		return
	}

	toolSchemas := make([][]any, len(config.Tools))
	hasSchemas := config.ResponseJsonSchema != nil
	for i, tool := range config.Tools {
		if tool == nil {
			continue
		}
		toolSchemas[i] = make([]any, len(tool.FunctionDeclarations))
		for j, declaration := range tool.FunctionDeclarations {
			if declaration == nil || declaration.ParametersJsonSchema == nil {
				continue
			}
			toolSchemas[i][j] = declaration.ParametersJsonSchema
			hasSchemas = true
		}
	}
	if !hasSchemas {
		return
	}

	if config.HTTPOptions == nil {
		config.HTTPOptions = &genai.HTTPOptions{}
	} else {
		httpOptions := *config.HTTPOptions
		config.HTTPOptions = &httpOptions
	}
	responseSchema := config.ResponseJsonSchema
	customize := config.HTTPOptions.ExtrasRequestProvider
	extraBody := cloneJSONMap(config.HTTPOptions.ExtraBody)
	config.HTTPOptions.ExtraBody = cloneJSONMap(extraBody)
	config.HTTPOptions.ExtrasRequestProvider = func(body map[string]any) map[string]any {
		// The SDK converts dynamic fields through float64 before this hook runs.
		// Restore the exact schemas, then reapply the ExtraBody snapshot because the
		// SDK merges those explicit overrides before calling this hook.
		if generationConfig, ok := body["generationConfig"].(map[string]any); ok && responseSchema != nil {
			generationConfig["responseJsonSchema"] = responseSchema
		}
		if tools := body["tools"]; tools != nil {
			for i, schemas := range toolSchemas {
				tool, ok := mapSliceItem(tools, i)
				if !ok {
					continue
				}
				declarations := tool["functionDeclarations"]
				for j, schema := range schemas {
					if schema == nil {
						continue
					}
					if declaration, ok := mapSliceItem(declarations, j); ok {
						declaration["parametersJsonSchema"] = schema
					}
				}
			}
		}
		mergeJSONMaps(body, extraBody)
		if customize != nil {
			return customize(body)
		}
		return body
	}
}

func mapSliceItem(values any, index int) (map[string]any, bool) {
	switch values := values.(type) {
	case []any:
		if index >= len(values) {
			return nil, false
		}
		value, ok := values[index].(map[string]any)
		return value, ok
	case []map[string]any:
		if index >= len(values) {
			return nil, false
		}
		return values[index], true
	default:
		return nil, false
	}
}

func cloneJSONMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	clone := make(map[string]any, len(value))
	for key, item := range value {
		clone[key] = cloneJSONValue(item)
	}
	return clone
}

func cloneJSONValue(value any) any {
	switch value := value.(type) {
	case map[string]any:
		return cloneJSONMap(value)
	case []any:
		clone := make([]any, len(value))
		for i, item := range value {
			clone[i] = cloneJSONValue(item)
		}
		return clone
	case []map[string]any:
		clone := make([]map[string]any, len(value))
		for i, item := range value {
			clone[i] = cloneJSONMap(item)
		}
		return clone
	default:
		return value
	}
}

func mergeJSONMaps(destination, source map[string]any) {
	if destination == nil || source == nil {
		return
	}
	for key, sourceValue := range source {
		destinationMap, destinationIsMap := destination[key].(map[string]any)
		sourceMap, sourceIsMap := sourceValue.(map[string]any)
		if destinationIsMap && sourceIsMap {
			mergeJSONMaps(destinationMap, sourceMap)
			continue
		}
		destination[key] = sourceValue
	}
}
