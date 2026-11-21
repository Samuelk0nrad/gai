package openai

import (
	"encoding/json"
	"fmt"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
)

const (
	// ExtensionNamespace is the namespace of OpenAI provider state kept on
	// canonical messages.
	ExtensionNamespace = "openai"
	// ExtensionHostedTool marks an extension part that holds one Responses
	// output item run by OpenAI itself, such as web_search_call. Data is the
	// item JSON exactly as OpenAI returned it.
	ExtensionHostedTool = "responses_hosted_tool"
	// ExtensionAnnotations marks the extension on a text part that holds the
	// output_text annotations array, such as url_citation, as OpenAI returned it.
	ExtensionAnnotations = "responses_annotations"
)

// hostedToolItems maps each hosted tool output item type that the Responses
// API accepts back as input to its input variant. The item JSON is replayed
// unchanged, so fields this SDK version does not model are kept too.
var hostedToolItems = map[string]func(json.RawMessage) responses.ResponseInputItemUnionParam{
	"web_search_call": func(raw json.RawMessage) responses.ResponseInputItemUnionParam {
		item := param.Override[responses.ResponseFunctionWebSearchParam](raw)
		return responses.ResponseInputItemUnionParam{OfWebSearchCall: &item}
	},
	"file_search_call": func(raw json.RawMessage) responses.ResponseInputItemUnionParam {
		item := param.Override[responses.ResponseFileSearchToolCallParam](raw)
		return responses.ResponseInputItemUnionParam{OfFileSearchCall: &item}
	},
	"code_interpreter_call": func(raw json.RawMessage) responses.ResponseInputItemUnionParam {
		item := param.Override[responses.ResponseCodeInterpreterToolCallParam](raw)
		return responses.ResponseInputItemUnionParam{OfCodeInterpreterCall: &item}
	},
	"image_generation_call": func(raw json.RawMessage) responses.ResponseInputItemUnionParam {
		item := param.Override[responses.ResponseInputItemImageGenerationCallParam](raw)
		return responses.ResponseInputItemUnionParam{OfImageGenerationCall: &item}
	},
	"mcp_call": func(raw json.RawMessage) responses.ResponseInputItemUnionParam {
		item := param.Override[responses.ResponseInputItemMcpCallParam](raw)
		return responses.ResponseInputItemUnionParam{OfMcpCall: &item}
	},
	"mcp_list_tools": func(raw json.RawMessage) responses.ResponseInputItemUnionParam {
		item := param.Override[responses.ResponseInputItemMcpListToolsParam](raw)
		return responses.ResponseInputItemUnionParam{OfMcpListTools: &item}
	},
}

func isHostedToolItem(itemType string) bool {
	_, ok := hostedToolItems[itemType]
	return ok
}

// hostedToolExtension keeps one hosted tool output item as an ordered
// extension part. It is not required: other providers may drop it without
// breaking their own protocol.
func hostedToolExtension(raw json.RawMessage) ai.ContentPart {
	data := append(json.RawMessage(nil), raw...)
	return ai.ContentPart{Kind: ai.ContentExtension, Extensions: []ai.Extension{{Namespace: ExtensionNamespace, Type: ExtensionHostedTool, Data: data}}}
}

func hostedToolInput(data json.RawMessage) (responses.ResponseInputItemUnionParam, error) {
	var head struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &head); err != nil {
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("decode OpenAI hosted tool item: %w", err)
	}
	build, ok := hostedToolItems[head.Type]
	if !ok {
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("%w: OpenAI hosted tool item %q", ai.ErrUnsupportedCapability, head.Type)
	}
	return build(append(json.RawMessage(nil), data...)), nil
}

// annotationsExtension returns the extension that keeps the output_text
// annotations on their text part, or nil when there are none.
func annotationsExtension(raw string) []ai.Extension {
	var annotations []json.RawMessage
	if raw == "" || json.Unmarshal([]byte(raw), &annotations) != nil || len(annotations) == 0 {
		return nil
	}
	return []ai.Extension{{Namespace: ExtensionNamespace, Type: ExtensionAnnotations, Data: json.RawMessage(raw)}}
}

// HostedToolCall is a typed view of one tool call that OpenAI ran itself
// during a Responses request, such as a web search. Callers can count these
// for billing without parsing raw provider JSON.
type HostedToolCall struct {
	// ID is the provider item ID, such as "ws_...".
	ID string
	// Type is the output item type, such as "web_search_call".
	Type string
	// Status is the item status, such as "completed".
	Status string
	// Action describes what a web search call did. It is zero for items
	// without an action.
	Action HostedToolAction
	// Raw is the item JSON exactly as OpenAI returned it.
	Raw json.RawMessage
}

// HostedToolAction is the action of a web_search_call item.
type HostedToolAction struct {
	// Type is "search", "open_page" or "find".
	Type string `json:"type"`
	// Query is the search query of a "search" action.
	Query string `json:"query,omitempty"`
	// URL is the page of an "open_page" or "find" action.
	URL string `json:"url,omitempty"`
	// Pattern is the text a "find" action looked for.
	Pattern string `json:"pattern,omitempty"`
}

// HostedToolCalls returns the hosted tool calls kept in m, in order.
func HostedToolCalls(m ai.Message) []HostedToolCall {
	var calls []HostedToolCall
	for _, part := range m.Parts {
		if part.Kind != ai.ContentExtension {
			continue
		}
		for _, ext := range part.Extensions {
			if ext.Namespace != ExtensionNamespace || ext.Type != ExtensionHostedTool {
				continue
			}
			var item struct {
				ID     string          `json:"id"`
				Type   string          `json:"type"`
				Status string          `json:"status"`
				Action json.RawMessage `json:"action"`
			}
			if json.Unmarshal(ext.Data, &item) != nil {
				continue
			}
			call := HostedToolCall{ID: item.ID, Type: item.Type, Status: item.Status, Raw: append(json.RawMessage(nil), ext.Data...)}
			if len(item.Action) > 0 {
				_ = json.Unmarshal(item.Action, &call.Action)
			}
			calls = append(calls, call)
		}
	}
	return calls
}

// Annotation is one output_text annotation, such as a url_citation. Fields
// that do not apply to its Type are empty. StartIndex and EndIndex are
// offsets into the text part that carries the annotation.
type Annotation struct {
	// Type is "url_citation", "file_citation", "container_file_citation" or
	// "file_path".
	Type       string `json:"type"`
	URL        string `json:"url,omitempty"`
	Title      string `json:"title,omitempty"`
	StartIndex int    `json:"start_index,omitempty"`
	EndIndex   int    `json:"end_index,omitempty"`
	FileID     string `json:"file_id,omitempty"`
	Filename   string `json:"filename,omitempty"`
	// Index is the position of a file citation or file path in the file list.
	Index int `json:"index,omitempty"`
	// ContainerID is the container of a container_file_citation.
	ContainerID string `json:"container_id,omitempty"`
}

// TextAnnotations returns the annotations kept on one text part.
func TextAnnotations(part ai.ContentPart) []Annotation {
	if part.Kind != ai.ContentText {
		return nil
	}
	var out []Annotation
	for _, ext := range part.Extensions {
		if ext.Namespace != ExtensionNamespace || ext.Type != ExtensionAnnotations {
			continue
		}
		var annotations []Annotation
		if json.Unmarshal(ext.Data, &annotations) == nil {
			out = append(out, annotations...)
		}
	}
	return out
}

// Citations returns the url_citation annotations of every text part in m,
// in order. Offsets are relative to the text part each one came from; use
// TextAnnotations to read them per part.
func Citations(m ai.Message) []Annotation {
	var out []Annotation
	for _, part := range m.Parts {
		for _, annotation := range TextAnnotations(part) {
			if annotation.Type == "url_citation" {
				out = append(out, annotation)
			}
		}
	}
	return out
}
