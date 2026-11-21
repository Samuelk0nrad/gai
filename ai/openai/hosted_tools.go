package openai

import (
	"encoding/json"
	"fmt"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/streamtext"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
)

const extensionNamespace = "openai"

// hostedToolItem describes a hosted tool output item type that the Responses
// API accepts back as input: its normalized kind and its input variant. The
// item JSON is replayed unchanged, so fields this SDK version does not model
// are kept too.
type hostedToolItem struct {
	kind  string
	input func(json.RawMessage) responses.ResponseInputItemUnionParam
}

var hostedToolItems = map[string]hostedToolItem{
	"web_search_call": {ai.HostedToolWebSearch, func(raw json.RawMessage) responses.ResponseInputItemUnionParam {
		item := param.Override[responses.ResponseFunctionWebSearchParam](raw)
		return responses.ResponseInputItemUnionParam{OfWebSearchCall: &item}
	}},
	"file_search_call": {ai.HostedToolFileSearch, func(raw json.RawMessage) responses.ResponseInputItemUnionParam {
		item := param.Override[responses.ResponseFileSearchToolCallParam](raw)
		return responses.ResponseInputItemUnionParam{OfFileSearchCall: &item}
	}},
	"code_interpreter_call": {ai.HostedToolCodeExecution, func(raw json.RawMessage) responses.ResponseInputItemUnionParam {
		item := param.Override[responses.ResponseCodeInterpreterToolCallParam](raw)
		return responses.ResponseInputItemUnionParam{OfCodeInterpreterCall: &item}
	}},
	"image_generation_call": {ai.HostedToolImageGeneration, func(raw json.RawMessage) responses.ResponseInputItemUnionParam {
		item := param.Override[responses.ResponseInputItemImageGenerationCallParam](raw)
		return responses.ResponseInputItemUnionParam{OfImageGenerationCall: &item}
	}},
	"mcp_call": {ai.HostedToolMCP, func(raw json.RawMessage) responses.ResponseInputItemUnionParam {
		item := param.Override[responses.ResponseInputItemMcpCallParam](raw)
		return responses.ResponseInputItemUnionParam{OfMcpCall: &item}
	}},
	"mcp_list_tools": {ai.HostedToolMCP, func(raw json.RawMessage) responses.ResponseInputItemUnionParam {
		item := param.Override[responses.ResponseInputItemMcpListToolsParam](raw)
		return responses.ResponseInputItemUnionParam{OfMcpListTools: &item}
	}},
}

func isHostedToolItem(itemType string) bool {
	_, ok := hostedToolItems[itemType]
	return ok
}

// hostedToolPart keeps one hosted tool output item as an ordered
// ai.HostedToolCall extension part.
func hostedToolPart(raw json.RawMessage) ai.ContentPart {
	var item struct {
		ID     string `json:"id"`
		Type   string `json:"type"`
		Status string `json:"status"`
		Action struct {
			Query string `json:"query"`
			URL   string `json:"url"`
		} `json:"action"`
	}
	_ = json.Unmarshal(raw, &item)
	return ai.HostedToolCallPart(extensionNamespace, ai.HostedToolCall{
		ID: item.ID, Kind: hostedToolItems[item.Type].kind, Name: item.Type, Status: item.Status,
		Query: item.Action.Query, URL: item.Action.URL, Raw: append(json.RawMessage(nil), raw...),
	})
}

func hostedToolInput(call ai.HostedToolCall) (responses.ResponseInputItemUnionParam, error) {
	var head struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(call.Raw, &head); err != nil {
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("decode OpenAI hosted tool item: %w", err)
	}
	item, ok := hostedToolItems[head.Type]
	if !ok {
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("%w: OpenAI hosted tool item %q", ai.ErrUnsupportedCapability, head.Type)
	}
	return item.input(append(json.RawMessage(nil), call.Raw...)), nil
}

// annotationCitations maps output_text annotations to citations on text,
// the text of their output_text content.
func annotationCitations(raw, text string) []ai.Citation {
	var annotations []json.RawMessage
	if raw == "" || json.Unmarshal([]byte(raw), &annotations) != nil {
		return nil
	}
	citations := make([]ai.Citation, 0, len(annotations))
	for _, annotation := range annotations {
		var fields struct {
			Type       string `json:"type"`
			URL        string `json:"url"`
			Title      string `json:"title"`
			Filename   string `json:"filename"`
			StartIndex int    `json:"start_index"`
			EndIndex   int    `json:"end_index"`
		}
		if json.Unmarshal(annotation, &fields) != nil {
			continue
		}
		citation := ai.Citation{Type: fields.Type, URL: fields.URL, Title: fields.Title, Raw: append(json.RawMessage(nil), annotation...)}
		if citation.Title == "" {
			citation.Title = fields.Filename
		}
		if span := streamtext.SliceRunes(text, fields.StartIndex, fields.EndIndex); span != "" {
			citation.Start, citation.End, citation.Text = fields.StartIndex, fields.EndIndex, span
		}
		citations = append(citations, citation)
	}
	return citations
}
