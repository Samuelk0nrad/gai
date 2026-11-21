package anthropic

import (
	"encoding/json"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/streamtext"
)

const (
	extensionNamespace = "anthropic"
	// hostedToolResultExtension keeps a server tool result block, such as
	// web_search_tool_result, as the API returned it, for replay.
	hostedToolResultExtension = "hosted_tool_result"
)

// serverToolResultBlocks are the content blocks that carry the result of a
// server tool call. The API expects them back after their server_tool_use.
var serverToolResultBlocks = map[string]bool{
	"web_search_tool_result":                 true,
	"web_fetch_tool_result":                  true,
	"code_execution_tool_result":             true,
	"bash_code_execution_tool_result":        true,
	"text_editor_code_execution_tool_result": true,
	"tool_search_tool_result":                true,
	"container_upload":                       true,
}

func serverToolKind(name string) string {
	switch name {
	case "web_search":
		return ai.HostedToolWebSearch
	case "web_fetch":
		return ai.HostedToolWebFetch
	case "code_execution", "bash_code_execution", "text_editor_code_execution":
		return ai.HostedToolCodeExecution
	default:
		return ai.HostedToolOther
	}
}

// serverToolUsePart keeps a server_tool_use block as a hosted tool call.
func serverToolUsePart(raw json.RawMessage) ai.ContentPart {
	var block struct {
		ID    string `json:"id"`
		Name  string `json:"name"`
		Input struct {
			Query string `json:"query"`
			URL   string `json:"url"`
		} `json:"input"`
	}
	_ = json.Unmarshal(raw, &block)
	return ai.HostedToolCallPart(extensionNamespace, ai.HostedToolCall{
		ID: block.ID, Kind: serverToolKind(block.Name), Name: block.Name,
		Query: block.Input.Query, URL: block.Input.URL, Raw: append(json.RawMessage(nil), raw...),
	})
}

func serverToolResultPart(raw json.RawMessage) ai.ContentPart {
	return ai.ContentPart{Kind: ai.ContentExtension, Extensions: []ai.Extension{{Namespace: extensionNamespace, Type: hostedToolResultExtension, Data: append(json.RawMessage(nil), raw...)}}}
}

// streamedServerToolUse rebuilds a server_tool_use block from its start
// event and the input JSON its deltas carried.
func streamedServerToolUse(start string, input string) json.RawMessage {
	var block map[string]json.RawMessage
	if json.Unmarshal([]byte(start), &block) != nil {
		block = map[string]json.RawMessage{}
	}
	if input != "" && json.Valid([]byte(input)) {
		block["input"] = json.RawMessage(input)
	}
	raw, _ := json.Marshal(block)
	return raw
}

// hostedBlock returns the raw server tool block an extension part keeps for
// replay, or nil when the part holds none.
func hostedBlock(extensions []ai.Extension) json.RawMessage {
	for _, ext := range extensions {
		if ext.Namespace != extensionNamespace {
			continue
		}
		if ext.Type == hostedToolResultExtension && json.Valid(ext.Data) {
			return append(json.RawMessage(nil), ext.Data...)
		}
		if call, ok := ai.DecodeHostedToolCall(ext); ok && len(call.Raw) > 0 {
			return call.Raw
		}
	}
	return nil
}

// textCitations maps the citations of one text block. Anthropic cites a
// whole text block, so each citation spans text.
func textCitations(raws []json.RawMessage, text string) []ai.Citation {
	citations := make([]ai.Citation, 0, len(raws))
	for _, raw := range raws {
		var fields struct {
			Type          string `json:"type"`
			URL           string `json:"url"`
			Title         string `json:"title"`
			DocumentTitle string `json:"document_title"`
			CitedText     string `json:"cited_text"`
		}
		if json.Unmarshal(raw, &fields) != nil {
			continue
		}
		citation := ai.Citation{Type: fields.Type, URL: fields.URL, Title: fields.Title, Source: fields.CitedText, Text: text, Raw: append(json.RawMessage(nil), raw...)}
		if citation.Title == "" {
			citation.Title = fields.DocumentTitle
		}
		if n := ai.RuneLen(text); n > 0 {
			citation.End = n
		}
		citations = append(citations, citation)
	}
	return citations
}

// placeCitations shifts the citations of a token that attaches them to the
// caller's open text part, so their offsets cover the cited block's text at
// the end of that part. Other tokens are returned unchanged.
func placeCitations(run *streamtext.Run, token ai.Token) ai.Token {
	p := token.Part
	if p == nil || p.Kind != ai.ContentText || p.Text != "" || len(p.Extensions) != 1 || p.Extensions[0].Type != ai.ExtensionCitations || p.Extensions[0].Namespace != extensionNamespace {
		return token
	}
	citations := p.Citations()
	span := 0
	for _, c := range citations {
		span = max(span, c.End)
	}
	if placed, ok := run.AttachCitations(extensionNamespace, citations, span); ok {
		return placed
	}
	return token
}
