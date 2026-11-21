package ai

import (
	"encoding/json"
	"unicode/utf8"
)

// Extension types shared by every provider that runs tools itself. The
// extension namespace names the provider that produced the data, and only
// that provider replays it.
const (
	// ExtensionHostedToolCall marks an extension part that holds one
	// HostedToolCall as JSON.
	ExtensionHostedToolCall = "hosted_tool_call"
	// ExtensionCitations marks an extension that holds a JSON array of
	// Citation. On a text part, the citations support that part's text.
	ExtensionCitations = "citations"
)

// Hosted tool kinds, normalized across providers.
const (
	HostedToolWebSearch       = "web_search"
	HostedToolWebFetch        = "web_fetch"
	HostedToolURLContext      = "url_context"
	HostedToolFileSearch      = "file_search"
	HostedToolCodeExecution   = "code_execution"
	HostedToolImageGeneration = "image_generation"
	HostedToolMaps            = "maps"
	HostedToolMCP             = "mcp"
	HostedToolOther           = "other"
)

// HostedToolCall is one tool call that the provider ran itself during a
// generation, such as a web search. The caller does not execute it, but it
// stays in the conversation so the provider can see it on the next request,
// and so callers can count hosted calls, which providers often bill per call.
type HostedToolCall struct {
	// Provider is the extension namespace the call came from, such as
	// "openai", "anthropic" or "google". It is set when the call is read.
	Provider string `json:"-"`
	// ID is the provider's call ID, when it has one.
	ID string `json:"id,omitempty"`
	// Kind is one of the HostedTool constants.
	Kind string `json:"kind"`
	// Name is the provider's own tool or item type, such as
	// "web_search_call", "web_search" or "GOOGLE_SEARCH_WEB".
	Name string `json:"name,omitempty"`
	// Status is the provider's status for the call, when it reports one.
	Status string `json:"status,omitempty"`
	// Query is the search query, when the call is a search.
	Query string `json:"query,omitempty"`
	// URL is the page the call opened or fetched, when it did.
	URL string `json:"url,omitempty"`
	// Raw is the provider's own item. The provider replays it unchanged. It is
	// empty when the call was reported only as metadata and is not replayed.
	Raw json.RawMessage `json:"raw,omitempty"`
}

// Citation links part of a response to a source, such as a web page found by
// a hosted search.
type Citation struct {
	// Provider is the extension namespace the citation came from. It is set
	// when the citation is read.
	Provider string `json:"-"`
	// Type is the provider's citation type, such as "url_citation" or
	// "web_search_result_location".
	Type  string `json:"type,omitempty"`
	URL   string `json:"url,omitempty"`
	Title string `json:"title,omitempty"`
	// Text is the response text the citation supports, when known.
	Text string `json:"text,omitempty"`
	// Source is the quoted source text, when the provider returns it.
	Source string `json:"source,omitempty"`
	// Start and End are rune offsets into the Text of the text part that
	// carries the citation. They are both zero when the provider gives no
	// position, or when the citation is not on a text part.
	Start int `json:"start,omitempty"`
	End   int `json:"end,omitempty"`
	// Raw is the provider's own citation JSON.
	Raw json.RawMessage `json:"raw,omitempty"`
}

// HostedToolCallPart returns an extension part that holds call. The part is
// optional, so other providers skip it.
func HostedToolCallPart(namespace string, call HostedToolCall) ContentPart {
	data, _ := json.Marshal(call)
	return ContentPart{Kind: ContentExtension, Extensions: []Extension{{Namespace: namespace, Type: ExtensionHostedToolCall, Data: data}}}
}

// CitationsExtension returns the optional extension that holds citations, or
// nil when there are none.
func CitationsExtension(namespace string, citations []Citation) []Extension {
	if len(citations) == 0 {
		return nil
	}
	data, _ := json.Marshal(citations)
	return []Extension{{Namespace: namespace, Type: ExtensionCitations, Data: data}}
}

// DecodeHostedToolCall decodes a hosted tool call extension.
func DecodeHostedToolCall(ext Extension) (HostedToolCall, bool) {
	if ext.Type != ExtensionHostedToolCall {
		return HostedToolCall{}, false
	}
	var call HostedToolCall
	if json.Unmarshal(ext.Data, &call) != nil {
		return HostedToolCall{}, false
	}
	call.Provider = ext.Namespace
	return call, true
}

// HostedToolCalls returns the hosted tool calls in m, in order.
func (m Message) HostedToolCalls() []HostedToolCall {
	var calls []HostedToolCall
	for _, part := range m.Parts {
		if part.Kind != ContentExtension {
			continue
		}
		for _, ext := range part.Extensions {
			if call, ok := DecodeHostedToolCall(ext); ok {
				calls = append(calls, call)
			}
		}
	}
	return calls
}

// Citations returns the citations kept on this part.
func (p ContentPart) Citations() []Citation {
	var out []Citation
	for _, ext := range p.Extensions {
		if ext.Type != ExtensionCitations {
			continue
		}
		var citations []Citation
		if json.Unmarshal(ext.Data, &citations) != nil {
			continue
		}
		for i := range citations {
			citations[i].Provider = ext.Namespace
		}
		out = append(out, citations...)
	}
	return out
}

// Citations returns the citations of every part in m, in order.
func (m Message) Citations() []Citation {
	var out []Citation
	for _, part := range m.Parts {
		out = append(out, part.Citations()...)
	}
	return out
}

// replayState lists the required extensions that GAI's provider adapters
// keep only so the same provider can continue a conversation: OpenAI
// reasoning items, Anthropic thinking signatures, redacted thinking and
// fallback blocks, and Google thought signatures. They hold no content a
// reader needs.
var replayState = map[[2]string]bool{
	{"openai", "responses_reasoning"}:   true,
	{"anthropic", "thinking_signature"}: true,
	{"anthropic", "redacted_thinking"}:  true,
	{"anthropic", "fallback"}:           true,
	{"google", "thought_signature"}:     true,
}

// WithoutProviderState returns a copy of m without provider state: optional
// extensions, the replay state GAI's provider adapters keep (reasoning items,
// thinking and thought signatures, fallback blocks), and extension parts left
// with no extensions. Reasoning text stays. Any other required extension is
// kept, so text renderers still reject it. ok is false when no part is left.
// Use it to show a message as text, such as in a summary prompt; never replay
// the result in place of m.
func (m Message) WithoutProviderState() (Message, bool) {
	out := Message{Role: m.Role, Extensions: keptExtensions(m.Extensions)}
	for _, part := range CloneParts(m.Parts) {
		part = withoutProviderState(part)
		if part.Kind == ContentExtension && len(part.Extensions) == 0 {
			continue
		}
		out.Parts = append(out.Parts, part)
	}
	return out, len(out.Parts) > 0
}

func withoutProviderState(part ContentPart) ContentPart {
	part.Extensions = keptExtensions(part.Extensions)
	if part.ToolCall != nil {
		part.ToolCall.Extensions = keptExtensions(part.ToolCall.Extensions)
	}
	if part.ToolResult != nil {
		kept := part.ToolResult.Parts[:0]
		for _, child := range part.ToolResult.Parts {
			child = withoutProviderState(child)
			if child.Kind == ContentExtension && len(child.Extensions) == 0 {
				continue
			}
			kept = append(kept, child)
		}
		part.ToolResult.Parts = kept
	}
	return part
}

// keptExtensions returns the required extensions that are not known replay
// state.
func keptExtensions(extensions []Extension) []Extension {
	var kept []Extension
	for _, ext := range extensions {
		if ext.Required && !replayState[[2]string{ext.Namespace, ext.Type}] {
			kept = append(kept, ext)
		}
	}
	return kept
}

// RuneLen returns the length of s in runes, the unit of Citation offsets.
func RuneLen(s string) int { return utf8.RuneCountInString(s) }
