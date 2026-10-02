package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"strings"
)

// Role identifies the author of a semantic message, independent of storage or transport.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ContentKind selects exactly one payload in ContentPart.
type ContentKind string

const (
	ContentText       ContentKind = "text"
	ContentReasoning  ContentKind = "reasoning"
	ContentToolCall   ContentKind = "tool_call"
	ContentToolResult ContentKind = "tool_result"
	ContentJSON       ContentKind = "json"
	ContentMedia      ContentKind = "media"
	ContentExtension  ContentKind = "extension"
)

// Extension retains opaque provider state on the message or part it belongs to.
// Data is JSON, including a base64 JSON string for opaque bytes. Adapters must
// reject required extensions they cannot replay. Storage preserves all extensions.
type Extension struct {
	Namespace string          `json:"namespace"`
	Type      string          `json:"type"`
	Data      json.RawMessage `json:"data"`
	Required  bool            `json:"required,omitempty"`
}

// MediaPart preserves media without requiring a provider-specific SDK type.
// Exactly one of URI and Data must be supplied. Text renderers cannot encode media.
type MediaPart struct {
	MIMEType string `json:"mime_type"`
	URI      string `json:"uri,omitempty"`
	Data     []byte `json:"data,omitempty"`
}

// ContentPart is an ordered, serializable semantic payload. Empty text is valid.
// Extensions belong to this exact part, not to a flattened response string.
type ContentPart struct {
	Kind       ContentKind     `json:"kind"`
	Text       string          `json:"text,omitempty"`
	ToolCall   *ToolCall       `json:"tool_call,omitempty"`
	ToolResult *ToolResult     `json:"tool_result,omitempty"`
	JSON       json.RawMessage `json:"json,omitempty"`
	Media      *MediaPart      `json:"media,omitempty"`
	Extensions []Extension     `json:"extensions,omitempty"`
}

// ToolResult records the result for one identified tool call. Result parts may
// contain text, JSON, media or extensions, but cannot invoke another tool.
type ToolResult struct {
	ToolCallID string        `json:"tool_call_id"`
	Name       string        `json:"name"`
	Parts      []ContentPart `json:"parts"`
	IsError    bool          `json:"is_error,omitempty"`
}

// Message is the canonical conversation value used by context, execution,
// providers and persistence. Storage IDs and execution attempts belong to wrappers.
type Message struct {
	Role       Role          `json:"role"`
	Parts      []ContentPart `json:"parts"`
	Extensions []Extension   `json:"extensions,omitempty"`
}

func TextParts(text string) []ContentPart        { return []ContentPart{{Kind: ContentText, Text: text}} }
func TextMessage(role Role, text string) Message { return Message{Role: role, Parts: TextParts(text)} }
func (m Message) Text() string                   { return partsText(m.Parts, ContentText) }
func (m Message) Reasoning() string              { return partsText(m.Parts, ContentReasoning) }
func (r ToolResult) Text() string                { return partsText(r.Parts, ContentText) }
func partsText(parts []ContentPart, kind ContentKind) string {
	var out strings.Builder
	for _, p := range parts {
		if p.Kind == kind {
			out.WriteString(p.Text)
		}
		if kind == ContentText && p.Kind == ContentJSON {
			out.Write(p.JSON)
		}
	}
	return out.String()
}
func (m Message) ToolCalls() []ToolCall {
	var calls []ToolCall
	for _, p := range m.Parts {
		if p.ToolCall != nil {
			calls = append(calls, p.ToolCall.Clone())
		}
	}
	return calls
}
func (m Message) ToolResults() []ToolResult {
	var results []ToolResult
	for _, p := range m.Parts {
		if p.ToolResult != nil {
			r := *p.ToolResult
			r.Parts = CloneParts(r.Parts)
			results = append(results, r)
		}
	}
	return results
}
func CloneExtensions(src []Extension) []Extension {
	if src == nil {
		return nil
	}
	out := append([]Extension{}, src...)
	for i := range out {
		out[i].Data = append(json.RawMessage(nil), out[i].Data...)
	}
	return out
}
func CloneParts(src []ContentPart) []ContentPart {
	if src == nil {
		return nil
	}
	out := append([]ContentPart{}, src...)
	for i := range out {
		p := &out[i]
		p.Extensions = CloneExtensions(p.Extensions)
		p.JSON = append(json.RawMessage(nil), p.JSON...)
		if p.ToolCall != nil {
			c := p.ToolCall.Clone()
			p.ToolCall = &c
		}
		if p.ToolResult != nil {
			r := *p.ToolResult
			r.Parts = CloneParts(r.Parts)
			p.ToolResult = &r
		}
		if p.Media != nil {
			v := *p.Media
			v.Data = append([]byte(nil), v.Data...)
			p.Media = &v
		}
	}
	return out
}
func (m Message) Clone() Message {
	m.Parts = CloneParts(m.Parts)
	m.Extensions = CloneExtensions(m.Extensions)
	return m
}
func CloneMessages(src []Message) []Message {
	if src == nil {
		return nil
	}
	out := make([]Message, len(src))
	for i := range src {
		out[i] = src[i].Clone()
	}
	return out
}
func validateExtensions(extensions []Extension) error {
	for _, e := range extensions {
		if strings.TrimSpace(e.Namespace) == "" || strings.TrimSpace(e.Type) == "" || !json.Valid(e.Data) {
			return fmt.Errorf("invalid message extension")
		}
	}
	return nil
}
func (p ContentPart) Validate() error {
	if err := validateExtensions(p.Extensions); err != nil {
		return err
	}
	// Text has an implicit presence tag; the other payloads have explicit presence.
	count := 0
	if p.ToolCall != nil {
		count++
	}
	if p.ToolResult != nil {
		count++
	}
	if p.JSON != nil {
		count++
	}
	if p.Media != nil {
		count++
	}
	switch p.Kind {
	case ContentText, ContentReasoning:
		if count != 0 {
			return fmt.Errorf("%s part has another payload", p.Kind)
		}
	case ContentToolCall:
		if count != 1 || p.ToolCall == nil || p.Text != "" {
			return fmt.Errorf("invalid tool call part")
		}
		if err := p.ToolCall.Validate(); err != nil {
			return err
		}
		if !json.Valid(p.ToolCall.Args) {
			return fmt.Errorf("%w: arguments must be JSON", ErrInvalidToolCall)
		}
		if err := validateExtensions(p.ToolCall.Extensions); err != nil {
			return err
		}
	case ContentToolResult:
		if count != 1 || p.ToolResult == nil || p.Text != "" {
			return fmt.Errorf("invalid tool result part")
		}
		r := p.ToolResult
		if r.ToolCallID == "" || r.Name == "" {
			return fmt.Errorf("tool result requires call ID and name")
		}
		for _, part := range r.Parts {
			if part.Kind == ContentToolCall || part.Kind == ContentToolResult || part.Kind == ContentReasoning {
				return fmt.Errorf("invalid nested tool result part %q", part.Kind)
			}
			if err := part.Validate(); err != nil {
				return err
			}
		}
	case ContentJSON:
		if count != 1 || !json.Valid(p.JSON) || p.Text != "" {
			return fmt.Errorf("invalid JSON part")
		}
	case ContentMedia:
		if count != 1 || p.Media == nil || p.Text != "" {
			return fmt.Errorf("invalid media part")
		}
		if p.Media.MIMEType == "" || ((p.Media.URI == "") == (len(p.Media.Data) == 0)) {
			return fmt.Errorf("media requires MIME type and exactly one URI or data payload")
		}
	case ContentExtension:
		if count != 0 || p.Text != "" || len(p.Extensions) == 0 {
			return fmt.Errorf("invalid extension part")
		}
	default:
		return fmt.Errorf("unknown content kind %q", p.Kind)
	}
	return nil
}
func (m Message) Validate() error {
	switch m.Role {
	case RoleSystem, RoleUser, RoleAssistant, RoleTool:
	default:
		return fmt.Errorf("invalid message role %q", m.Role)
	}
	if len(m.Parts) == 0 {
		return fmt.Errorf("message requires content parts")
	}
	if err := validateExtensions(m.Extensions); err != nil {
		return err
	}
	for _, p := range m.Parts {
		if err := p.Validate(); err != nil {
			return err
		}
		if (p.Kind == ContentToolCall || p.Kind == ContentReasoning) && m.Role != RoleAssistant {
			return fmt.Errorf("%s requires assistant role", p.Kind)
		}
		if p.Kind == ContentToolResult && m.Role != RoleTool {
			return fmt.Errorf("tool result requires tool role")
		}
		if m.Role == RoleTool && p.Kind != ContentToolResult {
			return fmt.Errorf("tool message requires tool result parts")
		}
	}
	return nil
}

// UnsupportedContentError reports content a transport cannot preserve.
type UnsupportedContentError struct {
	Provider string
	Kind     ContentKind
	Detail   string
}

func (e *UnsupportedContentError) Error() string {
	return fmt.Sprintf("%s: %s content unsupported: %s", e.Provider, e.Kind, e.Detail)
}
func (e *UnsupportedContentError) Unwrap() error { return ErrUnsupportedCapability }

// RenderMessages is the explicit text renderer for canonical messages.
// It preserves roles, order and tool identity. Opaque state and media produce
// explicit errors instead of being silently dropped or exposed as prompt text.
func RenderMessages(ctx context.Context, messages []Message) (string, error) {
	var out bytes.Buffer
	esc := func(s string) { _ = xml.EscapeText(&out, []byte(s)) }
	for _, m := range messages {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		if err := m.Validate(); err != nil {
			return "", err
		}
		if len(m.Extensions) > 0 {
			return "", &UnsupportedContentError{Provider: "text", Kind: ContentExtension, Detail: "message extensions"}
		}
		out.WriteString("<" + string(m.Role) + ">\n")
		for _, p := range m.Parts {
			if err := renderContentPart(&out, p, esc); err != nil {
				return "", err
			}
			out.WriteByte('\n')
		}
		out.WriteString("</" + string(m.Role) + ">\n")
	}
	return out.String(), nil
}
func renderContentPart(out *bytes.Buffer, p ContentPart, esc func(string)) error {
	if len(p.Extensions) > 0 {
		return &UnsupportedContentError{Provider: "text", Kind: ContentExtension, Detail: "part extensions"}
	}
	switch p.Kind {
	case ContentText:
		esc(p.Text)
	case ContentReasoning:
		out.WriteString("<reasoning>")
		esc(p.Text)
		out.WriteString("</reasoning>")
	case ContentJSON:
		out.WriteString("<json>")
		esc(string(p.JSON))
		out.WriteString("</json>")
	case ContentToolCall:
		c := p.ToolCall
		if len(c.Extensions) > 0 {
			return &UnsupportedContentError{Provider: "text", Kind: ContentExtension, Detail: "tool continuity state"}
		}
		out.WriteString("<tool_call id=\"")
		esc(c.ID)
		out.WriteString("\" name=\"")
		esc(c.Name)
		out.WriteString("\">")
		esc(string(c.Args))
		out.WriteString("</tool_call>")
	case ContentToolResult:
		r := p.ToolResult
		out.WriteString("<tool_result id=\"")
		esc(r.ToolCallID)
		out.WriteString("\" name=\"")
		esc(r.Name)
		fmt.Fprintf(out, "\" is_error=\"%t\">", r.IsError)
		for _, child := range r.Parts {
			if err := renderContentPart(out, child, esc); err != nil {
				return err
			}
		}
		out.WriteString("</tool_result>")
	default:
		return &UnsupportedContentError{Provider: "text", Kind: p.Kind, Detail: "no text representation"}
	}
	return nil
}
