package mistral

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"

	"github.com/lace-ai/gai/ai"
)

const (
	// ExtensionNamespace identifies Mistral provider state stored in canonical
	// message extensions.
	ExtensionNamespace = "mistral"
	// ExtensionThinkingSignature stores a thinking chunk's replay signature on
	// the canonical reasoning part it belongs to. Data is a JSON string. The
	// extension is not required: other providers may drop it, while Mistral
	// replays it with the thinking chunk.
	ExtensionThinkingSignature = "thinking_signature"
)

const (
	chunkText     = "text"
	chunkThinking = "thinking"
	chunkImageURL = "image_url"
)

// chatContentChunk is one request content chunk. Only the fields of the
// selected type are encoded.
type chatContentChunk struct {
	Type      string
	Text      string
	Thinking  []chatContentChunk
	Signature string
	ImageURL  string
}

func (c chatContentChunk) MarshalJSON() ([]byte, error) {
	switch c.Type {
	case chunkText:
		return json.Marshal(struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}{c.Type, c.Text})
	case chunkThinking:
		thinking := c.Thinking
		if thinking == nil {
			thinking = []chatContentChunk{}
		}
		return json.Marshal(struct {
			Type      string             `json:"type"`
			Thinking  []chatContentChunk `json:"thinking"`
			Signature string             `json:"signature,omitempty"`
		}{c.Type, thinking, c.Signature})
	case chunkImageURL:
		return json.Marshal(struct {
			Type     string `json:"type"`
			ImageURL string `json:"image_url"`
		}{c.Type, c.ImageURL})
	default:
		return nil, fmt.Errorf("mistral: cannot encode content chunk %q", c.Type)
	}
}

func (c *chatContentChunk) UnmarshalJSON(data []byte) error {
	var wire struct {
		Type      string             `json:"type"`
		Text      string             `json:"text"`
		Thinking  []chatContentChunk `json:"thinking"`
		Signature *string            `json:"signature"`
		ImageURL  json.RawMessage    `json:"image_url"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*c = chatContentChunk{Type: wire.Type, Text: wire.Text, Thinking: wire.Thinking}
	if wire.Signature != nil {
		c.Signature = *wire.Signature
	}
	if len(wire.ImageURL) > 0 {
		imageURL, err := decodeImageURL(wire.ImageURL)
		if err != nil {
			return err
		}
		c.ImageURL = imageURL
	}
	return nil
}

func decodeImageURL(raw json.RawMessage) (string, error) {
	var value string
	if err := json.Unmarshal(raw, &value); err == nil {
		return value, nil
	}
	var object struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal(raw, &object); err != nil {
		return "", fmt.Errorf("decode image_url: %w", err)
	}
	return object.URL, nil
}

// chatContent is message content. It encodes as a plain string when every
// chunk is text, which keeps text-only requests in Mistral's simplest shape,
// and as an ordered chunk array otherwise.
type chatContent []chatContentChunk

func textContent(text string) chatContent {
	return chatContent{{Type: chunkText, Text: text}}
}

func (c chatContent) textOnly() bool {
	for _, chunk := range c {
		if chunk.Type != chunkText {
			return false
		}
	}
	return true
}

// Text returns the concatenated text chunks.
func (c chatContent) Text() string {
	var out strings.Builder
	for _, chunk := range c {
		if chunk.Type == chunkText {
			out.WriteString(chunk.Text)
		}
	}
	return out.String()
}

func (c chatContent) MarshalJSON() ([]byte, error) {
	if c.textOnly() {
		return json.Marshal(c.Text())
	}
	return json.Marshal([]chatContentChunk(c))
}

func (c *chatContent) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		*c = nil
		return nil
	}
	if trimmed[0] == '"' {
		var text string
		if err := json.Unmarshal(trimmed, &text); err != nil {
			return err
		}
		*c = textContent(text)
		return nil
	}
	var chunks []chatContentChunk
	if err := json.Unmarshal(trimmed, &chunks); err != nil {
		return err
	}
	*c = chunks
	return nil
}

// appendText adds text, merging it into a trailing text chunk.
func (c *chatContent) appendText(text string) {
	if n := len(*c); n > 0 && (*c)[n-1].Type == chunkText {
		(*c)[n-1].Text += text
		return
	}
	*c = append(*c, chatContentChunk{Type: chunkText, Text: text})
}

func reasoningChunk(part ai.ContentPart) (chatContentChunk, error) {
	signature, err := thinkingSignature(part.Extensions)
	if err != nil {
		return chatContentChunk{}, err
	}
	return chatContentChunk{
		Type:      chunkThinking,
		Thinking:  []chatContentChunk{{Type: chunkText, Text: part.Text}},
		Signature: signature,
	}, nil
}

func thinkingSignature(extensions []ai.Extension) (string, error) {
	signature := ""
	for _, ext := range extensions {
		if ext.Namespace != ExtensionNamespace || ext.Type != ExtensionThinkingSignature {
			continue
		}
		var value string
		if err := json.Unmarshal(ext.Data, &value); err != nil {
			return "", fmt.Errorf("mistral thinking signature must be a JSON string: %w", err)
		}
		if signature != "" && signature != value {
			return "", fmt.Errorf("mistral reasoning part has conflicting thinking signatures")
		}
		signature = value
	}
	return signature, nil
}

func signatureExtension(signature string) ai.Extension {
	data, _ := json.Marshal(signature)
	return ai.Extension{Namespace: ExtensionNamespace, Type: ExtensionThinkingSignature, Data: data}
}

// Supported inline and remote image MIME types for Mistral vision input.
var imageMIMETypes = map[string]bool{
	"image/jpeg": true,
	"image/png":  true,
	"image/webp": true,
	"image/gif":  true,
}

// normalizeImageMIME lowercases a MIME type and maps the common image/jpg
// alias to image/jpeg.
func normalizeImageMIME(mimeType string) string {
	mimeType = strings.ToLower(strings.TrimSpace(mimeType))
	if mimeType == "image/jpg" {
		return "image/jpeg"
	}
	return mimeType
}

func imageChunk(media *ai.MediaPart) (chatContentChunk, error) {
	mimeType := normalizeImageMIME(media.MIMEType)
	if !imageMIMETypes[mimeType] {
		return chatContentChunk{}, &ai.UnsupportedContentError{Provider: "mistral", Kind: ai.ContentMedia, Detail: fmt.Sprintf("MIME type %q (supported: image/jpeg, image/png, image/webp, image/gif)", media.MIMEType)}
	}
	if len(media.Data) > 0 {
		return chatContentChunk{Type: chunkImageURL, ImageURL: "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(media.Data)}, nil
	}
	parsed, err := url.Parse(media.URI)
	if err != nil {
		return chatContentChunk{}, &ai.UnsupportedContentError{Provider: "mistral", Kind: ai.ContentMedia, Detail: "invalid image URI"}
	}
	switch strings.ToLower(parsed.Scheme) {
	case "http", "https":
		if parsed.Host == "" {
			return chatContentChunk{}, &ai.UnsupportedContentError{Provider: "mistral", Kind: ai.ContentMedia, Detail: "image URL requires a host"}
		}
	case "data":
		header, payload, found := strings.Cut(media.URI[len("data:"):], ",")
		uriMIME, isBase64 := strings.CutSuffix(strings.ToLower(header), ";base64")
		if !found || !isBase64 || normalizeImageMIME(uriMIME) != mimeType {
			return chatContentChunk{}, &ai.UnsupportedContentError{Provider: "mistral", Kind: ai.ContentMedia, Detail: "data URI must be base64 encoded and match the part MIME type"}
		}
		// Send the canonical MIME type so aliases such as image/jpg are accepted.
		return chatContentChunk{Type: chunkImageURL, ImageURL: "data:" + mimeType + ";base64," + payload}, nil
	default:
		return chatContentChunk{}, &ai.UnsupportedContentError{Provider: "mistral", Kind: ai.ContentMedia, Detail: fmt.Sprintf("image URI scheme %q (use http, https, data, or inline bytes)", parsed.Scheme)}
	}
	return chatContentChunk{Type: chunkImageURL, ImageURL: media.URI}, nil
}

func containsImageInput(messages []ai.Message) bool {
	for _, message := range messages {
		for _, part := range message.Parts {
			if part.Kind == ai.ContentMedia {
				return true
			}
		}
	}
	return false
}

// responseChunk is one strictly decoded response chunk.
type responseChunk struct {
	Type      string            `json:"type"`
	Text      *string           `json:"text"`
	Thinking  []json.RawMessage `json:"thinking"`
	Signature *string           `json:"signature"`
}

// parseResponseContent converts response or delta content into ordered
// canonical parts. It accepts a plain string or a chunk array. Chunk types
// that GAI cannot represent faithfully are errors, never silently dropped.
func parseResponseContent(raw json.RawMessage) ([]ai.ContentPart, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, nil
	}
	switch trimmed[0] {
	case '"':
		var text string
		if err := json.Unmarshal(trimmed, &text); err != nil {
			return nil, fmt.Errorf("decode content: %w", err)
		}
		return []ai.ContentPart{{Kind: ai.ContentText, Text: text}}, nil
	case '[':
	default:
		return nil, fmt.Errorf("decode content: expected string or chunk array, got %s", firstBytes(trimmed))
	}
	var chunks []json.RawMessage
	if err := json.Unmarshal(trimmed, &chunks); err != nil {
		return nil, fmt.Errorf("decode content chunks: %w", err)
	}
	var parts []ai.ContentPart
	for i, rawChunk := range chunks {
		var chunk responseChunk
		if err := json.Unmarshal(rawChunk, &chunk); err != nil {
			return nil, fmt.Errorf("decode content chunk %d: %w", i, err)
		}
		switch chunk.Type {
		case chunkText, "":
			// An untyped chunk with text is treated as text, as before.
			if chunk.Text == nil {
				return nil, fmt.Errorf("decode content chunk %d: text chunk missing text", i)
			}
			parts = appendResponsePart(parts, ai.ContentPart{Kind: ai.ContentText, Text: *chunk.Text})
		case chunkThinking:
			text, err := thinkingText(chunk.Thinking)
			if err != nil {
				return nil, fmt.Errorf("decode content chunk %d: %w", i, err)
			}
			// Text first, then the signature, exactly as streamed deltas are
			// accumulated, so Generate and GenerateStream build the same parts.
			parts = appendResponsePart(parts, ai.ContentPart{Kind: ai.ContentReasoning, Text: text})
			if chunk.Signature != nil && *chunk.Signature != "" {
				parts = appendResponsePart(parts, ai.ContentPart{Kind: ai.ContentReasoning, Extensions: []ai.Extension{signatureExtension(*chunk.Signature)}})
			}
		default:
			return nil, unsupportedChunk(chunk.Type, false)
		}
	}
	return parts, nil
}

func thinkingText(items []json.RawMessage) (string, error) {
	var out strings.Builder
	for _, raw := range items {
		var item struct {
			Type string  `json:"type"`
			Text *string `json:"text"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			return "", fmt.Errorf("decode thinking item: %w", err)
		}
		if item.Type != chunkText && item.Type != "" {
			return "", unsupportedChunk(item.Type, true)
		}
		if item.Text == nil {
			return "", fmt.Errorf("thinking text item missing text")
		}
		out.WriteString(*item.Text)
	}
	return out.String(), nil
}

// appendResponsePart merges adjacent text or reasoning without extensions,
// matching ai.Message.AppendToken so synchronous and streamed output agree.
func appendResponsePart(parts []ai.ContentPart, part ai.ContentPart) []ai.ContentPart {
	if n := len(parts); n > 0 && parts[n-1].Kind == part.Kind && (part.Kind == ai.ContentText || part.Kind == ai.ContentReasoning) {
		last := &parts[n-1]
		if len(part.Extensions) == 0 && len(last.Extensions) == 0 {
			last.Text += part.Text
			return parts
		}
		if part.Text == "" && len(last.Extensions) == 0 {
			last.Extensions = append(last.Extensions, part.Extensions...)
			return parts
		}
	}
	return append(parts, part)
}

func unsupportedChunk(chunkType string, nested bool) error {
	where := "content"
	if nested {
		where = "thinking"
	}
	return fmt.Errorf("%w: Mistral %s chunk %q is not represented by GAI; use NativeClient for this response shape", ai.ErrUnsupportedCapability, where, chunkType)
}

func firstBytes(data []byte) string {
	if len(data) > 16 {
		return string(data[:16]) + "..."
	}
	return string(data)
}
