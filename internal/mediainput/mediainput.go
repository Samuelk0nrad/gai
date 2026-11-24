// Package mediainput resolves canonical ai.MediaPart values into the forms
// provider adapters send: a remote URL, base64 data, or decoded text. Adapters
// share it so that every provider validates URIs and data URIs the same way,
// and every rejection is an *ai.UnsupportedContentError naming the provider.
// Which MIME types a provider accepts is decided by the adapter, with Types.
package mediainput

import (
	"encoding/base64"
	"encoding/xml"
	"fmt"
	"io"
	"net/url"
	"path"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/Samuelk0nrad/gai/ai"
)

// Kind classifies media by how providers accept it.
type Kind int

const (
	// KindOther is any type without a more specific kind.
	KindOther Kind = iota
	// KindImage is image/*.
	KindImage
	// KindPDF is application/pdf.
	KindPDF
	// KindText is text/*, application/json or a +json type.
	KindText
	// KindAudio is audio/*.
	KindAudio
	// KindVideo is video/*.
	KindVideo
)

// MIMEPDF is the PDF MIME type.
const MIMEPDF = "application/pdf"

// Source is resolved media. Exactly one of URL and inline data is set.
type Source struct {
	// MIMEType is the normalized MIME type, such as "image/jpeg".
	MIMEType string
	// Kind classifies MIMEType.
	Kind Kind
	// URL is an http or https URL with a host.
	URL string
	// Name is the trimmed MediaPart.Name.
	Name string

	data    []byte // inline bytes
	payload string // base64 payload of a data URI
}

// Inline reports whether the media is inline bytes or a data URI.
func (s Source) Inline() bool { return s.URL == "" }

// NormalizeMIME lowercases a MIME type, drops parameters such as charset, and
// maps common aliases: image/jpg to image/jpeg, audio/mp3 to audio/mpeg and
// audio/x-wav to audio/wav.
func NormalizeMIME(mimeType string) string {
	mimeType, _, _ = strings.Cut(mimeType, ";")
	mimeType = strings.ToLower(strings.TrimSpace(mimeType))
	switch mimeType {
	case "image/jpg":
		return "image/jpeg"
	case "audio/mp3":
		return "audio/mpeg"
	case "audio/x-wav", "audio/wave":
		return "audio/wav"
	}
	return mimeType
}

// Classify returns the Kind of a MIME type.
func Classify(mimeType string) Kind {
	mimeType = NormalizeMIME(mimeType)
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		return KindImage
	case mimeType == MIMEPDF:
		return KindPDF
	case ai.IsTextMIME(mimeType):
		return KindText
	case strings.HasPrefix(mimeType, "audio/"):
		return KindAudio
	case strings.HasPrefix(mimeType, "video/"):
		return KindVideo
	default:
		return KindOther
	}
}

// Types is a set of normalized MIME types an adapter accepts.
type Types []string

// Has reports whether the set holds mimeType after normalization.
func (t Types) Has(mimeType string) bool { return slices.Contains(t, NormalizeMIME(mimeType)) }

// String lists the types for error messages.
func (t Types) String() string { return strings.Join(t, ", ") }

// Unsupported returns an *ai.UnsupportedContentError for media.
func Unsupported(provider, format string, args ...any) error {
	return &ai.UnsupportedContentError{Provider: provider, Kind: ai.ContentMedia, Detail: fmt.Sprintf(format, args...)}
}

// Resolve validates where media comes from and returns its source. Inline
// bytes and base64 data URIs whose MIME type matches the part are inline;
// http(s) URLs with a host are URLs. Other schemes and data URIs are
// rejected. It does not check the MIME type against a provider's list and
// does not encode inline bytes.
func Resolve(provider string, media *ai.MediaPart) (Source, error) {
	mimeType := NormalizeMIME(media.MIMEType)
	source := Source{MIMEType: mimeType, Kind: Classify(mimeType), Name: strings.TrimSpace(media.Name)}
	if len(media.Data) > 0 {
		source.data = media.Data
		return source, nil
	}
	parsed, err := url.Parse(media.URI)
	if err != nil {
		return Source{}, Unsupported(provider, "invalid media URI")
	}
	switch strings.ToLower(parsed.Scheme) {
	case "http", "https":
		if parsed.Host == "" {
			return Source{}, Unsupported(provider, "media URL requires a host")
		}
		source.URL = media.URI
		return source, nil
	case "data":
		uriMIME, payload, ok := ParseBase64DataURI(media.URI)
		if !ok || NormalizeMIME(uriMIME) != mimeType {
			return Source{}, Unsupported(provider, "media data URI must be base64 encoded and match the part MIME type")
		}
		// Check the payload without keeping a decoded copy, so a model
		// never accepts media that mapping later fails to decode. The
		// stream decoder checks padding per chunk, so data after padding
		// in a later chunk is checked separately.
		if _, err := io.Copy(io.Discard, base64.NewDecoder(base64.StdEncoding, strings.NewReader(payload))); err != nil || payload == "" || !paddingAtEnd(payload) {
			return Source{}, Unsupported(provider, "media data URI is not valid base64")
		}
		source.payload = payload
		return source, nil
	default:
		return Source{}, Unsupported(provider, "media URI scheme %q (use http, https, data, or inline bytes)", parsed.Scheme)
	}
}

// Base64 returns the standard base64 encoding of inline media.
func (s Source) Base64() string {
	if s.payload != "" {
		return s.payload
	}
	return base64.StdEncoding.EncodeToString(s.data)
}

// DataURL returns "data:<mime>;base64,<payload>" for inline media.
func (s Source) DataURL() string {
	return "data:" + s.MIMEType + ";base64," + s.Base64()
}

// Bytes returns inline media bytes, decoding a data URI.
func (s Source) Bytes(provider string) ([]byte, error) {
	if s.payload == "" {
		return s.data, nil
	}
	data, err := base64.StdEncoding.DecodeString(s.payload)
	if err != nil {
		return nil, Unsupported(provider, "media data URI is not valid base64")
	}
	return data, nil
}

// Text returns the UTF-8 text of inline text media. Text given by URL is
// rejected, because no provider fetches text files by URL.
func (s Source) Text(provider string) (string, error) {
	if !s.Inline() {
		return "", Unsupported(provider, "text media %q must be inline data or a base64 data URI; text files are not fetched by URL", s.MIMEType)
	}
	data, err := s.Bytes(provider)
	if err != nil {
		return "", err
	}
	if !utf8.Valid(data) {
		return "", Unsupported(provider, "text media %q is not valid UTF-8", s.MIMEType)
	}
	return string(data), nil
}

// CheckText reports whether inline text media decodes to UTF-8 text, the
// check Text and TextFile make. Adapters call it when accepting media.
func (s Source) CheckText(provider string) error {
	_, err := s.Text(provider)
	return err
}

// TextFile renders text media as a <file> element that keeps the file name
// and MIME type, for providers without a text-file input:
//
//	<file name="notes.txt" mime_type="text/plain">
//	...
//	</file>
func (s Source) TextFile(provider string) (string, error) {
	text, err := s.Text(provider)
	if err != nil {
		return "", err
	}
	var out strings.Builder
	out.WriteString("<file")
	if s.Name != "" {
		out.WriteString(` name="`)
		_ = xml.EscapeText(&out, []byte(s.Name))
		out.WriteString(`"`)
	}
	out.WriteString(` mime_type="`)
	_ = xml.EscapeText(&out, []byte(s.MIMEType))
	out.WriteString("\">\n")
	out.WriteString(text)
	out.WriteString("\n</file>")
	return out.String(), nil
}

// Filename returns Name when it ends with ext, otherwise Name (or "file")
// with ext appended. Providers that detect a file type from its name need it.
func (s Source) Filename(ext string) string {
	name := s.Name
	if name == "" {
		name = "file"
	}
	if strings.EqualFold(path.Ext(name), ext) {
		return name
	}
	return name + ext
}

// ParseBase64DataURI splits "data:<mime>;base64,<payload>".
func ParseBase64DataURI(uri string) (mimeType, payload string, ok bool) {
	if len(uri) < len("data:") || !strings.EqualFold(uri[:len("data:")], "data:") {
		return "", "", false
	}
	header, payload, found := strings.Cut(uri[len("data:"):], ",")
	if !found {
		return "", "", false
	}
	mimeType, isBase64 := strings.CutSuffix(strings.ToLower(header), ";base64")
	if !isBase64 {
		return "", "", false
	}
	return mimeType, payload, true
}

// ContainsMedia reports whether parts hold a media part. It does not look
// inside tool results.
func ContainsMedia(parts []ai.ContentPart) bool {
	for _, part := range parts {
		if part.Kind == ai.ContentMedia {
			return true
		}
	}
	return false
}

// paddingAtEnd reports whether only padding and line breaks follow the first
// padding character, as base64.StdEncoding.DecodeString requires.
func paddingAtEnd(payload string) bool {
	i := strings.IndexByte(payload, '=')
	return i < 0 || strings.Trim(payload[i:], "=\r\n") == ""
}
