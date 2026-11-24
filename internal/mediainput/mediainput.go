// Package mediainput resolves canonical ai.MediaPart values into the forms
// provider adapters send: a remote URL, base64 data, or decoded text. Adapters
// share it so that every provider accepts and rejects the same inputs, and
// every rejection is an *ai.UnsupportedContentError naming the provider.
package mediainput

import (
	"encoding/base64"
	"encoding/xml"
	"fmt"
	"net/url"
	"strings"
	"unicode/utf8"

	"github.com/Samuelk0nrad/gai/ai"
)

// Kind classifies media by how providers accept it.
type Kind int

const (
	// KindOther is media no adapter sends, such as audio or video.
	KindOther Kind = iota
	// KindImage is image/jpeg, image/png, image/webp or image/gif.
	KindImage
	// KindPDF is application/pdf.
	KindPDF
	// KindText is text/*, application/json or a +json type, sent as text.
	KindText
)

// MIMEPDF is the PDF MIME type.
const MIMEPDF = "application/pdf"

var imageTypes = map[string]bool{
	"image/jpeg": true,
	"image/png":  true,
	"image/webp": true,
	"image/gif":  true,
}

// Source is resolved media. Exactly one of URL and Base64 is set.
type Source struct {
	// MIMEType is the normalized MIME type, such as "image/jpeg".
	MIMEType string
	// Kind classifies MIMEType.
	Kind Kind
	// URL is an http or https URL with a host.
	URL string
	// Base64 is the standard base64 encoding of inline bytes or of a base64
	// data URI's payload.
	Base64 string
	// Name is the trimmed MediaPart.Name.
	Name string
}

// NormalizeMIME lowercases a MIME type, drops parameters such as charset, and
// maps the image/jpg alias to image/jpeg.
func NormalizeMIME(mimeType string) string {
	mimeType, _, _ = strings.Cut(mimeType, ";")
	mimeType = strings.ToLower(strings.TrimSpace(mimeType))
	if mimeType == "image/jpg" {
		return "image/jpeg"
	}
	return mimeType
}

// Classify returns the Kind of a MIME type.
func Classify(mimeType string) Kind {
	mimeType = NormalizeMIME(mimeType)
	switch {
	case imageTypes[mimeType]:
		return KindImage
	case mimeType == MIMEPDF:
		return KindPDF
	case ai.IsTextMIME(mimeType):
		return KindText
	default:
		return KindOther
	}
}

// Unsupported returns an *ai.UnsupportedContentError for media.
func Unsupported(provider, format string, args ...any) error {
	return &ai.UnsupportedContentError{Provider: provider, Kind: ai.ContentMedia, Detail: fmt.Sprintf(format, args...)}
}

// SupportedTypes lists the MIME types Resolve accepts, for error messages.
const SupportedTypes = "image/jpeg, image/png, image/webp, image/gif, application/pdf, text/*, application/json"

// Resolve checks media and returns its source. Inline bytes and base64 data
// URIs whose MIME type matches the part become Base64; http(s) URLs with a
// host become URL. Other MIME types, schemes and data URIs are rejected.
func Resolve(provider string, media *ai.MediaPart) (Source, error) {
	mimeType := NormalizeMIME(media.MIMEType)
	source := Source{MIMEType: mimeType, Kind: Classify(mimeType), Name: strings.TrimSpace(media.Name)}
	what := "media"
	switch source.Kind {
	case KindImage:
		what = "image"
	case KindPDF:
		what = "PDF"
	case KindText:
		what = "text file"
	default:
		return Source{}, Unsupported(provider, "MIME type %q (supported: %s)", media.MIMEType, SupportedTypes)
	}
	if len(media.Data) > 0 {
		source.Base64 = base64.StdEncoding.EncodeToString(media.Data)
		return source, nil
	}
	parsed, err := url.Parse(media.URI)
	if err != nil {
		return Source{}, Unsupported(provider, "invalid %s URI", what)
	}
	switch strings.ToLower(parsed.Scheme) {
	case "http", "https":
		if parsed.Host == "" {
			return Source{}, Unsupported(provider, "%s URL requires a host", what)
		}
		source.URL = media.URI
		return source, nil
	case "data":
		uriMIME, payload, ok := ParseBase64DataURI(media.URI)
		if !ok || NormalizeMIME(uriMIME) != mimeType {
			return Source{}, Unsupported(provider, "%s data URI must be base64 encoded and match the part MIME type", what)
		}
		source.Base64 = payload
		return source, nil
	default:
		return Source{}, Unsupported(provider, "%s URI scheme %q (use http, https, data, or inline bytes)", what, parsed.Scheme)
	}
}

// DataURL returns "data:<mime>;base64,<payload>" for inline media.
func (s Source) DataURL() string {
	return "data:" + s.MIMEType + ";base64," + s.Base64
}

// Bytes decodes inline media.
func (s Source) Bytes(provider string) ([]byte, error) {
	data, err := base64.StdEncoding.DecodeString(s.Base64)
	if err != nil {
		return nil, Unsupported(provider, "media data URI is not valid base64")
	}
	return data, nil
}

// Text returns the UTF-8 text of inline text media. Text given by URL is
// rejected, because no provider fetches text files by URL.
func (s Source) Text(provider string) (string, error) {
	if s.URL != "" {
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
