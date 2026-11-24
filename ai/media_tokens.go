package ai

import (
	"bytes"
	"context"
	"encoding/base64"
	"regexp"
	"strconv"
	"strings"
)

// Local media estimates. Providers encode media differently, and the byte size
// of an image says little about its token cost, so request budgets use fixed
// allowances instead of counting encoded bytes.
const (
	// mediaImageTokens is the allowance for one image. OpenAI's tile
	// accounting caps a high-detail image near 1,450 tokens, and its patch
	// accounting caps an image at 1,536 patches before model multipliers.
	// Anthropic resizes images to about 1.15 megapixels, about 1,600 tokens,
	// and Gemini tiles cost 258 tokens each. Large images on Mistral vision
	// models can cost more than this allowance.
	mediaImageTokens = 1600
	// mediaPDFPageTokens is the allowance for one PDF page. Providers
	// such as OpenAI send both the extracted text and an image of each page.
	mediaPDFPageTokens = 1500
	// mediaPDFUnknownPages is the page count assumed for a PDF given by URI,
	// whose size is unknown locally.
	mediaPDFUnknownPages = 10
	// mediaPDFBytesPerPage estimates pages when the page tree is compressed.
	mediaPDFBytesPerPage = 64 * 1024
)

// estimateMediaTokens returns the local allowance for one media part. Text
// media given as bytes is counted as text; images and PDFs use the fixed
// allowances above; other media counts one token per four bytes, at least one
// image allowance. Media given as a base64 data URI is decoded first, so it
// costs the same as the same bytes in Data. It never counts base64 text.
func estimateMediaTokens(ctx context.Context, media *MediaPart, counter TokenCounter) (int, error) {
	mimeType := normalizedMIME(media.MIMEType)
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		return mediaImageTokens, nil
	case mimeType == "application/pdf":
		return AddTokenCounts(pdfPageEstimate(media.inlineData()) * mediaPDFPageTokens)
	case IsTextMIME(mimeType):
		data := media.inlineData()
		if len(data) == 0 {
			return 0, nil
		}
		return countLocalText(ctx, string(data), counter)
	default:
		if n := (len(media.inlineData()) + 3) / 4; n > mediaImageTokens {
			return n, nil
		}
		return mediaImageTokens, nil
	}
}

// inlineData returns Data, or the decoded payload of a base64 data URI. It
// returns nil for other URIs and for data URIs that do not decode.
func (m *MediaPart) inlineData() []byte {
	if len(m.Data) > 0 {
		return m.Data
	}
	header, payload, ok := splitDataURI(m.URI)
	if !ok || !strings.HasSuffix(strings.ToLower(header), ";base64") {
		return nil
	}
	data, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return nil
	}
	return data
}

// splitDataURI splits "data:<header>,<payload>".
func splitDataURI(uri string) (header, payload string, ok bool) {
	if len(uri) < len("data:") || !strings.EqualFold(uri[:len("data:")], "data:") {
		return "", "", false
	}
	return strings.Cut(uri[len("data:"):], ",")
}

var (
	pdfPageObject = regexp.MustCompile(`/Type\s*/Page\b`)
	pdfPagesTree  = regexp.MustCompile(`/Type\s*/Pages\b`)
	pdfPageCount  = regexp.MustCompile(`/Count\s+(\d+)`)
)

// pdfPageEstimate reads the page count from an uncompressed page tree: the
// larger of the page objects found and the /Count of any /Type /Pages node.
// Other dictionaries, such as outlines, also carry /Count and are ignored.
// When the tree sits in a compressed object stream it falls back to the file
// size. A PDF without local data counts as mediaPDFUnknownPages.
func pdfPageEstimate(data []byte) int {
	if len(data) == 0 {
		return mediaPDFUnknownPages
	}
	pages := len(pdfPageObject.FindAllIndex(data, -1))
	for _, loc := range pdfPagesTree.FindAllIndex(data, -1) {
		start, end := pdfEnclosingDict(data, loc[0])
		if start < 0 {
			continue
		}
		for _, match := range pdfPageCount.FindAllSubmatch(data[start:end], -1) {
			if n, err := strconv.Atoi(string(match[1])); err == nil && n > pages && n < 1_000_000 {
				pages = n
			}
		}
	}
	if pages == 0 {
		pages = (len(data) + mediaPDFBytesPerPage - 1) / mediaPDFBytesPerPage
	}
	return max(pages, 1)
}

// pdfEnclosingDict returns the bounds of the innermost "<<...>>" dictionary
// around offset, skipping nested dictionaries, or -1 when there is none.
func pdfEnclosingDict(data []byte, offset int) (start, end int) {
	start = -1
	for i, depth := offset-1, 0; i > 0; i-- {
		if data[i-1] == '>' && data[i] == '>' {
			depth++
			i--
		} else if data[i-1] == '<' && data[i] == '<' {
			if depth == 0 {
				start = i - 1
				break
			}
			depth--
			i--
		}
	}
	if start < 0 {
		return -1, -1
	}
	for i, depth := offset, 0; i+1 < len(data); i++ {
		if bytes.HasPrefix(data[i:], []byte("<<")) {
			depth++
			i++
		} else if bytes.HasPrefix(data[i:], []byte(">>")) {
			if depth == 0 {
				return start, i + 2
			}
			depth--
			i++
		}
	}
	return -1, -1
}

// IsTextMIME reports whether a MIME type carries plain UTF-8 text that a
// provider without file input can send as text: text/* and JSON types.
// Parameters such as "; charset=utf-8" are ignored.
func IsTextMIME(mimeType string) bool {
	mimeType = normalizedMIME(mimeType)
	return strings.HasPrefix(mimeType, "text/") || mimeType == "application/json" || strings.HasSuffix(mimeType, "+json")
}

func normalizedMIME(mimeType string) string {
	mimeType, _, _ = strings.Cut(mimeType, ";")
	return strings.ToLower(strings.TrimSpace(mimeType))
}

// containsMedia reports whether parts, including nested tool-result parts,
// hold media.
func containsMedia(parts []ContentPart) bool {
	for _, part := range parts {
		if part.Media != nil {
			return true
		}
		if part.ToolResult != nil && containsMedia(part.ToolResult.Parts) {
			return true
		}
	}
	return false
}

// withoutMediaData returns a shallow copy of parts whose media keeps its MIME
// type, URI and name but not its bytes: Data is cleared and a data URI keeps
// only its header, such as "data:image/png;base64,". The input is not
// modified.
func withoutMediaData(parts []ContentPart) []ContentPart {
	out := make([]ContentPart, len(parts))
	for i, part := range parts {
		if part.Media != nil {
			media := *part.Media
			media.Data = nil
			if header, _, ok := splitDataURI(media.URI); ok {
				media.URI = "data:" + header + ","
			}
			part.Media = &media
		}
		if part.ToolResult != nil {
			result := *part.ToolResult
			result.Parts = withoutMediaData(result.Parts)
			part.ToolResult = &result
		}
		out[i] = part
	}
	return out
}

// estimateMessageMedia adds the media allowances of parts, including nested
// tool-result parts.
func estimateMessageMedia(ctx context.Context, parts []ContentPart, counter TokenCounter) (int, error) {
	total := 0
	for _, part := range parts {
		var tokens int
		var err error
		switch {
		case part.Media != nil:
			tokens, err = estimateMediaTokens(ctx, part.Media, counter)
		case part.ToolResult != nil:
			tokens, err = estimateMessageMedia(ctx, part.ToolResult.Parts, counter)
		}
		if err == nil {
			total, err = AddTokenCounts(total, tokens)
		}
		if err != nil {
			return 0, err
		}
	}
	return total, nil
}

// Placeholder returns a short text stand-in for media, such as
// "[image: board.jpg, image/png]" or "[file: application/pdf]", for text-only
// renderings like summaries. It never includes the bytes or the URI.
func (m MediaPart) Placeholder() string {
	mimeType := normalizedMIME(m.MIMEType)
	kind := "file"
	for _, prefix := range []string{"image", "audio", "video"} {
		if strings.HasPrefix(mimeType, prefix+"/") {
			kind = prefix
		}
	}
	// Collapse whitespace so a name cannot break the placeholder's line.
	if name := strings.Join(strings.Fields(m.Name), " "); name != "" {
		return "[" + kind + ": " + name + ", " + mimeType + "]"
	}
	return "[" + kind + ": " + mimeType + "]"
}
