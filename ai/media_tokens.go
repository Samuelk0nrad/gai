package ai

import (
	"bytes"
	"context"
	"encoding/base64"
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
	// mediaAudioBytesPerToken estimates audio from its size. Gemini counts
	// 32 tokens per second; 128 bytes per token is 4 KB/s, a 32 kbit/s
	// stream, so most encodings are overestimated.
	mediaAudioBytesPerToken = 128
	// mediaVideoBytesPerToken estimates video from its size. Gemini counts
	// about 300 tokens per second at default resolution; 512 bytes per token
	// is about 1.2 Mbit/s.
	mediaVideoBytesPerToken = 512
	// mediaAudioURLTokens and mediaVideoURLTokens are the allowances for
	// audio or video given by URI, whose size is unknown locally: about ten
	// minutes of audio and about five minutes of video.
	mediaAudioURLTokens = 20_000
	mediaVideoURLTokens = 90_000
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
		pages, err := pdfPageEstimate(ctx, media.inlineData())
		if err != nil {
			return 0, err
		}
		return AddTokenCounts(pages * mediaPDFPageTokens)
	case IsTextMIME(mimeType):
		data := media.inlineData()
		if len(data) == 0 {
			return 0, nil
		}
		return countLocalText(ctx, string(data), counter)
	case strings.HasPrefix(mimeType, "audio/"):
		return sizedMediaTokens(media, mediaAudioBytesPerToken, mediaAudioURLTokens), nil
	case strings.HasPrefix(mimeType, "video/"):
		return sizedMediaTokens(media, mediaVideoBytesPerToken, mediaVideoURLTokens), nil
	default:
		if n := (len(media.inlineData()) + 3) / 4; n > mediaImageTokens {
			return n, nil
		}
		return mediaImageTokens, nil
	}
}

// sizedMediaTokens estimates media from its inline size, with at least one
// image allowance, or returns urlTokens for media given by URI.
func sizedMediaTokens(media *MediaPart, bytesPerToken, urlTokens int) int {
	data := media.inlineData()
	if len(data) == 0 {
		return urlTokens
	}
	return max((len(data)+bytesPerToken-1)/bytesPerToken, mediaImageTokens)
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

// pdfPageEstimate reads the page count from an uncompressed page tree: the
// larger of the /Type /Page dictionaries found and the /Count of any
// /Type /Pages dictionary. Only a dictionary's own /Count counts, so outline
// and nested resource dictionaries are ignored. When the tree sits in a
// compressed object stream it falls back to the file size. A PDF without
// local data counts as mediaPDFUnknownPages.
//
// It scans the file once, skipping stream bodies, and checks ctx every MiB.
func pdfPageEstimate(ctx context.Context, data []byte) (int, error) {
	if len(data) == 0 {
		return mediaPDFUnknownPages, nil
	}
	type dict struct {
		typ      string
		count    int
		hasCount bool
	}
	var stack []dict
	pages, treeCount := 0, 0
	nextCheck := 0
	for i := 0; i < len(data); i++ {
		// Skipped strings and streams can jump past any fixed offset.
		if i >= nextCheck {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
			nextCheck = i + 1<<20
		}
		switch c := data[i]; {
		case c == '<' && i+1 < len(data) && data[i+1] == '<':
			stack = append(stack, dict{})
			i++
		case c == '<':
			// A hex string; its digits are not syntax.
			if end := bytes.IndexByte(data[i+1:], '>'); end >= 0 {
				i += end + 1
			} else {
				i = len(data)
			}
		case c == '(':
			i = pdfSkipLiteralString(data, i)
		case c == '%':
			// A comment runs to the end of the line.
			for i+1 < len(data) && data[i+1] != '\n' && data[i+1] != '\r' {
				i++
			}
		case c == '>' && i+1 < len(data) && data[i+1] == '>':
			if n := len(stack); n > 0 {
				d := stack[n-1]
				stack = stack[:n-1]
				switch {
				case d.typ == "Page":
					pages++
				case d.typ == "Pages" && d.hasCount:
					treeCount = max(treeCount, d.count)
				}
			}
			i++
		case c == '/' && len(stack) > 0:
			name, next := pdfName(data, i+1)
			top := &stack[len(stack)-1]
			j := pdfSkipSpace(data, next)
			switch {
			case name == "Type" && j < len(data) && data[j] == '/':
				top.typ, next = pdfName(data, j+1)
			case name == "Count":
				if n, end := pdfUint(data, j); end > j {
					top.count, top.hasCount, next = n, true, end
				}
			}
			i = next - 1
		case c == 's' && len(stack) == 0 && bytes.HasPrefix(data[i:], []byte("stream")):
			// Stream bodies are binary and may hold any bytes; skip them.
			end := bytes.Index(data[i+len("stream"):], []byte("endstream"))
			if end < 0 {
				i = len(data)
				break
			}
			i += len("stream") + end + len("endstream") - 1
		}
	}
	pages = max(pages, treeCount)
	if pages == 0 {
		pages = (len(data) + mediaPDFBytesPerPage - 1) / mediaPDFBytesPerPage
	}
	return max(pages, 1), nil
}

// pdfSkipLiteralString returns the index of the parenthesis that closes the
// literal string opening at i, honoring escapes and balanced parentheses, or
// the last index when it is unterminated.
func pdfSkipLiteralString(data []byte, i int) int {
	depth := 0
	for ; i < len(data); i++ {
		switch data[i] {
		case '\\':
			i++
		case '(':
			depth++
		case ')':
			if depth--; depth == 0 {
				return i
			}
		}
	}
	return len(data) - 1
}

// pdfName reads a PDF name after its slash and returns it with the index
// after it.
func pdfName(data []byte, i int) (string, int) {
	start := i
	for i < len(data) && !pdfDelimiter(data[i]) {
		i++
	}
	return string(data[start:i]), i
}

// pdfUint reads a non-negative integer below one million; end equals i when
// there is none.
func pdfUint(data []byte, i int) (n, end int) {
	end = i
	for end < len(data) && data[end] >= '0' && data[end] <= '9' && end-i < 7 {
		n = n*10 + int(data[end]-'0')
		end++
	}
	if end < len(data) && data[end] >= '0' && data[end] <= '9' {
		return 0, i
	}
	return n, end
}

func pdfSkipSpace(data []byte, i int) int {
	for i < len(data) && pdfSpace(data[i]) {
		i++
	}
	return i
}

func pdfSpace(c byte) bool {
	return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f' || c == 0
}

func pdfDelimiter(c byte) bool {
	return pdfSpace(c) || strings.IndexByte("()<>[]{}/%", c) >= 0
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
