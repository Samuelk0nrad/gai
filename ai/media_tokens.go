package ai

import (
	"context"
	"regexp"
	"strconv"
	"strings"
)

// Local media estimates. Providers encode media differently, and the byte size
// of an image says little about its token cost, so request budgets use fixed,
// deliberately high allowances instead of counting encoded bytes.
const (
	// mediaImageTokens is the allowance for one image. OpenAI's tile
	// accounting caps a high-detail image near 1,450 tokens, and its patch
	// accounting caps an image at 1,536 patches before model multipliers;
	// Mistral and Anthropic stay below this for typical images.
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
// image allowance. It never counts base64 or data URI text.
func estimateMediaTokens(ctx context.Context, media *MediaPart, counter TokenCounter) (int, error) {
	mimeType := normalizedMIME(media.MIMEType)
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		return mediaImageTokens, nil
	case mimeType == "application/pdf":
		return AddTokenCounts(pdfPageEstimate(media) * mediaPDFPageTokens)
	case IsTextMIME(mimeType):
		if len(media.Data) == 0 {
			return 0, nil
		}
		return countLocalText(ctx, string(media.Data), counter)
	default:
		if n := (len(media.Data) + 3) / 4; n > mediaImageTokens {
			return n, nil
		}
		return mediaImageTokens, nil
	}
}

var (
	pdfPageObject = regexp.MustCompile(`/Type\s*/Page\b`)
	pdfPageCount  = regexp.MustCompile(`/Count\s+(\d+)`)
)

// pdfPageEstimate reads the page count from an uncompressed page tree. When
// the tree sits in a compressed object stream it falls back to the file size.
func pdfPageEstimate(media *MediaPart) int {
	if len(media.Data) == 0 {
		return mediaPDFUnknownPages
	}
	pages := len(pdfPageObject.FindAllIndex(media.Data, -1))
	for _, match := range pdfPageCount.FindAllSubmatch(media.Data, -1) {
		if n, err := strconv.Atoi(string(match[1])); err == nil && n > pages && n < 1_000_000 {
			pages = n
		}
	}
	if pages == 0 {
		pages = (len(media.Data) + mediaPDFBytesPerPage - 1) / mediaPDFBytesPerPage
	}
	return max(pages, 1)
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
// type, URI and name but not its bytes. The input is not modified.
func withoutMediaData(parts []ContentPart) []ContentPart {
	out := make([]ContentPart, len(parts))
	for i, part := range parts {
		if part.Media != nil {
			media := *part.Media
			media.Data = nil
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
