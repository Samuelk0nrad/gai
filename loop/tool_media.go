package loop

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/Samuelk0nrad/gai/ai"
)

// ErrToolMediaUnsupported replaces a tool result whose media the model cannot
// read and Loop.Media did not convert. The model sees a tool error naming the
// media, and the run continues.
var ErrToolMediaUnsupported = fmt.Errorf("%w: media the model cannot read", ErrToolResultRejected)

// resultProcessor returns the configured processor followed by media
// adaptation for the run's model.
func (l *Loop) resultProcessor() ToolResultProcessor {
	adapter := l.mediaAdapter
	if adapter == nil {
		return l.ToolResultProcessor
	}
	media := ToolResultProcessorFunc(func(ctx context.Context, _ ToolPolicyInput, result ToolResult) (ToolResult, error) {
		return adaptToolResultMedia(ctx, adapter, result)
	})
	if l.ToolResultProcessor == nil {
		return media
	}
	chain, err := ChainToolResultProcessors(l.ToolResultProcessor, media)
	if err != nil {
		return l.ToolResultProcessor
	}
	return chain
}

// adaptToolResultMedia replaces media the model does not accept with what
// the media policy converts it to. Converted text is appended to Text, one
// line per part, and converted media replaces the original in Media. Media
// that cannot be sent becomes an ErrToolMediaUnsupported tool error.
func adaptToolResultMedia(ctx context.Context, adapter *ai.MediaAdapter, result ToolResult) (ToolResult, error) {
	if result.Err != nil || len(result.Media) == 0 {
		return result, nil
	}
	parts := make([]ai.ContentPart, len(result.Media))
	for i := range result.Media {
		parts[i] = ai.ContentPart{Kind: ai.ContentMedia, Media: &result.Media[i]}
	}
	adapted, err := adapter.Parts(ctx, parts, ai.MediaInToolResult)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ToolResult{}, ctxErr
		}
		placeholders := make([]string, len(result.Media))
		for i, media := range result.Media {
			placeholders[i] = media.Placeholder()
		}
		text := "the model cannot read the media this tool returned: " + strings.Join(placeholders, ", ")
		return ToolResult{Err: safeToolError{text: text, kind: errors.Join(ErrToolMediaUnsupported, ai.ErrUnsupportedCapability)}}, nil
	}
	out := ToolResult{Text: result.Text}
	var text strings.Builder
	text.WriteString(result.Text)
	for _, part := range adapted {
		switch part.Kind {
		case ai.ContentMedia:
			out.Media = append(out.Media, *part.Media)
		case ai.ContentText, ai.ContentJSON:
			if text.Len() > 0 {
				text.WriteString("\n")
			}
			if part.Kind == ai.ContentJSON {
				text.Write(part.JSON)
			} else {
				text.WriteString(part.Text)
			}
		}
	}
	out.Text = text.String()
	return out, nil
}
