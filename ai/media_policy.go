package ai

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"strings"
	"sync"
)

// MediaPlacement says where a media part sits in a request.
type MediaPlacement uint8

const (
	// MediaInUserMessage is media directly in a user message.
	MediaInUserMessage MediaPlacement = iota + 1
	// MediaInToolResult is media inside a tool result.
	MediaInToolResult
)

func (p MediaPlacement) String() string {
	switch p {
	case MediaInUserMessage:
		return "user message"
	case MediaInToolResult:
		return "tool result"
	default:
		return fmt.Sprintf("MediaPlacement(%d)", uint8(p))
	}
}

// MediaAcceptor is an optional Model capability. AcceptsMedia reports, without
// I/O, whether the model's adapter sends media at placement: nil when it does,
// otherwise an error matching ErrUnsupportedCapability, usually an
// *UnsupportedContentError. A Model without this capability is assumed to
// accept all media, and its adapter reports unsupported media when a request
// is mapped.
type MediaAcceptor interface {
	AcceptsMedia(media MediaPart, placement MediaPlacement) error
}

// ErrMediaNotConverted is returned by a MediaConverter that declines to
// convert a media part. The MediaPolicy fallback then applies.
var ErrMediaNotConverted = errors.New("media not converted")

// MediaConverter converts media a model cannot send into content it can, such
// as a .docx file into PDF or text, a HEIC photo into JPEG, audio into a
// transcript, or video into still frames and a transcript.
//
// ConvertMedia returns the replacement parts: text, JSON, and media that
// accepts approves. accepts reports whether the model takes a candidate media
// part at the original placement, so a converter can choose a target format.
// Return ErrMediaNotConverted to decline; any other error stops the request.
// Implementations must be safe for concurrent use and must not modify media.
type MediaConverter interface {
	ConvertMedia(ctx context.Context, media MediaPart, accepts func(MediaPart) error) ([]ContentPart, error)
}

// MediaConverterFunc adapts a function to MediaConverter.
type MediaConverterFunc func(ctx context.Context, media MediaPart, accepts func(MediaPart) error) ([]ContentPart, error)

// ConvertMedia calls f.
func (f MediaConverterFunc) ConvertMedia(ctx context.Context, media MediaPart, accepts func(MediaPart) error) ([]ContentPart, error) {
	return f(ctx, media, accepts)
}

// MediaFallback chooses what happens to media that the model does not accept
// and no converter converted.
type MediaFallback uint8

const (
	// MediaFallbackError keeps the media, so mapping the request fails with
	// the adapter's *UnsupportedContentError. It is the default.
	MediaFallbackError MediaFallback = iota
	// MediaFallbackPlaceholder replaces the media with its placeholder text,
	// such as "[file: report.docx, application/vnd.openxmlformats-...]".
	MediaFallbackPlaceholder
)

// MediaPolicy controls media a model does not accept. The zero value converts
// nothing and leaves unsupported media to fail when the request is mapped.
type MediaPolicy struct {
	// Converter converts unsupported media. Nil converts nothing.
	Converter MediaConverter
	// Fallback applies when the media is not converted.
	Fallback MediaFallback
}

// MediaConversionError reports a converter failure, or converter output the
// model does not accept.
type MediaConversionError struct {
	// Media is the placeholder of the media being converted; it never holds
	// bytes or URIs.
	Media string
	Err   error
}

func (e *MediaConversionError) Error() string {
	return fmt.Sprintf("convert media %s: %v", e.Media, e.Err)
}

func (e *MediaConversionError) Unwrap() error { return e.Err }

// MediaAdapter applies a MediaPolicy for one model. It converts each distinct
// media part at most once and reuses the result, so a conversation rebuilt
// every round does not convert the same file again. It is safe for concurrent
// use.
type MediaAdapter struct {
	acceptor MediaAcceptor
	policy   MediaPolicy

	mu       sync.Mutex
	cache    map[[sha256.Size]byte][]ContentPart
	inflight map[[sha256.Size]byte]*mediaConversion
}

// mediaConversion is a conversion in progress, shared by concurrent callers
// that need the same media converted.
type mediaConversion struct {
	done  chan struct{}
	parts []ContentPart
	err   error
}

// NewMediaAdapter returns an adapter for model. If model is not a
// MediaAcceptor, the adapter changes nothing.
func NewMediaAdapter(model Model, policy MediaPolicy) *MediaAdapter {
	acceptor, _ := model.(MediaAcceptor)
	return &MediaAdapter{
		acceptor: acceptor, policy: policy,
		cache:    map[[sha256.Size]byte][]ContentPart{},
		inflight: map[[sha256.Size]byte]*mediaConversion{},
	}
}

// Messages returns messages whose media in user messages and tool results the
// model accepts, after conversion and fallback. Unchanged messages are
// returned as is; changed messages and parts are copies.
func (a *MediaAdapter) Messages(ctx context.Context, messages []Message) ([]Message, error) {
	if a == nil || a.acceptor == nil {
		return messages, nil
	}
	var out []Message
	for i, message := range messages {
		if message.Role != RoleUser && message.Role != RoleTool {
			continue
		}
		parts, changed, err := a.adaptParts(ctx, message.Parts, MediaInUserMessage)
		if err != nil {
			return nil, err
		}
		if !changed {
			continue
		}
		if out == nil {
			out = append([]Message(nil), messages...)
		}
		out[i].Parts = keepContent(parts, message.Parts)
	}
	if out == nil {
		return messages, nil
	}
	return out, nil
}

// Parts adapts parts found at placement, including media inside tool results.
func (a *MediaAdapter) Parts(ctx context.Context, parts []ContentPart, placement MediaPlacement) ([]ContentPart, error) {
	if a == nil || a.acceptor == nil {
		return parts, nil
	}
	adapted, changed, err := a.adaptParts(ctx, parts, placement)
	if err != nil || !changed || placement != MediaInUserMessage {
		return adapted, err
	}
	return keepContent(adapted, parts), nil
}

// keepContent returns adapted, or, when conversion removed every part of a
// message, the placeholders of its media, since a message needs content.
func keepContent(adapted, original []ContentPart) []ContentPart {
	if len(adapted) > 0 || len(original) == 0 {
		return adapted
	}
	var placeholders []string
	for _, part := range original {
		if part.Kind == ContentMedia && part.Media != nil {
			placeholders = append(placeholders, part.Media.Placeholder())
		}
	}
	return TextParts(strings.Join(placeholders, "\n"))
}

func (a *MediaAdapter) adaptParts(ctx context.Context, parts []ContentPart, placement MediaPlacement) ([]ContentPart, bool, error) {
	var out []ContentPart
	changed := false
	for i, part := range parts {
		var replacement []ContentPart
		switch {
		case part.Kind == ContentMedia && part.Media != nil:
			adapted, replaced, err := a.adaptMedia(ctx, *part.Media, placement, hasRequiredExtension(part.Extensions))
			if err != nil {
				return nil, false, err
			}
			if !replaced {
				break
			}
			// An empty replacement removes the part; keep a non-nil slice
			// so it still counts as a change.
			replacement = append([]ContentPart{}, adapted...)
		case part.Kind == ContentToolResult && part.ToolResult != nil:
			resultParts, resultChanged, err := a.adaptParts(ctx, part.ToolResult.Parts, MediaInToolResult)
			if err != nil {
				return nil, false, err
			}
			if !resultChanged {
				break
			}
			result := *part.ToolResult
			result.Parts = resultParts
			part.ToolResult = &result
			replacement = []ContentPart{part}
		}
		if replacement == nil {
			if changed {
				out = append(out, part)
			}
			continue
		}
		if !changed {
			changed = true
			out = append([]ContentPart(nil), parts[:i]...)
		}
		out = append(out, replacement...)
	}
	if !changed {
		return parts, false, nil
	}
	return out, true, nil
}

// adaptMedia reports replaced false when the model accepts media; otherwise
// it returns the replacement parts, which may be empty. Concurrent callers
// with the same media share one conversion. Media whose part carries a
// required extension is never replaced, since the extension would be lost.
func (a *MediaAdapter) adaptMedia(ctx context.Context, media MediaPart, placement MediaPlacement, requiredExtension bool) (parts []ContentPart, replaced bool, err error) {
	rejection := a.acceptor.AcceptsMedia(media, placement)
	if rejection == nil {
		return nil, false, nil
	}
	if requiredExtension {
		return nil, false, &UnsupportedContentError{Provider: "media", Kind: ContentExtension, Detail: "media with a required extension cannot be converted: " + rejection.Error()}
	}
	key := mediaKey(media, placement)
	for {
		a.mu.Lock()
		if cached, ok := a.cache[key]; ok {
			a.mu.Unlock()
			return CloneParts(cached), true, nil
		}
		if running, ok := a.inflight[key]; ok {
			a.mu.Unlock()
			select {
			case <-running.done:
			case <-ctx.Done():
				return nil, false, ctx.Err()
			}
			// A conversion canceled by its own caller is retried by
			// waiters whose context is still live.
			if running.err != nil && isContextError(running.err) && ctx.Err() == nil {
				continue
			}
			if running.err != nil {
				return nil, false, running.err
			}
			return CloneParts(running.parts), true, nil
		}
		conversion := &mediaConversion{done: make(chan struct{})}
		a.inflight[key] = conversion
		a.mu.Unlock()

		conversion.parts, conversion.err = a.convert(ctx, media, placement, rejection)
		a.mu.Lock()
		delete(a.inflight, key)
		if conversion.err == nil {
			a.cache[key] = CloneParts(conversion.parts)
		}
		a.mu.Unlock()
		close(conversion.done)
		if conversion.err != nil {
			return nil, false, conversion.err
		}
		return CloneParts(conversion.parts), true, nil
	}
}

func isContextError(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

func hasRequiredExtension(extensions []Extension) bool {
	for _, ext := range extensions {
		if ext.Required {
			return true
		}
	}
	return false
}

func (a *MediaAdapter) convert(ctx context.Context, media MediaPart, placement MediaPlacement, rejection error) ([]ContentPart, error) {
	if a.policy.Converter != nil {
		accepts := func(candidate MediaPart) error { return a.acceptor.AcceptsMedia(candidate, placement) }
		parts, err := a.policy.Converter.ConvertMedia(ctx, media, accepts)
		switch {
		case err == nil:
			if err := a.checkConverted(parts, placement); err != nil {
				return nil, &MediaConversionError{Media: media.Placeholder(), Err: err}
			}
			return CloneParts(parts), nil
		case !errors.Is(err, ErrMediaNotConverted):
			return nil, &MediaConversionError{Media: media.Placeholder(), Err: err}
		}
	}
	if a.policy.Fallback == MediaFallbackPlaceholder {
		return []ContentPart{{Kind: ContentText, Text: media.Placeholder()}}, nil
	}
	return nil, rejection
}

// checkConverted requires text, JSON, or media the model accepts.
func (a *MediaAdapter) checkConverted(parts []ContentPart, placement MediaPlacement) error {
	for _, part := range parts {
		if err := part.Validate(); err != nil {
			return err
		}
		switch part.Kind {
		case ContentText, ContentJSON:
		case ContentMedia:
			if err := a.acceptor.AcceptsMedia(*part.Media, placement); err != nil {
				return fmt.Errorf("converter returned media the model does not accept: %w", err)
			}
		default:
			return fmt.Errorf("converter returned %q content; only text, JSON and media are allowed", part.Kind)
		}
	}
	return nil
}

func mediaKey(media MediaPart, placement MediaPlacement) [sha256.Size]byte {
	h := sha256.New()
	var n [8]byte
	for _, field := range [][]byte{[]byte(media.MIMEType), []byte(media.URI), []byte(media.Name), media.Data} {
		binary.LittleEndian.PutUint64(n[:], uint64(len(field)))
		h.Write(n[:])
		h.Write(field)
	}
	h.Write([]byte{byte(placement)})
	var key [sha256.Size]byte
	h.Sum(key[:0])
	return key
}
