// Package streamtext follows how ai.Message.AppendToken merges streamed text,
// so a provider can place citations on the text part a caller builds.
package streamtext

import (
	"encoding/json"

	"github.com/Samuelk0nrad/gai/ai"
)

// Run tracks the text part that ai.Message.AppendToken is building from the
// tokens a stream has emitted. Observe every emitted token in order.
type Run struct {
	// open is true while the last part is text without extensions, so the
	// next text token without extensions is merged into it.
	open  bool
	runes int
}

// Len returns the rune length of the open text part, or 0 when the next text
// token starts a new part.
func (r *Run) Len() int {
	if !r.open {
		return 0
	}
	return r.runes
}

// Observe applies one emitted token, using the merge rule of AppendToken.
func (r *Run) Observe(t ai.Token) {
	p := t.Part
	if p == nil {
		return
	}
	if p.Kind != ai.ContentText {
		r.open, r.runes = false, 0
		return
	}
	if len(p.Extensions) > 0 {
		// An empty text token with extensions attaches them to the open part,
		// and any other text with extensions starts its own part. Either way
		// the last part now has extensions and is closed to merging.
		r.open, r.runes = false, 0
		return
	}
	n := ai.RuneLen(p.Text)
	if r.open {
		r.runes += n
		return
	}
	r.open, r.runes = true, n
}

// AttachCitations returns an empty text token that attaches citations to the
// open text part. The citations' offsets are relative to the last span runes
// of that part, the text they belong to, and are shifted to be relative to
// the whole part. Call it right before emitting the token.
func (r *Run) AttachCitations(namespace string, citations []ai.Citation, span int) (ai.Token, bool) {
	if len(citations) == 0 {
		return ai.Token{}, false
	}
	// When the part holds less than the cited span, the cited text did not
	// all reach it, for example because a text tool-call detector turned part
	// of it into a tool call. The offsets would then point past the text, so
	// they are dropped; Citation.Text still names the cited text.
	shift := r.Len() - span
	shifted := make([]ai.Citation, len(citations))
	for i, c := range citations {
		if shift < 0 || c.End > span {
			c.Start, c.End = 0, 0
		} else if c.End > c.Start {
			c.Start += shift
			c.End += shift
		}
		shifted[i] = c
	}
	return ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Extensions: ai.CitationsExtension(namespace, shifted)}}, true
}

// SliceRunes returns runes [start, end) of s, or "" when the range is invalid.
func SliceRunes(s string, start, end int) string {
	if start < 0 || end <= start {
		return ""
	}
	i, from, to := 0, -1, -1
	for b := range s {
		if i == start {
			from = b
		}
		if i == end {
			to = b
			break
		}
		i++
	}
	if from < 0 {
		return ""
	}
	if to < 0 {
		if end > i {
			return ""
		}
		to = len(s)
	}
	return s[from:to]
}

// RawObject returns raw when it is a valid JSON value, else nil.
func RawObject(raw string) json.RawMessage {
	if raw == "" || !json.Valid([]byte(raw)) {
		return nil
	}
	return append(json.RawMessage(nil), raw...)
}
