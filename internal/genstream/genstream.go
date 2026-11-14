// Package genstream connects a provider's raw token stream to text-encoded
// tool-call detection while recording the stream callers receive, so a
// generation span can capture the output callers see rather than the raw text.
package genstream

import (
	"context"
	"strings"
	"sync"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/internal/appclock"
)

// Stream runs ai.DetectToolCallsInStream over a provider's raw tokens and
// forwards the detected tokens to the caller. When the context's
// ContentCapturePolicy enables completion capture, it also records their text
// and tool-call identities; tool arguments are never recorded.
type Stream struct {
	out       chan ai.Token
	drained   chan struct{}
	finished  chan struct{}
	closeOnce sync.Once
	capture   bool
	text      strings.Builder
	calls     []ai.ContentPart
}

// Detect starts detection over raw. The provider must close raw when its
// stream ends and then call Close, after finishing its generation observation,
// so the caller sees the stream end only after the generation span ended.
func Detect(ctx context.Context, raw <-chan ai.Token, debug gai.ObservationSink) *Stream {
	policy, ok := gai.ContentCapturePolicyFromContext(ctx)
	s := &Stream{
		out:      make(chan ai.Token, 8),
		drained:  make(chan struct{}),
		finished: make(chan struct{}),
		capture:  ok && policy.Completion == gai.CaptureEnabled,
	}
	detected := ai.DetectToolCallsInStream(ctx, raw, debug)
	go func() {
		defer close(s.out)
		sending := true
		for token := range detected {
			s.record(token)
			if sending && !ai.SendToken(ctx, s.out, token) {
				// Keep draining so detection, and the provider behind it,
				// can finish.
				sending = false
			}
		}
		close(s.drained)
		<-s.finished
	}()
	return s
}

// Close lets the caller's stream end. Call it once the generation observation
// has finished. It is safe to call more than once.
func (s *Stream) Close() {
	s.closeOnce.Do(func() { close(s.finished) })
}

// Tokens returns the detected stream for the caller.
func (s *Stream) Tokens() <-chan ai.Token { return s.out }

// Output returns the caller-visible output for GenerationResult.Output, or nil
// when completion capture is off. Call it after closing raw: it waits until
// detection has ended and the whole stream was passed to the caller, so
// detection's observations are emitted before the generation's. The wait
// pauses ctx's application clock, so it does not count as inference time.
func (s *Stream) Output(ctx context.Context) *ai.Message {
	resume := appclock.Pause(ctx)
	<-s.drained
	resume()
	if !s.capture {
		return nil
	}
	message := &ai.Message{Role: ai.RoleAssistant}
	if s.text.Len() > 0 {
		message.Parts = append(message.Parts, ai.ContentPart{Kind: ai.ContentText, Text: s.text.String()})
	}
	message.Parts = append(message.Parts, s.calls...)
	return message
}

func (s *Stream) record(token ai.Token) {
	if !s.capture {
		return
	}
	switch token.Type() {
	case ai.TokenTypeText:
		s.text.WriteString(token.Text())
	case ai.TokenTypeToolCall:
		if call := token.ToolCall(); call != nil {
			s.calls = append(s.calls, ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: call.ID, Name: call.Name}})
		}
	}
}
