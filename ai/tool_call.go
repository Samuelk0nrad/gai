package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/internal/appclock"
)

// ToolCall describes a model request to invoke a function tool.
type ToolCall struct {
	// ID identifies the call within the model interaction.
	ID string
	// Type is the tool-call kind and must be "function".
	Type string
	// Name is the function tool to invoke.
	Name string
	// Args contains the function arguments as JSON.
	Args json.RawMessage
	// Extensions retains provider continuity state on this exact call.
	Extensions []Extension
}

// Clone snapshots a call and its scoped continuity state.
func (tc ToolCall) Clone() ToolCall {
	tc.Args = append(json.RawMessage(nil), tc.Args...)
	tc.Extensions = CloneExtensions(tc.Extensions)
	return tc
}

// Validate checks that the tool call has an ID, the "function" type, and a
// non-empty name.
func (tc *ToolCall) Validate() error {
	if tc == nil {
		return fmt.Errorf("%w: tool call nil", ErrInvalidToolCall)
	}
	if strings.TrimSpace(tc.ID) == "" {
		return fmt.Errorf("%w: id empty", ErrInvalidToolCall)
	}
	if tc.Type != "function" {
		return fmt.Errorf("%w: type not function", ErrInvalidToolCall)
	}
	if strings.TrimSpace(tc.Name) == "" {
		return fmt.Errorf("%w: name empty", ErrInvalidToolCall)
	}
	return nil
}

// String returns a diagnostic representation of the tool call.
func (tc *ToolCall) String() string {
	if tc == nil {
		return "<nil>"
	}
	var builder strings.Builder
	builder.WriteString("id: ")
	builder.WriteString(tc.ID)
	builder.WriteString(",type: ")
	builder.WriteString(tc.Type)
	builder.WriteString(",name: ")
	builder.WriteString(tc.Name)
	builder.WriteString(",arguments: ")
	builder.Write(tc.Args)

	return builder.String()
}

var toolCallCounter uint64

// GenerateToolCallID creates a process-local unique call ID using toolName as
// a readable component.
func GenerateToolCallID(toolName string) string {
	name := strings.TrimSpace(toolName)
	if name == "" {
		name = "tool"
	}
	seq := atomic.AddUint64(&toolCallCounter, 1)
	return fmt.Sprintf("call_%s_%d_%d", name, time.Now().UnixNano(), seq)
}

func parseToolCall(payload []byte) (*ToolCall, bool) {
	s := strings.TrimSpace(string(payload))
	if s == "" || !strings.HasPrefix(s, "{") {
		return nil, false
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(s), &raw); err != nil {
		return nil, false
	}

	var typ, name string
	var args json.RawMessage

	if v, ok := raw["type"]; ok {
		if err := json.Unmarshal(v, &typ); err != nil {
			return nil, false
		}
	}
	if v, ok := raw["name"]; ok {
		if err := json.Unmarshal(v, &name); err != nil {
			return nil, false
		}
	}
	if v, ok := raw["arguments"]; ok {
		args = v
	}

	if typ != "function" {
		return nil, false
	}
	if strings.TrimSpace(name) == "" {
		return nil, false
	}
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}

	return &ToolCall{
		ID:   GenerateToolCallID(name),
		Type: typ,
		Name: name,
		Args: args,
	}, true
}

// DetectToolCallsInStream scans a token stream for text-encoded tool calls.
// When it detects a valid tool-call JSON object, it emits a ToolCall token.
// Otherwise, buffered tokens are replayed unchanged.
func DetectToolCallsInStream(ctx context.Context, in <-chan Token, debug gai.ObservationSink) <-chan Token {
	out := make(chan Token, 8)

	go func() {
		defer close(out)
		var detector *ToolCallDetector
		detector = NewToolCallDetector(ctx, debug, func(token Token) bool {
			return SendToken(detector.ctx, out, token)
		})
		defer detector.Close()

		for {
			select {
			case <-detector.ctx.Done():
				detector.cancel()
				return
			case t, ok := <-in:
				if !ok {
					return
				}
				if !detector.Push(t) {
					return
				}
			}
		}
	}()

	return out
}

// ToolCallDetector detects text-encoded tool calls in tokens pushed to it, in
// the pushing goroutine. Providers use it to detect calls inline, so the
// output they deliver, and observe, is the detected stream.
// DetectToolCallsInStream runs the same detection on its own goroutine.
type ToolCallDetector struct {
	ctx      context.Context
	observer *toolCallStreamObserver
	send     func(Token) bool
	pending  []Token
	result   toolCallStreamResult
	canceled bool
	stopped  bool
	closed   bool
	err      error
	// clockCtx carries the generation clock that detection pauses, so its
	// work and observations count as application time, not inference time.
	clockCtx context.Context

	// JSON tracking state
	seenNonWS       bool
	newLines        int
	isJSONCandidate bool
	objDepth        int
	arrDepth        int
	inString        bool
	escape          bool
}

// NewToolCallDetector starts a detection observation under ctx and returns a
// detector that passes its output to send. send reports whether the token was
// delivered; once it returns false, detection stops.
func NewToolCallDetector(ctx context.Context, debug gai.ObservationSink, send func(Token) bool) *ToolCallDetector {
	ctx, observer := newToolCallStreamObserver(ctx, debug)
	return &ToolCallDetector{ctx: ctx, observer: observer, send: send}
}

// PauseIn makes later Push and Close calls pause the generation clock in
// ctx, the context StartGenerationObservation returned. Detection and its
// observation sink calls are then excluded from inference duration, as token
// sends already are.
func (d *ToolCallDetector) PauseIn(ctx context.Context) {
	d.clockCtx = ctx
}

func (d *ToolCallDetector) pause() (resume func()) {
	if d.clockCtx == nil {
		return func() {}
	}
	return appclock.Pause(d.clockCtx)
}

// Push processes one input token and passes any resulting tokens to send. It
// returns false when detection has stopped: a token could not be delivered or
// an invalid token ended the stream. Further pushes are ignored.
func (d *ToolCallDetector) Push(t Token) bool {
	if d.stopped || d.closed {
		return false
	}
	defer d.pause()()
	if !d.push(t) {
		d.stopped = true
		return false
	}
	return true
}

// Err returns the validation error that stopped detection, if any.
func (d *ToolCallDetector) Err() error { return d.err }

// Close ends the input. Unless detection was canceled, buffered tokens are
// replayed and an unfinished candidate is rejected. It then finishes the
// detection observation. It is safe to call more than once.
func (d *ToolCallDetector) Close() {
	if d.closed {
		return
	}
	d.closed = true
	defer d.pause()()
	if !d.stopped {
		d.end()
	}
	// Cancellation observable at terminal finalization takes precedence over
	// a concurrently closed input.
	if d.canceled || d.ctx.Err() != nil {
		d.observer.Canceled(d.ctx, d.result)
		return
	}
	d.observer.Finished(d.ctx, d.result)
}

// cancel records that the context ended before the input did: the buffered
// candidate is rejected and nothing more is delivered.
func (d *ToolCallDetector) cancel() {
	if d.isJSONCandidate {
		pendingPayload := joinTokenText(d.pending)
		d.observer.Pending(&d.result, pendingPayload)
		d.result.eofPending = true
		d.observer.CandidateRejected(&d.result, "stream_canceled", pendingPayload)
	}
	d.canceled = true
	d.stopped = true
}

func (d *ToolCallDetector) deliver(t Token) bool {
	if !d.send(t) {
		d.canceled = true
		return false
	}
	d.result.outputTokenEvents++
	return true
}

func (d *ToolCallDetector) resetTracking() {
	d.seenNonWS = false
	d.newLines = 0
	d.isJSONCandidate = false
	d.objDepth = 0
	d.arrDepth = 0
	d.inString = false
	d.escape = false
}

func (d *ToolCallDetector) flushPending() bool {
	for _, t := range d.pending {
		if !d.deliver(t) {
			return false
		}
	}
	d.resetTracking()
	d.pending = nil
	return true
}

func (d *ToolCallDetector) flushBeforeCandidate(current []byte, idx int) (int, bool) {
	carryUsage := 0
	if idx > 0 {
		d.pending[len(d.pending)-1].Part.Text = string(current[:idx])
	} else {
		carryUsage = d.pending[len(d.pending)-1].TokenUsage
		d.pending = d.pending[:len(d.pending)-1]
	}
	if len(bytes.TrimSpace(joinTokenText(d.pending))) == 0 {
		carryUsage += sumTokenUsage(d.pending)
		d.resetTracking()
		d.pending = nil
		return carryUsage, true
	}
	return carryUsage, d.flushPending()
}

func (d *ToolCallDetector) maybeToolCall(last string) (handled bool, keepGoing bool) {
	if !d.isJSONCandidate {
		return false, true
	}
	if d.inString || d.objDepth != 0 || d.arrDepth != 0 {
		d.observer.CandidateRejected(&d.result, fmt.Sprintf("inString=%v objDepth=%d arrDepth=%d", d.inString, d.objDepth, d.arrDepth), joinTokenText(d.pending))
		return false, true
	}

	payload := []byte(last)
	if len(d.pending) > 0 {
		payload = append(joinTokenText(d.pending[:len(d.pending)-1]), payload...)
	}
	if tc, ok := parseToolCall(payload); ok {
		detected := d.observer.snapshotDetected(tc)
		if !d.deliver(Token{
			Part:       &ContentPart{Kind: ContentToolCall, ToolCall: tc},
			TokenUsage: sumTokenUsage(d.pending),
		}) {
			return false, false
		}
		d.observer.Detected(&d.result, detected)
		d.resetTracking()
	} else {
		d.observer.CandidateRejected(&d.result, "parse_failed", payload)
		// Replay only the balanced candidate; the remaining suffix is
		// handled separately without duplicating text or usage.
		d.pending[len(d.pending)-1].Part.Text = last
		if !d.flushPending() {
			return false, false
		}
	}

	d.pending = nil
	return true, true
}

// push processes one token and reports whether detection continues.
func (d *ToolCallDetector) push(t Token) bool {
	d.result.inputTokenEvents++
	if err := t.Validate(); err != nil {
		if d.isJSONCandidate {
			d.observer.CandidateRejected(&d.result, "invalid_event", joinTokenText(d.pending))
		}
		if !d.flushPending() {
			return false
		}
		d.err = err
		d.deliver(Token{Err: err})
		return false
	}
	// Only unsigned text participates in the text tool protocol. Other
	// canonical parts and execution metadata pass through intact.
	t = t.Clone()
	if t.Part == nil || t.Part.Kind != ContentText || len(t.Part.Extensions) > 0 {
		if t.Type() == TokenTypeCompletion && d.isJSONCandidate {
			pendingPayload := joinTokenText(d.pending)
			d.observer.Pending(&d.result, pendingPayload)
			d.result.eofPending = true
			d.observer.CandidateRejected(&d.result, "end_of_stream", pendingPayload)
		} else if t.Type() == TokenTypeErr && d.isJSONCandidate {
			d.observer.CandidateRejected(&d.result, "stream_error", joinTokenText(d.pending))
		} else if d.isJSONCandidate {
			d.observer.CandidateRejected(&d.result, "interrupted", joinTokenText(d.pending))
		}
		d.pending = append(d.pending, t)
		return d.flushPending()
	}

	remaining := []byte(t.Part.Text)
	remainingUsage := t.TokenUsage
	if len(remaining) == 0 {
		d.pending = append(d.pending, t)
	}
	for len(remaining) > 0 {
		d.pending = append(d.pending, Token{Part: &ContentPart{Kind: ContentText, Text: string(remaining)}, TokenUsage: remainingUsage})
		remainingUsage = 0

		var tokenStr strings.Builder
		handledCandidate := false
		for idx, b := range remaining {
			tokenStr.WriteByte(b)

			if !d.seenNonWS && !d.isJSONCandidate {
				if isWS(b) {
					continue
				}
				d.seenNonWS = true
				if b == '{' {
					d.isJSONCandidate = true
					d.objDepth = 1
				}
				continue
			}

			if !d.isJSONCandidate {
				if b == '\n' {
					d.newLines++
				}
				if d.newLines >= 2 && b == '{' {
					candidateUsage, ok := d.flushBeforeCandidate(remaining, idx)
					if !ok {
						return false
					}
					d.pending = append(d.pending, Token{Part: &ContentPart{Kind: ContentText, Text: string(remaining[idx:])}, TokenUsage: candidateUsage})
					tokenStr.Reset()
					tokenStr.WriteByte(b)
					d.isJSONCandidate = true
					d.objDepth = 1
					d.seenNonWS = true
					d.newLines = 0
				}
				continue
			}

			if d.inString {
				if d.escape {
					d.escape = false
					continue
				}
				if b == '\\' {
					d.escape = true
					continue
				}
				if b == '"' {
					d.inString = false
				}
				continue
			}

			switch b {
			case '"':
				d.inString = true
			case '{':
				d.objDepth++
			case '}':
				d.objDepth--
			case '[':
				d.arrDepth++
			case ']':
				d.arrDepth--
			}

			// If JSON candidate is balanced at this byte, decide now.
			if d.isJSONCandidate && !d.inString && d.objDepth == 0 && d.arrDepth == 0 {
				handled, keepGoing := d.maybeToolCall(tokenStr.String())
				if !keepGoing {
					return false
				}
				if handled {
					handledCandidate = true
					if idx+1 < len(remaining) {
						remaining = append([]byte(nil), remaining[idx+1:]...)
						d.seenNonWS = false
					} else {
						remaining = nil
					}
					break
				}
			}
		}

		if !handledCandidate {
			break
		}
	}
	return true
}

// end handles the end of input: an unresolved JSON candidate is rejected,
// then all buffered tokens are replayed unchanged.
func (d *ToolCallDetector) end() {
	if len(d.pending) > 0 {
		pendingPayload := joinTokenText(d.pending)
		d.observer.Pending(&d.result, pendingPayload)
		if d.isJSONCandidate {
			d.result.eofPending = true
			d.observer.CandidateRejected(&d.result, "end_of_stream", pendingPayload)
		}
		d.flushPending()
	}
}

func joinTokenText(tokens []Token) []byte {
	var b bytes.Buffer
	for _, t := range tokens {
		b.WriteString(t.Text())
	}
	return b.Bytes()
}

func isWS(b byte) bool {
	return b == ' ' || b == '\n' || b == '\r' || b == '\t'
}

func sumTokenUsage(tokens []Token) int {
	total := 0
	for _, token := range tokens {
		total += token.TokenUsage
	}
	return total
}
