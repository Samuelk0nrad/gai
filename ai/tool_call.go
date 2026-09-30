package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/lace-ai/gai"
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
	// ThoughtSignature is opaque provider state that must accompany a tool call
	// in a subsequent provider-native history request.
	ThoughtSignature []byte
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
		ctx, observer := newToolCallStreamObserver(ctx, debug)
		defer close(out)

		var pending []Token
		result := toolCallStreamResult{}
		canceled := false
		defer func() {
			// Cancellation observable at terminal finalization takes precedence
			// over a concurrently closed input channel.
			if canceled || ctx.Err() != nil {
				observer.Canceled(ctx, result)
				return
			}
			observer.Finished(ctx, result)
		}()

		// JSON tracking state
		seenNonWS := false
		newLines := 0
		isJSONCandidate := false
		objDepth := 0
		arrDepth := 0
		inString := false
		escape := false

		resetTracking := func() {
			seenNonWS = false
			newLines = 0
			isJSONCandidate = false
			objDepth = 0
			arrDepth = 0
			inString = false
			escape = false
		}

		flushPending := func() bool {
			for _, t := range pending {
				if !SendToken(ctx, out, t) {
					canceled = true
					return false
				}
				result.outputTokenEvents++
			}
			resetTracking()
			pending = nil
			return true
		}

		flushBeforeCandidate := func(current []byte, idx int) bool {
			if len(pending) == 0 {
				return true
			}

			if idx > 0 {
				pending[len(pending)-1].Data = current[:idx]
			} else {
				pending = pending[:len(pending)-1]
			}
			if len(bytes.TrimSpace(joinTokenData(pending))) == 0 {
				resetTracking()
				pending = nil
				return true
			}
			return flushPending()
		}

		maybeToolCall := func(last string) (handled bool, keepGoing bool) {
			if !isJSONCandidate {
				return false, true
			}
			if inString || objDepth != 0 || arrDepth != 0 {
				observer.CandidateRejected(&result, fmt.Sprintf("inString=%v objDepth=%d arrDepth=%d", inString, objDepth, arrDepth), joinTokenData(pending))
				return false, true
			}

			payload := []byte(last)
			if len(pending) > 0 {
				payload = append(joinTokenData(pending[:len(pending)-1]), payload...)
			}
			if tc, ok := parseToolCall(payload); ok {
				detected := observer.snapshotDetected(tc)
				if !SendToken(ctx, out, Token{
					Type:     TokenTypeToolCall,
					Data:     payload,
					ToolCall: tc,
				}) {
					canceled = true
					return false, false
				}
				result.outputTokenEvents++
				observer.Detected(&result, detected)
				resetTracking()
			} else {
				observer.CandidateRejected(&result, "parse_failed", payload)
				if !flushPending() {
					return false, false
				}
			}

			pending = nil
			return true, true
		}

		for {
			var t Token
			var ok bool
			select {
			case <-ctx.Done():
				if isJSONCandidate {
					pendingPayload := joinTokenData(pending)
					observer.Pending(&result, pendingPayload)
					result.eofPending = true
					observer.CandidateRejected(&result, "stream_canceled", pendingPayload)
				}
				canceled = true
				return
			case t, ok = <-in:
				if !ok {
					goto streamDone
				}
			}
			result.inputTokenEvents++
			// non-text tokens: passthrough.
			if t.Type != TokenTypeText {
				if t.Type == TokenTypeCompletion && isJSONCandidate {
					pendingPayload := joinTokenData(pending)
					observer.Pending(&result, pendingPayload)
					result.eofPending = true
					observer.CandidateRejected(&result, "end_of_stream", pendingPayload)
				} else if t.Type == TokenTypeErr && isJSONCandidate {
					observer.CandidateRejected(&result, "stream_error", joinTokenData(pending))
				} else if isJSONCandidate {
					observer.CandidateRejected(&result, "interrupted", joinTokenData(pending))
				}
				pending = append(pending, t)
				if !flushPending() {
					return
				}
				continue
			}

			remaining := t.Data
			for len(remaining) > 0 {
				pending = append(pending, Token{Type: TokenTypeText, Data: remaining})

				var tokenStr strings.Builder
				handledCandidate := false
				for idx, b := range remaining {
					tokenStr.WriteByte(b)

					if !seenNonWS && !isJSONCandidate {
						if isWS(b) {
							continue
						}
						seenNonWS = true
						if b == '{' {
							isJSONCandidate = true
							objDepth = 1
						}
						continue
					}

					if !isJSONCandidate {
						if b == '\n' {
							newLines++
						}
						if newLines >= 2 && b == '{' {
							if !flushBeforeCandidate(remaining, idx) {
								return
							}
							pending = append(pending, Token{Type: TokenTypeText, Data: remaining[idx:]})
							tokenStr.Reset()
							tokenStr.WriteByte(b)
							isJSONCandidate = true
							objDepth = 1
							seenNonWS = true
							newLines = 0
						}
						continue
					}

					if inString {
						if escape {
							escape = false
							continue
						}
						if b == '\\' {
							escape = true
							continue
						}
						if b == '"' {
							inString = false
						}
						continue
					}

					switch b {
					case '"':
						inString = true
					case '{':
						objDepth++
					case '}':
						objDepth--
					case '[':
						arrDepth++
					case ']':
						arrDepth--
					}

					// If JSON candidate is balanced at this byte, decide now.
					if isJSONCandidate && !inString && objDepth == 0 && arrDepth == 0 {
						handled, keepGoing := maybeToolCall(tokenStr.String())
						if !keepGoing {
							return
						}
						if handled {
							handledCandidate = true
							if idx+1 < len(remaining) {
								remaining = append([]byte(nil), remaining[idx+1:]...)
								seenNonWS = false
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
		}

	streamDone:
		// End of stream: an unresolved JSON candidate is rejected, then all
		// buffered tokens are replayed unchanged.
		if len(pending) > 0 {
			pendingPayload := joinTokenData(pending)
			observer.Pending(&result, pendingPayload)
			if isJSONCandidate {
				result.eofPending = true
				observer.CandidateRejected(&result, "end_of_stream", pendingPayload)
			}
			if !flushPending() {
				return
			}
		}
	}()

	return out
}

func joinTokenData(tokens []Token) []byte {
	var b bytes.Buffer
	for _, t := range tokens {
		b.Write(t.Data)
	}
	return b.Bytes()
}

func isWS(b byte) bool {
	return b == ' ' || b == '\n' || b == '\r' || b == '\t'
}
