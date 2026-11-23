package history

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput"
)

// prefixLayout records where stored history appears in a continued request.
type prefixLayout struct {
	// summary is the index of the summary message, or -1 without a summary.
	summary int
	// turnEnds holds the exclusive end index of each folded turn's messages.
	turnEnds []int
	// firstKeptUser is the index of the first kept turn's user message, or -1
	// when there is no kept turn or it has no user message in the prefix.
	firstKeptUser int
}

// locateHistory finds the summary and each folded turn in messages, in order.
// A turn's messages must be contiguous; other messages may sit between turns,
// such as context added after the history source. Stored messages match their
// projected form too, so older tool results may appear as previews.
func locateHistory(messages []ai.Message, previous *Summary, turns []modelinput.Turn, firstKept *modelinput.Turn) (prefixLayout, error) {
	layout := prefixLayout{summary: -1, firstKeptUser: -1}
	cursor := 0
	if previous != nil {
		want := summaryMessage(previous)
		found := -1
		for i := cursor; i < len(messages); i++ {
			if equalMessage(messages[i], want) {
				found = i
				break
			}
		}
		if found < 0 {
			return layout, fmt.Errorf("%w: summary %q not found", ErrSummaryPrefixMismatch, previous.ID)
		}
		layout.summary = found
		cursor = found + 1
	}
	for i := range turns {
		block := turnMessages(&turns[i])
		start := findBlock(messages, cursor, block)
		if start < 0 {
			return layout, fmt.Errorf("%w: turn %q not found", ErrSummaryPrefixMismatch, turns[i].ID)
		}
		cursor = start + len(block)
		layout.turnEnds = append(layout.turnEnds, cursor)
	}
	if firstKept != nil && firstKept.UserMessage != nil {
		for i := cursor; i < len(messages); i++ {
			if equivalentMessage(messages[i], firstKept.UserMessage.Message) {
				layout.firstKeptUser = i
				break
			}
		}
	}
	return layout, nil
}

func turnMessages(turn *modelinput.Turn) []ai.Message {
	var messages []ai.Message
	if turn.UserMessage != nil {
		messages = append(messages, turn.UserMessage.Message)
	}
	for _, message := range turn.Messages {
		messages = append(messages, message.Message)
	}
	return messages
}

func findBlock(messages []ai.Message, from int, block []ai.Message) int {
	if len(block) == 0 {
		return from
	}
search:
	for start := from; start+len(block) <= len(messages); start++ {
		for j := range block {
			if !equivalentMessage(messages[start+j], block[j]) {
				continue search
			}
		}
		return start
	}
	return -1
}

// equivalentMessage reports whether a request message carries the stored
// message, either unchanged or in its history preview projection.
func equivalentMessage(requestMessage, stored ai.Message) bool {
	if equalMessage(requestMessage, stored) {
		return true
	}
	preview := (&Part{Messages: []ai.Message{stored}}).ConversationMessages()
	return len(preview) == 1 && equalMessage(requestMessage, preview[0])
}

// equalMessage compares canonical messages by meaning. JSON payloads compare as
// decoded values, so a store round trip that compacts whitespace, escapes HTML
// or reorders keys still matches. Nil and empty slices are equal.
func equalMessage(a, b ai.Message) bool {
	return a.Role == b.Role && equalParts(a.Parts, b.Parts) && equalExtensions(a.Extensions, b.Extensions)
}

func equalParts(a, b []ai.ContentPart) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !equalPart(a[i], b[i]) {
			return false
		}
	}
	return true
}

func equalPart(a, b ai.ContentPart) bool {
	if a.Kind != b.Kind || a.Text != b.Text || !equalJSON(a.JSON, b.JSON) || !equalExtensions(a.Extensions, b.Extensions) {
		return false
	}
	if (a.ToolCall == nil) != (b.ToolCall == nil) || (a.ToolResult == nil) != (b.ToolResult == nil) || (a.Media == nil) != (b.Media == nil) {
		return false
	}
	if a.ToolCall != nil {
		x, y := a.ToolCall, b.ToolCall
		if x.ID != y.ID || x.Type != y.Type || x.Name != y.Name || !equalJSON(x.Args, y.Args) || !equalExtensions(x.Extensions, y.Extensions) {
			return false
		}
	}
	if a.ToolResult != nil {
		x, y := a.ToolResult, b.ToolResult
		if x.ToolCallID != y.ToolCallID || x.Name != y.Name || x.IsError != y.IsError || !equalParts(x.Parts, y.Parts) {
			return false
		}
	}
	if a.Media != nil {
		x, y := a.Media, b.Media
		if x.MIMEType != y.MIMEType || x.URI != y.URI || !bytes.Equal(x.Data, y.Data) {
			return false
		}
	}
	return true
}

func equalExtensions(a, b []ai.Extension) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Namespace != b[i].Namespace || a[i].Type != b[i].Type || a[i].Required != b[i].Required || !equalJSON(a[i].Data, b[i].Data) {
			return false
		}
	}
	return true
}

func equalJSON(a, b json.RawMessage) bool {
	if bytes.Equal(a, b) {
		return true
	}
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	left, ok := decodeJSON(a)
	if !ok {
		return false
	}
	right, ok := decodeJSON(b)
	return ok && reflect.DeepEqual(left, right)
}

func decodeJSON(data json.RawMessage) (any, bool) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, false
	}
	if decoder.More() {
		return nil, false
	}
	return value, true
}
