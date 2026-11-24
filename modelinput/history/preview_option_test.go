package history_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput"
	"github.com/Samuelk0nrad/gai/modelinput/history"
)

const truncatedMarker = "\n[tool result truncated]"

// toolResultTurn is a turn whose tool call returned resultText.
func toolResultTurn(n int, resultText string) modelinput.Turn {
	id := fmt.Sprint("t", n)
	user := stored(ai.TextMessage(ai.RoleUser, fmt.Sprint("Question ", n, ": list the open tasks.")))
	return modelinput.Turn{ID: id, Count: n, UserMessage: &user, Messages: []modelinput.StoredMessage{
		stored(ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "c" + id, Type: "function", Name: "search", Args: json.RawMessage(`{}`)}}}}),
		stored(ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "c" + id, Name: "search", Parts: ai.TextParts(resultText)}}}}),
		stored(ai.TextMessage(ai.RoleAssistant, fmt.Sprint("Answer ", n))),
	}}
}

func toolResultState(results ...string) *history.HistoryState {
	state := &history.HistoryState{SchemaVersion: history.HistorySchemaVersion}
	for i, result := range results {
		state.Turns = append(state.Turns, toolResultTurn(i+1, result))
	}
	return state
}

func TestHistorySourceToolResultPreviewOption(t *testing.T) {
	t.Parallel()
	full := strings.Repeat("界", 600)
	for _, tt := range []struct {
		name string
		opts []history.Option
		want string
	}{
		{name: "default", want: strings.Repeat("界", 500) + truncatedMarker},
		{name: "custom", opts: []history.Option{history.WithToolResultPreview(40)}, want: strings.Repeat("界", 40) + truncatedMarker},
		{name: "larger than result", opts: []history.Option{history.WithToolResultPreview(1000)}, want: full},
		{name: "zero keeps whole", opts: []history.Option{history.WithToolResultPreview(0)}, want: full},
		{name: "negative keeps whole", opts: []history.Option{history.WithToolResultPreview(-1)}, want: full},
		{name: "last option wins", opts: []history.Option{history.WithToolResultPreview(0), nil, history.WithToolResultPreview(10)}, want: strings.Repeat("界", 10) + truncatedMarker},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			store := &historyStore{state: toolResultState(full)}
			source := history.NewHistory("s", store, tt.opts...)
			counter := ai.TextTokenEstimator{}
			source.SetTokenCounter(counter)
			part, tokens, err := source.FunctionWithTokens(t.Context(), 100000)
			if err != nil {
				t.Fatal(err)
			}
			messages := part.(*history.Part).ConversationMessages()
			if got := messages[2].ToolResults()[0].Text(); got != tt.want {
				t.Fatalf("tool result has %d runes, want %d", len([]rune(got)), len([]rune(tt.want)))
			}
			// Selection counted the same projection the part emits.
			recount, err := part.(*history.Part).Tokens(t.Context(), counter)
			if err != nil || recount != tokens {
				t.Fatalf("selection counted %d tokens, part counts %d (err %v)", tokens, recount, err)
			}

			builder := modelinput.New(modelinput.Definition{TokenBudget: 100000, ContextSources: []modelinput.ContextSource{history.NewHistory("s", store, tt.opts...)}, PromptInput: modelinput.PromptInput{User: ai.TextParts("now")}})
			if _, err := builder.BuildContext(t.Context()); err != nil {
				t.Fatal(err)
			}
			request, err := builder.BuildRequest(t.Context(), nil)
			if err != nil {
				t.Fatal(err)
			}
			if got := request.Messages[2].ToolResults()[0].Text(); got != tt.want {
				t.Fatalf("request tool result has %d runes, want %d", len([]rune(got)), len([]rune(tt.want)))
			}
			if store.state.Turns[0].Messages[1].Message.ToolResults()[0].Text() != full {
				t.Fatal("preview changed the stored result")
			}
		})
	}
}

func TestCompactorSelectsWithTheToolResultPreview(t *testing.T) {
	t.Parallel()
	state := toolResultState(strings.Repeat("word ", 2000), "short")
	// The previewed history fits this budget; the whole history does not.
	budget := 1000
	for _, tt := range []struct {
		name     string
		opts     []history.Option
		compacts bool
	}{
		{name: "default preview fits"},
		{name: "whole results exceed", opts: []history.Option{history.WithToolResultPreview(0)}, compacts: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			store := &sharedHistoryStore{state: state.Clone()}
			model := &recordingModel{reply: textReply("summary")}
			compactor, err := history.NewCompactor("s", store, history.CompactorDefinition{Model: model}, tt.opts...)
			if err != nil {
				t.Fatal(err)
			}
			prefix := transcriptFor(state)
			result, err := compactor.Compact(t.Context(), budget, history.WithPrefix(&prefix))
			if err != nil || result.Changed != tt.compacts {
				t.Fatalf("changed=%v, want %v (err %v)", result.Changed, tt.compacts, err)
			}
		})
	}
}

func TestContinuationMatchesPreviewsOfAnyLength(t *testing.T) {
	t.Parallel()
	long := strings.Repeat("task row; ", 300)
	state := toolResultState(long, "short")
	previewed := func(runes int) ai.AIRequest {
		prefix := transcriptFor(state)
		// The first turn's messages follow the system message.
		copy(prefix.Messages[1:5], previewMessagesAt(t, prefix.Messages[1:5], runes))
		return prefix
	}
	for _, runes := range []int{0, 1, 40, 500, 2000} {
		t.Run(fmt.Sprint(runes), func(t *testing.T) {
			t.Parallel()
			prefix := previewed(runes)
			if !strings.HasSuffix(prefix.Messages[3].ToolResults()[0].Text(), truncatedMarker) {
				t.Fatal("test prefix is not previewed")
			}
			store := &sharedHistoryStore{state: state.Clone()}
			model := &recordingModel{reply: textReply("summary")}
			if _, err := continuationCompactor(t, store, model, history.CompactorDefinition{Amount: .5}).Compact(t.Context(), 10, history.WithPrefix(&prefix)); err != nil {
				t.Fatal(err)
			}
		})
	}
	t.Run("preview of other text", func(t *testing.T) {
		t.Parallel()
		prefix := previewed(40)
		result := prefix.Messages[3].Parts[0].ToolResult
		result.Parts[0].Text = strings.Repeat("x", 40) + truncatedMarker
		store := &sharedHistoryStore{state: state.Clone()}
		model := &recordingModel{reply: textReply("summary")}
		_, err := continuationCompactor(t, store, model, history.CompactorDefinition{Amount: .5}).Compact(t.Context(), 10, history.WithPrefix(&prefix))
		if !errors.Is(err, history.ErrSummaryPrefixMismatch) || len(model.calls()) != 0 {
			t.Fatalf("err=%v calls=%d", err, len(model.calls()))
		}
	})
}

func TestAutoCompactionProjectsWithTheSourcePreview(t *testing.T) {
	t.Parallel()
	long := strings.Repeat("tool output ", 100)
	for _, tt := range []struct {
		name string
		opts []history.Option
		want string
	}{
		{name: "default", want: string([]rune(long)[:500]) + truncatedMarker},
		{name: "custom", opts: []history.Option{history.WithToolResultPreview(30)}, want: string([]rune(long)[:30]) + truncatedMarker},
		{name: "whole", opts: []history.Option{history.WithToolResultPreview(0)}, want: long},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			store := &sharedHistoryStore{state: toolResultState(long, long, long)}
			model := &recordingModel{reply: textReply("summary")}
			source := history.NewHistory("s", store, tt.opts...)
			preceding := ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleSystem, "system")}}
			result, err := source.CompactHistory(t.Context(), history.AutoCompaction{HistoryBudget: 10, Model: model, Counter: ai.TextTokenEstimator{}, Prefix: fixedPrefix(preceding)})
			if err != nil || !result.Changed {
				t.Fatalf("result=%+v err=%v", result, err)
			}
			request := model.calls()[0]
			// System, three turns of four messages, and the instruction.
			if len(request.Messages) != 14 {
				t.Fatalf("request = %d messages", len(request.Messages))
			}
			for _, index := range []int{3, 7} {
				if got := request.Messages[index].ToolResults()[0].Text(); got != tt.want {
					t.Fatalf("older tool result %d has %d runes, want %d", index, len([]rune(got)), len([]rune(tt.want)))
				}
			}
			if got := request.Messages[11].ToolResults()[0].Text(); got != long {
				t.Fatal("the newest turn must stay raw")
			}
		})
	}
}

// previewMessagesAt projects messages with a preview of runes, as a source
// built with WithToolResultPreview(runes) would. Zero cuts everything.
func previewMessagesAt(t *testing.T, messages []ai.Message, runes int) []ai.Message {
	t.Helper()
	if runes == 0 {
		messages = ai.CloneMessages(messages)
		for _, message := range messages {
			for _, part := range message.Parts {
				if part.ToolResult != nil {
					part.ToolResult.Parts = ai.TextParts(truncatedMarker)
				}
			}
		}
		return messages
	}
	store := &historyStore{state: &history.HistoryState{Turns: []modelinput.Turn{{ID: "x", Count: 1, Messages: storedAll(messages)}}}}
	source := history.NewHistory("x", store, history.WithToolResultPreview(runes))
	source.SetTokenCounter(ai.TextTokenEstimator{})
	part, err := source.Function(t.Context(), 1<<30)
	if err != nil {
		t.Fatal(err)
	}
	return part.(*history.Part).ConversationMessages()
}

func storedAll(messages []ai.Message) []modelinput.StoredMessage {
	out := make([]modelinput.StoredMessage, len(messages))
	for i, message := range messages {
		out[i] = stored(message)
	}
	return out
}

func TestContinuationMatchesMultipartPreviews(t *testing.T) {
	t.Parallel()
	var parts []ai.ContentPart
	for i := 0; i < 50; i++ {
		parts = append(parts, ai.TextParts(strings.Repeat("界", 10))...)
	}
	state := toolResultState("placeholder", "short")
	state.Turns[0].Messages[1].Message.Parts[0].ToolResult.Parts = parts
	// 120 ends on a part boundary, so the marker fills a part of its own;
	// 125 cuts inside a part.
	for _, runes := range []int{120, 125, 499} {
		t.Run(fmt.Sprint(runes), func(t *testing.T) {
			t.Parallel()
			prefix := transcriptFor(state)
			copy(prefix.Messages[1:5], previewMessagesAt(t, prefix.Messages[1:5], runes))
			if !strings.HasSuffix(prefix.Messages[3].ToolResults()[0].Text(), truncatedMarker) {
				t.Fatal("test prefix is not previewed")
			}
			store := &sharedHistoryStore{state: state.Clone()}
			model := &recordingModel{reply: textReply("summary")}
			if _, err := continuationCompactor(t, store, model, history.CompactorDefinition{Amount: .5}).Compact(t.Context(), 10, history.WithPrefix(&prefix)); err != nil {
				t.Fatal(err)
			}
		})
	}
}
