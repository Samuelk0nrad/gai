package history_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput"
	"github.com/Samuelk0nrad/gai/modelinput/history"
)

// recordingModel replies with a scripted stream and records each request.
type recordingModel struct {
	mu       sync.Mutex
	requests []ai.AIRequest
	reply    func(ai.AIRequest) []ai.Token
}

func (m *recordingModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	m.mu.Lock()
	m.requests = append(m.requests, req.Copy())
	m.mu.Unlock()
	out := make(chan ai.Token, 8)
	go func() {
		defer close(out)
		for _, token := range m.reply(req) {
			ai.SendToken(ctx, out, token)
		}
	}()
	return out
}

func (m *recordingModel) calls() []ai.AIRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]ai.AIRequest(nil), m.requests...)
}

func textReply(text string) func(ai.AIRequest) []ai.Token {
	return func(ai.AIRequest) []ai.Token {
		part := ai.ContentPart{Kind: ai.ContentText, Text: text}
		return []ai.Token{{Part: &part}, {Completion: &ai.Completion{UsageReported: true, FinishReason: "end_turn", Usage: ai.Usage{InputTokens: 900, CachedTokens: 850, OutputTokens: 12}}}}
	}
}

func stored(message ai.Message) modelinput.StoredMessage {
	return modelinput.StoredMessage{Message: message}
}

// conversationTurn is a user question and an assistant answer.
func conversationTurn(n int, question, answer string) modelinput.Turn {
	user := stored(ai.TextMessage(ai.RoleUser, question))
	return modelinput.Turn{ID: fmt.Sprint("t", n), Count: n, UserMessage: &user, Messages: []modelinput.StoredMessage{stored(ai.TextMessage(ai.RoleAssistant, answer))}}
}

func threeTurnState() *history.HistoryState {
	return &history.HistoryState{SchemaVersion: history.HistorySchemaVersion, Turns: []modelinput.Turn{
		conversationTurn(1, "First question about the Go module layout and how packages import each other.", "First answer"),
		conversationTurn(2, "Second question about how the history store compares revisions on write.", "Second answer"),
		conversationTurn(3, "Third question about prompt caching and what makes the cached prefix match.", "Third answer"),
	}}
}

// transcriptFor is a run's final request: system prompt, then every turn.
func transcriptFor(state *history.HistoryState) ai.AIRequest {
	request := ai.AIRequest{
		Messages:   []ai.Message{ai.TextMessage(ai.RoleSystem, "You are helpful.")},
		Tools:      []ai.ToolDefinition{{Type: "function", Name: "lookup", Description: "Look something up.", Parameters: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice: ai.ToolChoice{Mode: ai.ToolChoiceAuto},
		Reasoning:  ai.ReasoningConfig{Enabled: true, BudgetTokens: 64},
		MaxTokens:  512,
	}
	if state.Summary != nil {
		request.Messages = append(request.Messages, ai.TextMessage(ai.RoleUser, "Conversation summary:\n"+state.Summary.Content.Text))
	}
	for _, turn := range state.Turns {
		request.Messages = append(request.Messages, turn.UserMessage.Message.Clone())
		for _, message := range turn.Messages {
			request.Messages = append(request.Messages, message.Message.Clone())
		}
	}
	return request
}

func continuationCompactor(t *testing.T, store history.HistoryStore, model ai.Model, def history.CompactorDefinition) *history.Compactor {
	t.Helper()
	def.Model = model
	compactor, err := history.NewCompactor("s", store, def)
	if err != nil {
		t.Fatal(err)
	}
	return compactor
}

func ref[T any](v T) *T { return &v }

// fixedPrefix is an AutoCompaction prefix that renders request.
func fixedPrefix(request ai.AIRequest) func(context.Context) (ai.AIRequest, error) {
	return func(context.Context) (ai.AIRequest, error) { return request, nil }
}

func lastText(request ai.AIRequest) string {
	return request.Messages[len(request.Messages)-1].Text()
}

func TestContinuationSummaryAppendsOneInstructionToTheUnchangedPrefix(t *testing.T) {
	state := threeTurnState()
	store := &sharedHistoryStore{state: state.Clone()}
	model := &recordingModel{reply: textReply("compact summary")}
	prefix := transcriptFor(state)
	prefix.ResponseFormat = ai.ResponseFormat{Type: ai.ResponseFormatJSONObject}
	compactor := continuationCompactor(t, store, model, history.CompactorDefinition{})

	result, err := compactor.Compact(t.Context(), 10, history.WithPrefix(&prefix), history.WithExpectedRevision("1"))
	if err != nil {
		t.Fatal(err)
	}
	calls := model.calls()
	if len(calls) != 1 {
		t.Fatalf("calls = %d", len(calls))
	}
	request := calls[0]
	want := prefix.Copy()
	want.ResponseFormat = ai.ResponseFormat{} // omitted by default
	// The summary has its own output limit, not the answer's; the limit
	// does not affect the prompt cache.
	want.MaxTokens = summaryReserve(prefix)
	got := request.Copy()
	got.Messages = got.Messages[:len(prefix.Messages)]
	if !reflect.DeepEqual(got, want) || len(request.Messages) != len(prefix.Messages)+1 {
		t.Fatalf("summary request does not continue the prefix:\n got %+v\nwant %+v", request, want)
	}
	instruction := request.Messages[len(request.Messages)-1]
	if instruction.Role != ai.RoleUser || !strings.Contains(instruction.Text(), `"Third question about prompt caching`) || !strings.Contains(instruction.Text(), "Do not call any tool") {
		t.Fatalf("instruction = %q", instruction.Text())
	}
	if !result.Changed || result.SummarizedTurns != 2 || !result.UsageReported || result.Usage.CachedTokens != 850 {
		t.Fatalf("result = %+v", result)
	}
	if store.state.Summary == nil || store.state.Summary.Content.Text != "compact summary" || len(store.state.Turns) != 1 || store.state.Turns[0].ID != "t3" {
		t.Fatalf("stored = %+v", store.state)
	}
}

func TestContinuationSummaryMergesTheEarlierSummary(t *testing.T) {
	state := threeTurnState()
	state.Summary = history.NewSummary("old", "t0", "t0", 0, 0, ai.ContentPart{Kind: ai.ContentText, Text: "earlier facts"})
	for _, present := range []bool{true, false} {
		t.Run(fmt.Sprint(present), func(t *testing.T) {
			store := &sharedHistoryStore{state: state.Clone()}
			model := &recordingModel{reply: textReply("merged")}
			prefix := transcriptFor(state)
			if !present {
				prefix.Messages = append(prefix.Messages[:1], prefix.Messages[2:]...)
			}
			_, err := continuationCompactor(t, store, model, history.CompactorDefinition{}).Compact(t.Context(), 10, history.WithPrefix(&prefix))
			if !present {
				if !errors.Is(err, history.ErrSummaryPrefixMismatch) || len(model.calls()) != 0 || store.casCalls != 0 {
					t.Fatalf("err=%v calls=%d writes=%d", err, len(model.calls()), store.casCalls)
				}
				return
			}
			if err != nil || !strings.Contains(lastText(model.calls()[0]), `"Conversation summary:"`) {
				t.Fatalf("err=%v", err)
			}
			if store.state.Summary.StartTurnID != "t0" || store.state.Summary.EndTurnID != "t2" {
				t.Fatalf("summary coverage = %+v", store.state.Summary)
			}
		})
	}
}

func TestContinuationMatchesStoredTurnsAfterRoundTripsAndPreviews(t *testing.T) {
	long := strings.Repeat("result line. ", 100)
	toolTurn := func(args string, resultText string) modelinput.Turn {
		user := stored(ai.TextMessage(ai.RoleUser, "Look up the deployment status for the payments service."))
		return modelinput.Turn{ID: "t1", Count: 1, UserMessage: &user, Messages: []modelinput.StoredMessage{
			stored(ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "c1", Type: "function", Name: "lookup", Args: json.RawMessage(args)}}}}),
			stored(ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "c1", Name: "lookup", Parts: ai.TextParts(resultText)}}}}),
			stored(ai.TextMessage(ai.RoleAssistant, "It is deployed.")),
		}}
	}
	// Storage compacted and reordered the arguments; the run streamed them
	// with whitespace and HTML escapes.
	storedTurn := toolTurn(`{"b":1,"a":"<x>"}`, long)
	sentTurn := toolTurn("{ \"a\": \"\\u003cx\\u003e\", \"b\": 1 }", long)
	state := &history.HistoryState{SchemaVersion: history.HistorySchemaVersion, Turns: []modelinput.Turn{storedTurn, conversationTurn(2, "Next question that stays in the recent history window.", "ok")}}
	state.Turns[1].ID, state.Turns[1].Count = "t2", 2
	for _, preview := range []bool{false, true} {
		t.Run(fmt.Sprint("preview=", preview), func(t *testing.T) {
			sent := &history.HistoryState{Turns: []modelinput.Turn{sentTurn, state.Turns[1]}}
			prefix := transcriptFor(sent)
			if preview {
				// Older turns reach the prompt through the history preview.
				part := &history.Part{Messages: prefix.Messages[1:5]}
				copy(prefix.Messages[1:5], part.ConversationMessages())
				if !strings.Contains(prefix.Messages[3].Parts[0].ToolResult.Parts[0].Text, "[tool result truncated]") {
					t.Fatal("test prefix is not previewed")
				}
			}
			store := &sharedHistoryStore{state: state.Clone()}
			model := &recordingModel{reply: textReply("summary")}
			if _, err := continuationCompactor(t, store, model, history.CompactorDefinition{Amount: .5}).Compact(t.Context(), 10, history.WithPrefix(&prefix)); err != nil {
				t.Fatal(err)
			}
		})
	}
	t.Run("changed content", func(t *testing.T) {
		prefix := transcriptFor(&history.HistoryState{Turns: []modelinput.Turn{toolTurn(`{"a":"<x>","b":2}`, long), state.Turns[1]}})
		store := &sharedHistoryStore{state: state.Clone()}
		model := &recordingModel{reply: textReply("summary")}
		_, err := continuationCompactor(t, store, model, history.CompactorDefinition{Amount: .5}).Compact(t.Context(), 10, history.WithPrefix(&prefix))
		if !errors.Is(err, history.ErrSummaryPrefixMismatch) || len(model.calls()) != 0 {
			t.Fatalf("err=%v calls=%d", err, len(model.calls()))
		}
	})
}

func TestContinuationCompactValidatesBeforeModelWork(t *testing.T) {
	state := threeTurnState()
	structured := transcriptFor(state)
	structured.ResponseFormat = ai.ResponseFormat{Type: ai.ResponseFormatJSONObject}
	for _, tc := range []struct {
		name string
		def  history.CompactorDefinition
		opts []history.CompactOption
		want error
		load bool
	}{
		{"no prefix", history.CompactorDefinition{}, nil, history.ErrSummaryPrefixRequired, false},
		// The transcript of a run that did not complete is nil.
		{"nil prefix", history.CompactorDefinition{}, []history.CompactOption{history.WithPrefix(nil)}, history.ErrSummaryPrefixRequired, false},
		{"empty expected revision", history.CompactorDefinition{}, []history.CompactOption{history.WithPrefix(ref(transcriptFor(state))), history.WithExpectedRevision("")}, history.ErrInvalidExpectedRevision, false},
		{"stale expected revision", history.CompactorDefinition{}, []history.CompactOption{history.WithPrefix(ref(transcriptFor(state))), history.WithExpectedRevision("0")}, &history.RevisionConflictError{}, true},
		{"rejected response format", history.CompactorDefinition{ResponseFormat: history.SummaryResponseFormatReject}, []history.CompactOption{history.WithPrefix(&structured)}, history.ErrInvalidSummaryPrefix, true},
		{"too small", history.CompactorDefinition{}, []history.CompactOption{history.WithPrefix(ref(transcriptFor(state))), history.WithRequestBudget(ai.RequestBudgetConfig{Limit: 20})}, history.ErrSummaryRequestTooLarge, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store := &sharedHistoryStore{state: state.Clone()}
			model := &recordingModel{reply: textReply("summary")}
			_, err := continuationCompactor(t, store, model, tc.def).Compact(t.Context(), 10, tc.opts...)
			var conflict *history.RevisionConflictError
			if _, isConflict := tc.want.(*history.RevisionConflictError); isConflict {
				if !errors.As(err, &conflict) || conflict.Expected != "0" || conflict.Actual != "1" {
					t.Fatalf("err = %v", err)
				}
			} else if !errors.Is(err, tc.want) {
				t.Fatalf("err = %v, want %v", err, tc.want)
			}
			if len(model.calls()) != 0 || store.casCalls != 0 {
				t.Fatalf("calls=%d writes=%d", len(model.calls()), store.casCalls)
			}
		})
	}
	if _, err := history.NewCompactor("s", &sharedHistoryStore{}, history.CompactorDefinition{Model: &recordingModel{}, ResponseFormat: 9}); !errors.Is(err, history.ErrInvalidSummaryResponseFormat) {
		t.Fatal(err)
	}
}

func TestContinuationSendsARequiredToolChoiceAsAuto(t *testing.T) {
	state := threeTurnState()
	prefix := transcriptFor(state)
	prefix.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"lookup"}}
	store := &sharedHistoryStore{state: state.Clone()}
	model := &recordingModel{reply: textReply("summary")}
	if _, err := continuationCompactor(t, store, model, history.CompactorDefinition{}).Compact(t.Context(), 10, history.WithPrefix(&prefix)); err != nil {
		t.Fatal(err)
	}
	if choice := model.calls()[0].ToolChoice; choice.Mode != ai.ToolChoiceAuto || len(choice.Names) != 0 {
		t.Fatalf("tool choice = %+v", choice)
	}
}

func TestContinuationKeepsResponseFormatAndAddsReasoningBudget(t *testing.T) {
	state := threeTurnState()
	prefix := transcriptFor(state)
	prefix.ResponseFormat = ai.ResponseFormat{Type: ai.ResponseFormatJSONObject}
	store := &sharedHistoryStore{state: state.Clone()}
	// A structured reply arrives as a JSON part, not text.
	model := &recordingModel{reply: func(ai.AIRequest) []ai.Token {
		part := ai.ContentPart{Kind: ai.ContentJSON, JSON: json.RawMessage(`{"summary":"x"}`)}
		return []ai.Token{{Part: &part}, {Completion: &ai.Completion{FinishReason: "end_turn"}}}
	}}
	def := history.CompactorDefinition{ResponseFormat: history.SummaryResponseFormatKeep, SummaryMaxTokens: 100}
	if _, err := continuationCompactor(t, store, model, def).Compact(t.Context(), 10, history.WithPrefix(&prefix)); err != nil {
		t.Fatal(err)
	}
	request := model.calls()[0]
	if !reflect.DeepEqual(request.ResponseFormat, prefix.ResponseFormat) || request.MaxTokens != 164 {
		t.Fatalf("format=%+v max=%d", request.ResponseFormat, request.MaxTokens)
	}
	if got := store.state.Summary.Content.Text; got != `{"summary":"x"}` {
		t.Fatalf("summary = %q", got)
	}
}

// summaryReserve is the output a default continuation summary reserves.
func summaryReserve(prefix ai.AIRequest) int {
	return history.DefaultSummaryMaxTokens + prefix.Reasoning.BudgetTokens
}

func TestContinuationCutsThePrefixWhenItDoesNotFit(t *testing.T) {
	state := threeTurnState()
	prefix := transcriptFor(state)
	// Pad the kept turn so only cut prefixes fit, and the second turn so
	// that a smaller budget fits only the first.
	pad := strings.Repeat("padding ", 400)
	state.Turns[2].Messages[0].Message = ai.TextMessage(ai.RoleAssistant, pad)
	prefix.Messages[len(prefix.Messages)-1] = ai.TextMessage(ai.RoleAssistant, pad)
	filler := strings.Repeat("filler ", 150)
	state.Turns[1].Messages[0].Message = ai.TextMessage(ai.RoleAssistant, filler)
	prefix.Messages[4] = ai.TextMessage(ai.RoleAssistant, filler)
	counter := ai.TextTokenEstimator{}
	full, err := ai.EstimateRequestTokens(t.Context(), prefix, counter)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name     string
		limit    int
		folded   int
		messages int
	}{
		// System, t1 and t2 plus the instruction.
		{"fold every selected turn", full.InputTokens + summaryReserve(prefix) - 100, 2, 6},
		// System and t1 plus the instruction.
		{"fold fewer turns", 0, 1, 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			limit := tc.limit
			if limit == 0 {
				oneTurn := prefix.Copy()
				oneTurn.Messages = append(oneTurn.Messages[:3], ai.TextMessage(ai.RoleUser, "x"))
				estimate, err := ai.EstimateRequestTokens(t.Context(), oneTurn, counter)
				if err != nil {
					t.Fatal(err)
				}
				// Room for the instruction, not for the padded second turn.
				limit = estimate.InputTokens + summaryReserve(prefix) + 300
			}
			store := &sharedHistoryStore{state: state.Clone()}
			model := &recordingModel{reply: textReply("summary")}
			result, err := continuationCompactor(t, store, model, history.CompactorDefinition{}).Compact(t.Context(), 10, history.WithPrefix(&prefix), history.WithRequestBudget(ai.RequestBudgetConfig{Limit: limit}))
			if err != nil {
				t.Fatal(err)
			}
			request := model.calls()[0]
			if len(request.Messages) != tc.messages || !strings.Contains(lastText(request), "the whole conversation above") || result.SummarizedTurns != tc.folded {
				t.Fatalf("messages=%d folded=%d instruction=%q", len(request.Messages), result.SummarizedTurns, lastText(request))
			}
			if len(store.state.Turns) != 3-tc.folded {
				t.Fatalf("stored turns = %d", len(store.state.Turns))
			}
		})
	}
}

// instructionCounter counts how many requests were estimated: each estimate
// counts the summary instruction once.
type instructionCounter struct {
	ai.TextTokenEstimator
	mu        sync.Mutex
	estimates int
}

func (c *instructionCounter) CountTokens(ctx context.Context, text string) (int, error) {
	if strings.HasPrefix(text, "Stop the conversation here.") {
		c.mu.Lock()
		c.estimates++
		c.mu.Unlock()
	}
	return c.TextTokenEstimator.CountTokens(ctx, text)
}

func TestContinuationFoldsTheMostTurnsThatFitWithFewEstimates(t *testing.T) {
	const turns, fit = 64, 5
	answer := strings.Repeat("answer ", 100)
	state := &history.HistoryState{SchemaVersion: history.HistorySchemaVersion}
	for n := 1; n <= turns; n++ {
		state.Turns = append(state.Turns, conversationTurn(n, fmt.Sprintf("Question %d about a different part of the system.", n), answer))
	}
	prefix := transcriptFor(state)
	instruction := "Stop the conversation here. Do not call any tool and do not reply to the last message.\n\n" +
		"Write a summary that another agent will use in place of the whole conversation above.\n\n" + history.DefaultContinuationInstructions
	cut := prefix.Copy()
	cut.Messages = append(cut.Messages[:1+2*fit], ai.TextMessage(ai.RoleUser, instruction))
	estimate, err := ai.EstimateRequestTokens(t.Context(), cut, ai.TextTokenEstimator{})
	if err != nil {
		t.Fatal(err)
	}
	// Room for fit turns and the instruction, not for one more turn.
	limit := estimate.InputTokens + summaryReserve(prefix) + 50
	counter := &instructionCounter{}
	model := &recordingModel{reply: textReply("summary")}
	output, err := history.NewContinuationSummarizer(model).SummarizeHistory(t.Context(), history.SummaryInput{
		Prefix: &prefix, Turns: state.Turns[:turns-1], Kept: state.Turns[turns-1:],
		RequestBudget: ai.RequestBudgetConfig{Limit: limit}, Counter: counter,
	})
	if err != nil {
		t.Fatal(err)
	}
	if output.FoldedTurns != fit || len(model.calls()[0].Messages) != 2+2*fit {
		t.Fatalf("folded=%d messages=%d", output.FoldedTurns, len(model.calls()[0].Messages))
	}
	// The full prefix, the cut after every folded turn, then a binary search
	// over the 62 smaller cuts: at most 2 + 6 estimates, not 63.
	if counter.estimates > 8 {
		t.Fatalf("estimated %d requests", counter.estimates)
	}
}

func TestContinuationCutsThePrefixWhenTheBoundaryIsAmbiguous(t *testing.T) {
	state := threeTurnState()
	state.Turns[0].UserMessage.Message = ai.TextMessage(ai.RoleUser, "Continue.")
	state.Turns[2].UserMessage.Message = ai.TextMessage(ai.RoleUser, "Continue.")
	store := &sharedHistoryStore{state: state.Clone()}
	model := &recordingModel{reply: textReply("summary")}
	if _, err := continuationCompactor(t, store, model, history.CompactorDefinition{}).Compact(t.Context(), 10, history.WithPrefix(ref(transcriptFor(state)))); err != nil {
		t.Fatal(err)
	}
	if request := model.calls()[0]; len(request.Messages) != 6 || !strings.Contains(lastText(request), "the whole conversation above") {
		t.Fatalf("request = %+v", request.Messages)
	}
}

func TestCompactionKeepsTheUsageOfAFailedSummaryCall(t *testing.T) {
	state := threeTurnState()
	store := &sharedHistoryStore{state: state.Clone()}
	model := &recordingModel{reply: func(ai.AIRequest) []ai.Token {
		part := ai.ContentPart{Kind: ai.ContentText, Text: "cut off"}
		return []ai.Token{{Part: &part}, {Completion: &ai.Completion{FinishReason: "max_tokens", UsageReported: true, Usage: ai.Usage{InputTokens: 900, CachedTokens: 850, OutputTokens: 2048}}}}
	}}
	result, err := continuationCompactor(t, store, model, history.CompactorDefinition{}).Compact(t.Context(), 10, history.WithPrefix(ref(transcriptFor(state))))
	if !errors.Is(err, history.ErrInvalidSummaryReply) || !result.UsageReported || result.Usage.InputTokens != 900 || result.Usage.OutputTokens != 2048 || result.Changed {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

func TestContinuationRejectsInvalidReplies(t *testing.T) {
	call := ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "c", Type: "function", Name: "lookup", Args: json.RawMessage(`{}`)}}
	text := ai.ContentPart{Kind: ai.ContentText, Text: "partial"}
	blank := ai.ContentPart{Kind: ai.ContentText, Text: "  "}
	failure := errors.New("provider down")
	for name, tokens := range map[string][]ai.Token{
		"tool call": {{Part: &call}},
		"truncated": {{Part: &text}, {Completion: &ai.Completion{FinishReason: "max_tokens"}}},
		"empty":     {{Part: &blank}},
		"error":     {{Err: failure}},
	} {
		t.Run(name, func(t *testing.T) {
			state := threeTurnState()
			store := &sharedHistoryStore{state: state.Clone()}
			model := &recordingModel{reply: func(ai.AIRequest) []ai.Token { return tokens }}
			_, err := continuationCompactor(t, store, model, history.CompactorDefinition{}).Compact(t.Context(), 10, history.WithPrefix(ref(transcriptFor(state))))
			if !errors.Is(err, history.ErrInvalidSummaryReply) && !errors.Is(err, failure) {
				t.Fatalf("err = %v", err)
			}
			if store.casCalls != 0 {
				t.Fatal("invalid summary was written")
			}
		})
	}
}

func TestAutoCompactionAppendsTheFullStoredHistory(t *testing.T) {
	long := strings.Repeat("tool output ", 100)
	state := threeTurnState()
	withTool := func(turn *modelinput.Turn) {
		turn.Messages = append([]modelinput.StoredMessage{
			stored(ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: turn.ID, Type: "function", Name: "lookup", Args: json.RawMessage(`{}`)}}}}),
			stored(ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: turn.ID, Name: "lookup", Parts: ai.TextParts(long)}}}}),
		}, turn.Messages...)
	}
	withTool(&state.Turns[0])
	withTool(&state.Turns[2])
	store := &sharedHistoryStore{state: state.Clone()}
	model := &recordingModel{reply: textReply("summary")}
	source := history.NewHistory("s", store)
	preceding := ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleSystem, "system")}}
	result, err := source.CompactHistory(t.Context(), history.AutoCompaction{HistoryBudget: 10, Model: model, Counter: ai.TextTokenEstimator{}, Prefix: fixedPrefix(preceding)})
	if err != nil || !result.Changed || !result.PressureRemaining {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	request := model.calls()[0]
	// System, three turns (4 + 2 + 4 messages) and the instruction.
	if len(request.Messages) != 12 || request.Messages[0].Text() != "system" {
		t.Fatalf("request = %d messages", len(request.Messages))
	}
	older := request.Messages[3].Parts[0].ToolResult.Parts[0].Text
	newest := request.Messages[9].Parts[0].ToolResult.Parts[0].Text
	if !strings.HasSuffix(older, "[tool result truncated]") || newest != long {
		t.Fatalf("older turns must be previewed and the newest raw: %q / %q", older[len(older)-30:], newest[len(newest)-30:])
	}
}
