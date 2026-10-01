package history_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/history"
)

func TestHistoryBuilderUsesSameSelectedAndTruncatedCanonicalMessages(t *testing.T) {
	t.Parallel()
	fullResult := strings.Repeat("界", 501)
	call := ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{
		{Kind: ai.ContentText, Text: "searching"},
		{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "call-1", Type: "function", Name: "search", Args: []byte(`{"q":"first"}`)}},
		{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "call-2", Type: "function", Name: "search", Args: []byte(`{"q":"second"}`)}},
	}}
	results := ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{
		{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "call-2", Name: "search", Parts: ai.TextParts("failed"), IsError: true}},
		{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "call-1", Name: "search", Parts: ai.TextParts(fullResult)}},
	}}
	store := &historyStore{state: &history.HistoryState{Turns: []gaictx.Turn{
		{ID: "old", Count: 1, Messages: []gaictx.StoredMessage{{Message: ai.TextMessage(ai.RoleAssistant, "excluded")}}, TokenCount: map[string]int{"gai.estimate/utf8-bytes-v1": 999999}},
		{ID: "new", Count: 2, UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "earlier question")}, Messages: []gaictx.StoredMessage{{Message: call}, {Message: results}}},
	}}}
	builder := gaictx.New(gaictx.Definition{TokenBudget: 10000, ContextSources: []gaictx.ContextSource{history.NewHistory("session", store)}, PromptInput: gaictx.PromptInput{User: ai.TextParts("current question")}})
	// Use a persisted count under the effective estimator identity to exclude
	// the older turn independently of its short text.
	store.state.Turns[0].TokenCount = map[string]int{builder.TokenCounter().ID(): 999999}
	if _, err := builder.BuildContext(t.Context()); err != nil {
		t.Fatal(err)
	}
	request, err := builder.BuildRequest(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 4 || request.Messages[0].Text() != "earlier question" || request.Messages[3].Text() != "current question" {
		t.Fatalf("wrong selected history: %#v", request.Messages)
	}
	gotResults := request.Messages[2].ToolResults()
	if len(gotResults) != 2 || gotResults[0].ToolCallID != "call-2" || !gotResults[0].IsError || gotResults[1].ToolCallID != "call-1" {
		t.Fatalf("tool identity/error/order lost: %#v", gotResults)
	}
	wantResult := strings.Repeat("界", 500) + "\n[tool result truncated]"
	if gotResults[1].Text() != wantResult {
		t.Fatalf("native request did not apply shared truncation: %q", gotResults[1].Text())
	}
	fallback, err := renderHistoryRequest(builder, t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	wantFallback, err := ai.RenderMessages(t.Context(), request.Messages)
	if err != nil || fallback != wantFallback {
		t.Fatalf("native/fallback canonical content differs: %v", err)
	}
	if strings.Contains(fallback, "excluded") {
		t.Fatal("fallback reintroduced omitted history")
	}
	if store.state.Turns[1].Messages[1].Message.ToolResults()[1].Text() != fullResult {
		t.Fatal("history preview mutated persisted payload")
	}
}

func TestHistoryStateCanonicalRoundTrip(t *testing.T) {
	t.Parallel()
	want := history.HistoryState{SchemaVersion: history.HistorySchemaVersion, Summary: history.NewSummary("summary", "t0", "t0", 0, 0, ai.ContentPart{Kind: ai.ContentText, Text: "earlier"}), Turns: []gaictx.Turn{{ID: "turn", Count: 1, UserMessage: &gaictx.StoredMessage{SchemaVersion: gaictx.MessageSchemaVersion, ID: "message", SessionID: "session", TurnID: "turn", TokenCount: map[string]int{"counter": 7}, Message: ai.TextMessage(ai.RoleUser, "hello")}}}}
	data, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	var got history.HistoryState
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got.SchemaVersion != want.SchemaVersion || !reflect.DeepEqual(got.Turns, want.Turns) || got.Summary.Content.Text != "earlier" {
		t.Fatalf("state round trip changed storage metadata: %#v", got)
	}

}

func TestHistorySourcePreservesUnknownExtensionsBeforeNativeMapping(t *testing.T) {
	t.Parallel()
	message := ai.Message{Role: ai.RoleAssistant, Parts: ai.TextParts("answer"), Extensions: []ai.Extension{{Namespace: "future", Type: "signature", Data: []byte(`"bytes"`), Required: true}}}
	store := &historyStore{state: &history.HistoryState{Turns: []gaictx.Turn{{ID: "turn", Count: 1, Messages: []gaictx.StoredMessage{{Message: message}}}}}}
	builder := gaictx.New(gaictx.Definition{ContextSources: []gaictx.ContextSource{history.NewHistory("session", store)}})
	if _, err := builder.BuildContext(t.Context()); err != nil {
		t.Fatal(err)
	}
	request, err := builder.BuildRequest(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 1 || !reflect.DeepEqual(request.Messages[0], message) {
		t.Fatalf("history projection changed opaque state: %#v", request.Messages)
	}
	if _, err := renderHistoryRequest(builder, t.Context(), nil); err == nil {
		t.Fatal("fallback silently discarded unknown extensions")
	}
}

func TestHistoryStateRejectsInvalidSummaryWhenSaving(t *testing.T) {
	t.Parallel()
	for _, content := range []ai.ContentPart{
		{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", URI: "https://example.test/image.png"}},
		{Kind: ai.ContentText, Text: "summary", JSON: []byte(`{}`)},
		{Text: "missing canonical tag"},
	} {
		state := history.HistoryState{Summary: &history.Summary{Content: content}}
		if encoded, err := json.Marshal(state); err == nil {
			t.Fatalf("saved invalid summary as unreadable history: %s", encoded)
		}
	}
	want := ai.ContentPart{Kind: ai.ContentText, Text: "summary", Extensions: []ai.Extension{{Namespace: "future", Type: "continuity", Data: []byte(`"opaque"`), Required: true}}}
	encoded, err := json.Marshal(history.HistoryState{Summary: &history.Summary{Content: want}})
	if err != nil {
		t.Fatal(err)
	}
	var got history.HistoryState
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Summary.Content, want) {
		t.Fatal("valid summary lost provider extension in storage")
	}
}

func TestHistoryStateRejectsOldAndUnversionedFormats(t *testing.T) {
	t.Parallel()
	for _, payload := range []string{
		`{"Turns":[]}`,
		`{"schema_version":0,"Turns":[]}`,
		`{"schema_version":42,"Turns":[]}`,
		`{"schema_version":1,"Turns":[{"ID":"turn","UserMessage":{"Role":"user","Content":{"Text":"old input"}}}]}`,
		`{"schema_version":1,"Summary":{"Content":{"Text":"old summary"}}}`,
	} {
		var got history.HistoryState
		if err := json.Unmarshal([]byte(payload), &got); err == nil {
			t.Fatalf("noncanonical history accepted: %s", payload)
		}
	}
}

func TestSummaryRejectsMissingCanonicalContentKind(t *testing.T) {
	t.Parallel()
	var got history.Summary
	if err := json.Unmarshal([]byte(`{"Content":{"Text":"old summary"}}`), &got); err == nil {
		t.Fatal("summary without canonical kind accepted")
	}
}
