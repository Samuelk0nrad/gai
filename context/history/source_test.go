package history_test

import (
	"context"
	"strings"
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/history"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestHistoryPartRendersStructuredContent(t *testing.T) {
	t.Parallel()

	part := &history.Part{
		Messages: []ai.Message{
			{Role: ai.RoleUser, Parts: ai.TextParts("hello")},
			{Role: ai.RoleAssistant, Parts: []ai.ContentPart{
				{Kind: ai.ContentToolCall,
					ToolCall: &ai.ToolCall{ID: "call_search",
						Type: "function", Name: "search", Args: []byte(`{"q":"lace"}`)}}},
			},
			{Role: ai.RoleTool, Parts: []ai.ContentPart{
				{Kind: ai.ContentToolResult,
					ToolResult: &ai.ToolResult{ToolCallID: "call_search", Name: "search", Parts: ai.TextParts("found docs")}}},
			},
		},
	}

	got, err := (gaictx.XMLRenderer{}).Render(context.Background(), []gaictx.Part{part})
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	expected := []string{
		`<history>`,
		`<user>`,
		`hello`,
		`</user>`,
		`<assistant>`,
		`<tool_call id="call_search" name="search">`,
		`<arguments>`,
		`&#34;q&#34;`,
		`<tool>`,
		`<tool_result id="call_search" name="search" is_error="false">`,
		`found docs`,
	}
	for _, fragment := range expected {
		if !strings.Contains(got, fragment) {
			t.Fatalf("expected history render to contain %q:\n%s", fragment, got)
		}
	}
	rejected := []string{
		`<message role=`,
		`<user><text>`,
		`search({"q":"lace"})`,
		`search result: found docs`,
	}
	for _, fragment := range rejected {
		if strings.Contains(got, fragment) {
			t.Fatalf("expected history render not to contain %q:\n%s", fragment, got)
		}
	}
}

func TestHistoryPartRendersSimpleContent(t *testing.T) {
	t.Parallel()

	part := &history.Part{
		Messages: []ai.Message{
			{Role: ai.RoleUser, Parts: ai.TextParts(`hello <world> & "quotes"`)},
			{Role: ai.RoleAssistant, Parts: []ai.ContentPart{
				{Kind: ai.ContentToolCall,
					ToolCall: &ai.ToolCall{ID: "call_search",
						Type: "function", Name: "search", Args: []byte(`{"q":"lace<&>"}`)}}},
			},
			{Role: ai.RoleTool, Parts: []ai.ContentPart{
				{Kind: ai.ContentToolResult,
					ToolResult: &ai.ToolResult{ToolCallID: "call_search", Name: "search", Parts: ai.TextParts(`found <docs> & "notes"`)}}},
			},
			{Role: ai.RoleUser, Parts: ai.TextParts("older turns")},
		},
	}

	got, err := (gaictx.SimpleRenderer{}).Render(context.Background(), []gaictx.Part{part})
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	want := `<history>
user: hello <world> & "quotes"
assistant: {"arguments":{"q":"lace\u003c\u0026\u003e"},"name":"search","type":"function"}
tool res: found <docs> & "notes"
user: older turns
</history>`
	if got != want {
		t.Fatalf("unexpected simple render output:\nwant %q\n got %q", want, got)
	}
	if strings.Contains(got, "&lt;") || strings.Contains(got, "&amp;") || strings.Contains(got, "&#34;") {
		t.Fatalf("expected raw characters to be preserved: %q", got)
	}
}

func TestHistoryPartTruncatesToolResultToPrefix(t *testing.T) {
	t.Parallel()

	result := strings.Repeat("a", 499) + "👋" + "discarded"
	part := &history.Part{
		Messages: []ai.Message{
			{Role: ai.RoleTool, Parts: []ai.ContentPart{
				{Kind: ai.ContentToolResult,
					ToolResult: &ai.ToolResult{ToolCallID: "call_search", Name: "search", Parts: ai.TextParts(result)}}},
			},
		},
	}

	got, err := (gaictx.SimpleRenderer{}).Render(context.Background(), []gaictx.Part{part})
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}
	want := "<history>\ntool res:\n" + strings.Repeat("a", 499) + "👋\n[tool result truncated]\n</history>"
	if got != want {
		t.Fatalf("unexpected truncated tool result:\nwant %q\n got %q", want, got)
	}
	if strings.Contains(got, "discarded") {
		t.Fatalf("history contains the discarded tool-result suffix: %q", got)
	}
}

func TestHistoryPartRenderEmpty(t *testing.T) {
	t.Parallel()

	var part *history.Part
	node, err := part.Render(context.Background())
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}
	if node.Type != "history" || len(node.Children) != 0 {
		t.Fatalf("expected empty history node, got %+v", node)
	}
}

func TestHistoryPartRendersSummary(t *testing.T) {
	t.Parallel()

	part := &history.Part{
		Messages: []ai.Message{
			{Role: ai.RoleUser, Parts: ai.TextParts("older turns")},
		},
	}

	got, err := (gaictx.XMLRenderer{}).Render(context.Background(), []gaictx.Part{part})
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	expected := []string{
		`<history>`,
		`<user>`,
		`older turns`,
	}
	for _, fragment := range expected {
		if !strings.Contains(got, fragment) {
			t.Fatalf("expected summary render to contain %q:\n%s", fragment, got)
		}
	}
	if strings.Contains(got, `<message role="summary">`) || strings.Contains(got, `<summary><text>`) {
		t.Fatalf("expected summary not to render as a message:\n%s", got)
	}
}

type historyStore = sharedHistoryStore

func TestHistorySourceDoesNotDiscardTurnsExcludedFromPrompt(t *testing.T) {
	t.Parallel()

	store := &historyStore{
		state: &history.HistoryState{
			Summary: &history.Summary{
				ID:             "summary-1",
				StartTurnID:    "turn-1",
				EndTurnID:      "turn-2",
				StartTurnCount: 1,
				EndTurnCount:   2,
				Content:        summaryText("summary"),
			},
			Turns: []gaictx.Turn{
				{
					ID:    "turn-3",
					Count: 3,
					UserMessage: &gaictx.StoredMessage{
						Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("hello")},
					},
				},
				{
					ID:    "turn-4",
					Count: 4,
					UserMessage: &gaictx.StoredMessage{
						Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts(strings.Repeat("long ", 100))},
					},
				},
			},
		},
	}

	source := history.NewHistory("session-1", store)
	source.SetTokenCounter(&mocks.MockTokenCounter{})

	part, err := source.Function(context.Background(), 10)
	if err != nil {
		t.Fatalf("Function failed: %v", err)
	}
	if part == nil {
		t.Fatal("expected history part")
	}
	if store.saved != nil {
		t.Fatal("unchanged history was saved")
	}
	if store.state.Summary == nil || store.state.Summary.ID != "summary-1" {
		t.Fatalf("expected summary to be preserved, got %+v", store.state.Summary)
	}
	if len(store.state.Turns) != 2 || store.state.Turns[0].ID != "turn-3" || store.state.Turns[1].ID != "turn-4" {
		t.Fatalf("expected all persisted turns to be preserved, got %+v", store.state.Turns)
	}
}

func TestHistorySourceIncludesNewestFittingTurnsInChronologicalOrder(t *testing.T) {
	t.Parallel()

	store := &historyStore{
		state: &history.HistoryState{
			Turns: []gaictx.Turn{
				{
					ID:       "turn-1",
					Count:    1,
					Messages: []gaictx.StoredMessage{{Message: ai.Message{Role: ai.RoleAssistant, Parts: ai.TextParts(strings.Repeat("oldest ", 100))}}},
				},
				{
					ID:       "turn-2",
					Count:    2,
					Messages: []gaictx.StoredMessage{{Message: ai.Message{Role: ai.RoleAssistant, Parts: ai.TextParts("middle")}}},
				},
				{
					ID:       "turn-3",
					Count:    3,
					Messages: []gaictx.StoredMessage{{Message: ai.Message{Role: ai.RoleAssistant, Parts: ai.TextParts("newest")}}},
				},
			},
		},
	}

	source := history.NewHistory("session-1", store)
	source.SetTokenCounter(&mocks.MockTokenCounter{})

	// Each short message costs one text token and four framing tokens.
	result, err := source.Function(context.Background(), 10)
	if err != nil {
		t.Fatalf("Function failed: %v", err)
	}
	part := result.(*history.Part)
	if len(part.Messages) != 2 || part.Messages[0].Text() != "middle" || part.Messages[1].Text() != "newest" {
		t.Fatalf("expected newest fitting turns in chronological order, got %+v", part.Messages)
	}
	if store.saved != nil {
		t.Fatal("selection persisted history")
	}
	if len(store.state.Turns) != 3 {
		t.Fatalf("expected all persisted turns to be preserved, got %+v", store.state.Turns)
	}
}

func TestNewCompactorRequiresModelOrSummarizer(t *testing.T) {
	t.Parallel()

	_, err := history.NewCompactor("session-1", &historyStore{}, history.CompactorDefinition{})
	if err != history.ErrSummarizerRequired {
		t.Fatalf("expected ErrSummarizerRequired, got %v", err)
	}
}

func TestCompactorDoesNotSummarizeWhenHistoryFitsBudget(t *testing.T) {
	t.Parallel()

	store := &historyStore{
		state: &history.HistoryState{
			Turns: []gaictx.Turn{
				{
					ID:    "turn-1",
					Count: 1,
					UserMessage: &gaictx.StoredMessage{
						Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("first user")},
					},
				},
				{
					ID:    "turn-2",
					Count: 2,
					UserMessage: &gaictx.StoredMessage{
						Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("second user")},
					},
				},
				{
					ID:    "turn-3",
					Count: 3,
					UserMessage: &gaictx.StoredMessage{
						Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("third user")},
					},
				},
			},
		},
	}
	model := &mocks.MockModel{
		Responses: []mocks.MockModelResponse{
			{Res: ai.AIResponse{Message: ai.TextMessage(ai.RoleAssistant, "summary text")}},
		},
	}
	compactor, err := history.NewCompactor("session-1", store, history.CompactorDefinition{
		Amount:       0.67,
		Model:        model,
		TokenCounter: &mocks.MockTokenCounter{},
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	source := history.NewHistory("session-1", store)
	source.SetTokenCounter(&mocks.MockTokenCounter{})

	if _, err := compactor.Compact(t.Context(), 100); err != nil {
		t.Fatal(err)
	}
	part, err := source.Function(context.Background(), 100)
	if err != nil {
		t.Fatalf("Function failed: %v", err)
	}
	if part == nil {
		t.Fatal("expected history part")
	}
	if model.Count != 0 {
		t.Fatalf("expected summarizer not to run, got %d calls", model.Count)
	}
	if store.saved != nil {
		t.Fatal("unchanged history was saved")
	}
	if store.state.Summary != nil {
		t.Fatalf("expected no summary when history fits budget, got %+v", store.state.Summary)
	}
	if len(store.state.Turns) != 3 {
		t.Fatalf("expected all turns to remain unsummarized, got %+v", store.state.Turns)
	}
}

func TestCompactorSummarizesOldestTurnsWhenBudgetReached(t *testing.T) {
	t.Parallel()

	store := &historyStore{
		state: &history.HistoryState{
			Turns: []gaictx.Turn{
				{
					ID:    "turn-1",
					Count: 1,
					UserMessage: &gaictx.StoredMessage{
						Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("first user")},
					},
				},
				{
					ID:    "turn-2",
					Count: 2,
					UserMessage: &gaictx.StoredMessage{
						Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("second user")},
					},
				},
				{
					ID:    "turn-3",
					Count: 3,
					UserMessage: &gaictx.StoredMessage{
						Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("third user")},
					},
				},
			},
		},
	}
	model := &mocks.MockModel{
		Responses: []mocks.MockModelResponse{
			{Res: ai.AIResponse{Message: ai.TextMessage(ai.RoleAssistant, "summary text")}},
		},
	}
	compactor, err := history.NewCompactor("session-1", store, history.CompactorDefinition{
		Amount:       0.67,
		Model:        model,
		TokenCounter: &mocks.MockTokenCounter{},
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	source := history.NewHistory("session-1", store)
	source.SetTokenCounter(&mocks.MockTokenCounter{})

	if _, err := compactor.Compact(t.Context(), 5); err != nil {
		t.Fatal(err)
	}
	part, err := source.Function(context.Background(), 5)
	if err != nil {
		t.Fatalf("Function failed: %v", err)
	}
	if part == nil {
		t.Fatal("expected history part")
	}
	if model.Count != 1 {
		t.Fatalf("expected summarizer to run once, got %d calls", model.Count)
	}
	if store.saved == nil {
		t.Fatal("expected summarized state to be saved")
	}
	if store.saved[0].Summary == nil {
		t.Fatal("expected summary to be saved")
	}
	if got := store.saved[0].Summary.Content.Text; got != "summary text" {
		t.Fatalf("unexpected summary content: %q", got)
	}
	if store.saved[0].Summary.StartTurnID != "turn-1" || store.saved[0].Summary.EndTurnID != "turn-2" {
		t.Fatalf("expected summary to cover first two turns, got %+v", store.saved[0].Summary)
	}
	if len(store.saved[0].Turns) != 1 || store.saved[0].Turns[0].ID != "turn-3" {
		t.Fatalf("expected newest turn to remain unsummarized, got %+v", store.saved[0].Turns)
	}
}

func TestHistorySourceFunctionTable(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                 string
		state                *history.HistoryState
		tokenBudget          int
		summaryDef           *history.CompactorDefinition
		wantPart             bool
		wantSaved            bool
		wantSavedSummary     bool
		wantSavedTurnIDs     []string
		wantSummaryStartTurn string
		wantSummaryEndTurn   string
		wantSummaryContent   string
		wantModelCalls       int
	}{
		{
			name:        "missing state returns empty part",
			state:       nil,
			tokenBudget: 10,
			wantPart:    true,
			wantSaved:   false,
		},
		{
			name: "budget trims turns and preserves summary",
			state: &history.HistoryState{
				Summary: &history.Summary{
					ID:             "summary-1",
					StartTurnID:    "turn-1",
					EndTurnID:      "turn-2",
					StartTurnCount: 1,
					EndTurnCount:   2,
					Content:        summaryText("summary"),
				},
				Turns: []gaictx.Turn{
					{
						ID:    "turn-3",
						Count: 3,
						UserMessage: &gaictx.StoredMessage{
							Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("hello")},
						},
					},
					{
						ID:    "turn-4",
						Count: 4,
						UserMessage: &gaictx.StoredMessage{
							Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts(strings.Repeat("long ", 100))},
						},
					},
				},
			},
			tokenBudget:      10,
			wantPart:         true,
			wantSaved:        false,
			wantSavedSummary: true,
			wantSavedTurnIDs: []string{"turn-3", "turn-4"},
		},
		{
			name: "summary enabled but budget fits skips summarizer",
			state: &history.HistoryState{
				Turns: []gaictx.Turn{
					{
						ID:    "turn-1",
						Count: 1,
						UserMessage: &gaictx.StoredMessage{
							Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("first user")},
						},
					},
					{
						ID:    "turn-2",
						Count: 2,
						UserMessage: &gaictx.StoredMessage{
							Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("second user")},
						},
					},
					{
						ID:    "turn-3",
						Count: 3,
						UserMessage: &gaictx.StoredMessage{
							Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("third user")},
						},
					},
				},
			},
			tokenBudget: 100,
			summaryDef: &history.CompactorDefinition{
				Amount: 0.67,
			},
			wantPart:         true,
			wantSaved:        false,
			wantSavedSummary: false,
			wantSavedTurnIDs: []string{"turn-1", "turn-2", "turn-3"},
			wantModelCalls:   0,
		},
		{
			name: "summary enabled and budget reached summarizes oldest turns",
			state: &history.HistoryState{
				Turns: []gaictx.Turn{
					{
						ID:    "turn-1",
						Count: 1,
						UserMessage: &gaictx.StoredMessage{
							Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("first user")},
						},
					},
					{
						ID:    "turn-2",
						Count: 2,
						UserMessage: &gaictx.StoredMessage{
							Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("second user")},
						},
					},
					{
						ID:    "turn-3",
						Count: 3,
						UserMessage: &gaictx.StoredMessage{
							Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("third user")},
						},
					},
				},
			},
			tokenBudget: 5,
			summaryDef: &history.CompactorDefinition{
				Amount: 0.67,
			},
			wantPart:             true,
			wantSaved:            true,
			wantSavedSummary:     true,
			wantSavedTurnIDs:     []string{"turn-3"},
			wantSummaryStartTurn: "turn-1",
			wantSummaryEndTurn:   "turn-2",
			wantSummaryContent:   "summary text",
			wantModelCalls:       1,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			store := &historyStore{state: tt.state}
			model := &mocks.MockModel{
				Responses: []mocks.MockModelResponse{
					{Res: ai.AIResponse{Message: ai.TextMessage(ai.RoleAssistant, "summary text")}},
				},
			}

			source := history.NewHistory("session-1", store)
			if tt.summaryDef != nil {
				summaryDef := *tt.summaryDef
				summaryDef.Model = model
				summaryDef.TokenCounter = &mocks.MockTokenCounter{}
				compactor, err := history.NewCompactor("session-1", store, summaryDef)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := compactor.Compact(t.Context(), tt.tokenBudget); err != nil {
					t.Fatal(err)
				}
			}

			source.SetTokenCounter(&mocks.MockTokenCounter{})

			part, err := source.Function(context.Background(), tt.tokenBudget)
			if err != nil {
				t.Fatalf("Function failed: %v", err)
			}
			if got := part != nil; got != tt.wantPart {
				t.Fatalf("unexpected part presence: want %v got %v", tt.wantPart, got)
			}
			if got := store.saved != nil; got != tt.wantSaved {
				t.Fatalf("unexpected saved state presence: want %v got %v", tt.wantSaved, got)
			}
			if model.Count != tt.wantModelCalls {
				t.Fatalf("unexpected summarizer calls: want %d got %d", tt.wantModelCalls, model.Count)
			}

			if !tt.wantSaved {
				return
			}
			if got := store.saved[0].Summary != nil; got != tt.wantSavedSummary {
				t.Fatalf("unexpected saved summary presence: want %v got %v", tt.wantSavedSummary, got)
			}
			if tt.wantSavedSummary {
				if tt.wantSummaryStartTurn != "" && store.saved[0].Summary.StartTurnID != tt.wantSummaryStartTurn {
					t.Fatalf("unexpected summary start turn: want %q got %q", tt.wantSummaryStartTurn, store.saved[0].Summary.StartTurnID)
				}
				if tt.wantSummaryEndTurn != "" && store.saved[0].Summary.EndTurnID != tt.wantSummaryEndTurn {
					t.Fatalf("unexpected summary end turn: want %q got %q", tt.wantSummaryEndTurn, store.saved[0].Summary.EndTurnID)
				}
				if got := store.saved[0].Summary.Content.Text; tt.wantSummaryContent != "" && got != tt.wantSummaryContent {
					t.Fatalf("unexpected summary content: want %q got %q", tt.wantSummaryContent, got)
				}
			}

			gotTurnIDs := make([]string, 0, len(store.saved[0].Turns))
			for _, turn := range store.saved[0].Turns {
				gotTurnIDs = append(gotTurnIDs, turn.ID)
			}
			if len(gotTurnIDs) != len(tt.wantSavedTurnIDs) {
				t.Fatalf("unexpected saved turn count: want %d got %d", len(tt.wantSavedTurnIDs), len(gotTurnIDs))
			}
			for i := range gotTurnIDs {
				if gotTurnIDs[i] != tt.wantSavedTurnIDs[i] {
					t.Fatalf("unexpected saved turn id at %d: want %q got %q", i, tt.wantSavedTurnIDs[i], gotTurnIDs[i])
				}
			}
		})
	}
}

func TestDefaultEstimatorCountsCJKHistoryOnDemand(t *testing.T) {
	t.Parallel()
	store := &historyStore{state: &history.HistoryState{Turns: []gaictx.Turn{
		{ID: "old", Count: 1, UserMessage: &gaictx.StoredMessage{Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts(strings.Repeat("界", 40))}}},
		{ID: "new", Count: 2, UserMessage: &gaictx.StoredMessage{Message: ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("ok")}}},
	}}}
	source := history.NewHistory("session", store)
	builder := gaictx.New(gaictx.Definition{TokenBudget: 20, ContextSources: []gaictx.ContextSource{source}})
	if _, err := builder.BuildContext(t.Context()); err != nil {
		t.Fatal(err)
	}
	prompt, err := renderHistoryRequest(builder, t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(prompt, "ok") || strings.Contains(prompt, "界") {
		t.Fatalf("expected newest turn only, got %q", prompt)
	}
	if count, err := store.state.Turns[0].Tokens(t.Context(), builder.TokenCounter()); err != nil || count <= 20 {
		t.Fatalf("old CJK turn was not recounted above budget: %d", count)
	}
	if len(store.state.Turns) != 2 {
		t.Fatal("prompt selection discarded persisted history")
	}
}

func summaryText(text string) ai.ContentPart { return ai.ContentPart{Kind: ai.ContentText, Text: text} }

func renderHistoryRequest(builder *gaictx.Builder, ctx context.Context, conv gaictx.Conversation) (string, error) {
	request, err := builder.BuildRequest(ctx, conv)
	if err != nil {
		return "", err
	}
	return ai.RenderMessages(ctx, request.Messages)
}
