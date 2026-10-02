package history_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/history"
	"github.com/lace-ai/gai/context/tooldefinitions"
)

// Reuse immutable static prompt parts and a persistent history store while
// building a fresh prompt each time, as repeated agent runs do. History includes
// 8 KiB tool results whose outgoing projection is truncated to 500 runes.
func BenchmarkRepeatedPromptBuild(b *testing.B) {
	for _, turnCount := range []int{0, 10, 100} {
		b.Run(fmt.Sprintf("turns_%d", turnCount), func(b *testing.B) {
			ctx := context.Background()
			instructions := gaictx.NewTextPart(strings.Repeat("Use the supplied history and tools to answer accurately. ", 100))
			metadata, err := gaictx.NewJSONPart("metadata", map[string]any{
				"project": "repeatable prompt benchmark",
				"facts":   strings.Repeat("Shared immutable context. ", 100),
			})
			if err != nil {
				b.Fatal(err)
			}
			tools, err := tooldefinitions.New(nil, []gaictx.ToolSignature{benchmarkTool{}}, nil)
			if err != nil {
				b.Fatal(err)
			}
			toolPart, err := tools.Function(ctx, 2000000)
			if err != nil {
				b.Fatal(err)
			}
			store := &benchmarkHistoryStore{state: &history.HistoryState{}}
			for i := 0; i < turnCount; i++ {
				callID := fmt.Sprintf("call-%d", i)
				store.state.Turns = append(store.state.Turns, gaictx.Turn{
					ID: fmt.Sprintf("turn-%d", i), Count: i,
					UserMessage: &gaictx.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "Find the relevant documentation and explain this result.")},
					Messages: []gaictx.StoredMessage{
						{Message: ai.Message{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: callID, Type: "function", Name: "search", Args: []byte(`{"query":"documentation"}`)}}}}},
						{Message: ai.Message{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: callID, Name: "search", Parts: ai.TextParts(strings.Repeat("documentation result ", 400))}}}}},
					},
				})
			}
			input := gaictx.PromptInput{User: ai.TextParts("Give a concise answer with relevant evidence."), Context: []gaictx.Part{metadata, toolPart}}
			build := func() {
				builder := gaictx.New(gaictx.Definition{
					SystemInstructions: []gaictx.Part{instructions},
					ContextSources:     []gaictx.ContextSource{history.NewHistory("benchmark", store)},
					PromptInput:        input, TokenBudget: 2000000,
				})
				if _, err := builder.BuildContext(ctx); err != nil {
					b.Fatal(err)
				}
				request, err := builder.BuildRequest(ctx, nil)
				if err != nil {
					b.Fatal(err)
				}
				if got, want := len(request.Messages), 3*turnCount+4; got != want {
					b.Fatalf("request messages = %d, want %d", got, want)
				}
				benchmarkPromptRequest = request
			}
			build() // Warm the runtime before measuring repeated builds.
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				build()
			}
		})
	}
}

var benchmarkPromptRequest ai.AIRequest

type benchmarkTool struct{}

func (benchmarkTool) Name() string { return "search" }
func (benchmarkTool) Description() string {
	return "Searches project documentation for relevant evidence."
}
func (benchmarkTool) Params() ai.ToolParameters {
	return ai.ToolParameters{Properties: []ai.ToolParameter{{Name: "query", Type: ai.ToolParameterString}}}
}

type benchmarkHistoryStore struct{ state *history.HistoryState }

func (s *benchmarkHistoryStore) GetLastHistoryState(context.Context, string) (*history.HistoryState, error) {
	return s.state, nil
}
func (s *benchmarkHistoryStore) SaveHistoryState(_ context.Context, _ string, state *history.HistoryState) error {
	s.state = state
	return nil
}
