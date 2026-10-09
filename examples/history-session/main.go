// Run with: go run ./examples/history-session
// This example is deterministic and makes no network calls.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/context/history"
)

// textModel implements the same streaming boundary as a provider-backed model.
// The callback must be concurrency-safe when the service runs several sessions.
type textModel func(context.Context, ai.AIRequest) (string, error)

func (m textModel) GenerateStream(ctx context.Context, request ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)
	go func() {
		defer close(out)
		text, err := m(ctx, request)
		if err != nil {
			ai.SendToken(ctx, out, ai.Token{Err: err})
			return
		}
		part := ai.ContentPart{Kind: ai.ContentText, Text: text}
		ai.SendToken(ctx, out, ai.Token{Part: &part})
	}()
	return out
}

func main() {
	ctx := context.Background()
	store := &memoryStore{}
	initial := &history.HistoryState{SchemaVersion: history.HistorySchemaVersion}
	initial.Turns = append(initial.Turns, completedTurn("demo", 1, []ai.Message{
		ai.TextMessage(ai.RoleUser, "I prefer Go. "+strings.Repeat("Earlier context. ", 100)),
		ai.TextMessage(ai.RoleAssistant, "I will remember your preference."),
	}))
	if _, err := store.CompareAndSwapHistory(ctx, "demo", "", initial); err != nil {
		log.Fatal(err)
	}
	service := &chatService{
		store: store, requestWindow: 256,
		compaction: &history.CompactorDefinition{Amount: 1, Model: textModel(func(context.Context, ai.AIRequest) (string, error) {
			return "The user prefers Go.", nil
		})},
		model: textModel(func(_ context.Context, request ai.AIRequest) (string, error) {
			previous := 0
			for _, message := range request.Messages {
				if message.Role == ai.RoleAssistant {
					previous++
				}
			}
			return fmt.Sprintf("Answer based on %d previous accepted assistant messages.", previous), nil
		}),
	}
	for _, input := range []string{"Which language should I use?", "And for a small service?"} {
		result, err := service.Run(ctx, "demo", input)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(result.Text)
	}
	snapshot, err := store.LoadHistory(ctx, "demo")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Stored revision %s: summary=%t, completed turns=%d\n", snapshot.Revision, snapshot.State.Summary != nil, len(snapshot.State.Turns))
}
