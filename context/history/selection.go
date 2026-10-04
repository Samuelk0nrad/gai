package history

import (
	"context"
	"fmt"
	"sort"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

type selection struct {
	part                                          Part
	tokens, turnsVisited, turnsIncluded, messages int
	summaryTokens                                 int
	summaryIncluded, budgetReached, summaryFailed bool
	failedTurn, budgetTurn                        *gaictx.Turn
}

// selectHistory only reads a stable state. It owns its sorted slice and returned
// messages, and has no persistence, model-call, or observation side effects.
func selectHistory(ctx context.Context, state *HistoryState, tokenBudget int, counter ai.TokenCounter) (selection, error) {
	var selected selection
	if err := ctx.Err(); err != nil {
		return selected, err
	}
	if state == nil {
		return selected, nil
	}
	turns := sortTurnsByCount(append([]gaictx.Turn(nil), state.Turns...))
	if state.Summary != nil {
		if state.Summary.Content.Kind != ai.ContentText {
			return selected, fmt.Errorf("summary requires a text content part")
		}
		if err := state.Summary.Content.Validate(); err != nil {
			return selected, err
		}
		// Count the same prefixed summary message returned in the prompt.
		summaryParts := ai.TextParts("Conversation summary:\n" + state.Summary.Content.Text)
		if len(state.Summary.Content.Extensions) > 0 {
			// Opaque state belongs to the original part, whose text must stay
			// unchanged. Only plain summaries can coalesce the prefix.
			summaryParts = ai.TextParts("Conversation summary:\n")
			summaryParts = append(summaryParts, ai.CloneParts([]ai.ContentPart{state.Summary.Content})...)
		}
		summaryPart := Part{Messages: []ai.Message{{Role: ai.RoleUser, Parts: summaryParts}}}
		summaryTokenCount, err := summaryPart.Tokens(ctx, counter)
		if err != nil {
			selected.summaryFailed = true
			return selected, err
		}
		if selected.tokens+summaryTokenCount > tokenBudget {
			selected.budgetReached = true
			return selected, nil
		}
		selected.summaryIncluded = true
		selected.part.Messages = append(selected.part.Messages, summaryPart.Messages...)
		selected.tokens += summaryTokenCount
		selected.summaryTokens = summaryTokenCount
	}

	firstIncluded := len(turns)
	budgetReached := false
	for i := len(turns) - 1; i >= 0; i-- {
		turn := &turns[i]
		selected.turnsVisited++
		// Selection and final accounting use the same preview projection. The
		// candidate owns no token cache and only reads the stored messages.
		candidate := Part{}
		if turn.UserMessage != nil {
			candidate.Messages = append(candidate.Messages, turn.UserMessage.Message)
		}
		for _, message := range turn.Messages {
			candidate.Messages = append(candidate.Messages, message.Message)
		}
		tokens, err := candidate.Tokens(ctx, counter)
		if err != nil {
			turnCopy := *turn
			selected.failedTurn = &turnCopy
			return selected, err
		}
		if selected.tokens+tokens > tokenBudget {
			turnCopy := *turn
			selected.budgetTurn = &turnCopy
			budgetReached = true
			break
		}
		selected.tokens += tokens
		firstIncluded = i
		selected.turnsIncluded++
	}

	for _, turn := range turns[firstIncluded:] {
		if turn.UserMessage != nil {
			selected.part.Messages = append(selected.part.Messages, turn.UserMessage.Message.Clone())
			selected.messages++
		}
		for _, message := range turn.Messages {
			selected.part.Messages = append(selected.part.Messages, message.Message.Clone())
			selected.messages++
		}
	}

	selected.budgetReached = budgetReached
	return selected, nil
}

// sortTurnsByCount() Sort turns by Count in ascending order (oldest first)
func sortTurnsByCount(turns []gaictx.Turn) []gaictx.Turn {
	sort.SliceStable(turns, func(i, j int) bool {
		return turns[i].Count < turns[j].Count
	})
	return turns
}
