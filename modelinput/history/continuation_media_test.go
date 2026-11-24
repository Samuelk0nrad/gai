package history_test

import (
	"context"
	"errors"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput/history"
)

// mediaState is threeTurnState whose first question carries a large file.
func mediaState() *history.HistoryState {
	state := threeTurnState()
	user := state.Turns[0].UserMessage
	user.Message.Parts = append(user.Message.Parts, ai.ContentPart{Kind: ai.ContentMedia, Media: &ai.MediaPart{
		MIMEType: "application/zip", Data: make([]byte, 400_000), Name: "logs.zip", // about 100,000 tokens
	}})
	return state
}

// zipToText replaces every media part with a short text part.
func zipToText(_ context.Context, messages []ai.Message) ([]ai.Message, error) {
	out := make([]ai.Message, len(messages))
	for i, message := range messages {
		out[i] = message.Clone()
		for j, part := range out[i].Parts {
			if part.Kind == ai.ContentMedia {
				out[i].Parts[j] = ai.ContentPart{Kind: ai.ContentText, Text: "<listing of " + part.Media.Name + ">"}
			}
		}
	}
	return out, nil
}

// The budget counts the adapted messages, which are also what the model
// receives, while the stored history is still located in the originals.
func TestContinuationBudgetsTheAdaptedMessages(t *testing.T) {
	state := mediaState()
	budget := history.WithRequestBudget(ai.RequestBudgetConfig{Limit: 3000})

	store := &sharedHistoryStore{state: state.Clone()}
	model := &recordingModel{reply: textReply("summary")}
	_, err := continuationCompactor(t, store, model, history.CompactorDefinition{}).Compact(t.Context(), 10, history.WithPrefix(ref(transcriptFor(state))), budget)
	if !errors.Is(err, history.ErrSummaryRequestTooLarge) {
		t.Fatalf("without an adapter: err = %v", err)
	}

	store = &sharedHistoryStore{state: state.Clone()}
	model = &recordingModel{reply: textReply("summary")}
	summarizer := history.NewContinuationSummarizer(model, history.WithContinuationMessages(zipToText))
	result, err := continuationCompactor(t, store, model, history.CompactorDefinition{Summarizer: summarizer}).Compact(t.Context(), 10, history.WithPrefix(ref(transcriptFor(state))), budget)
	if err != nil || !result.Changed {
		t.Fatalf("result = %+v, err = %v", result, err)
	}
	calls := model.calls()
	if len(calls) != 1 {
		t.Fatalf("calls = %d", len(calls))
	}
	if got := calls[0].Messages[1].Text(); got != state.Turns[0].UserMessage.Message.Text()+"<listing of logs.zip>" {
		t.Fatalf("first question = %q", got)
	}
}

func TestContinuationRejectsAnAdapterThatDropsMessages(t *testing.T) {
	state := threeTurnState()
	store := &sharedHistoryStore{state: state.Clone()}
	model := &recordingModel{reply: textReply("summary")}
	drop := func(_ context.Context, messages []ai.Message) ([]ai.Message, error) { return messages[1:], nil }
	summarizer := history.NewContinuationSummarizer(model, history.WithContinuationMessages(drop))
	_, err := continuationCompactor(t, store, model, history.CompactorDefinition{Summarizer: summarizer}).Compact(t.Context(), 10, history.WithPrefix(ref(transcriptFor(state))))
	if !errors.Is(err, history.ErrInvalidSummaryPrefix) || len(model.calls()) != 0 {
		t.Fatalf("err = %v, calls = %d", err, len(model.calls()))
	}
}
