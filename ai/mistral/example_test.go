package mistral_test

import (
	"context"
	"fmt"
	"os"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/ai/mistral"
)

// Reasoning output arrives as canonical reasoning parts before the answer.
// Replaying response.Message keeps thinking chunks and signatures for the
// next turn.
func ExampleModel_Generate_reasoning() {
	provider := mistral.New(os.Getenv("MISTRAL_API_KEY"), nil)
	model, err := provider.TypedModel(mistral.MistralSmallLatest)
	if err != nil {
		fmt.Println(err)
		return
	}
	history := []ai.Message{ai.TextMessage(ai.RoleUser, "Is 1013 prime?")}
	response, err := model.Generate(context.Background(), ai.AIRequest{
		Messages:  history,
		Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortHigh},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("thinking:", response.Reasoning())
	fmt.Println("answer:", response.Text())
	history = append(history, response.Message, ai.TextMessage(ai.RoleUser, "And 1015?"))
	_ = history
}

// Images keep their position relative to text in a user message.
func ExampleModel_Generate_image() {
	provider := mistral.New(os.Getenv("MISTRAL_API_KEY"), nil)
	model, err := provider.TypedModel(mistral.MistralSmallLatest)
	if err != nil {
		fmt.Println(err)
		return
	}
	pngBytes := []byte{0x89, 'P', 'N', 'G'}
	response, err := model.Generate(context.Background(), ai.AIRequest{Messages: []ai.Message{{
		Role: ai.RoleUser,
		Parts: []ai.ContentPart{
			{Kind: ai.ContentText, Text: "Compare these images."},
			{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/jpeg", URI: "https://example.com/photo.jpg"}},
			{Kind: ai.ContentMedia, Media: &ai.MediaPart{MIMEType: "image/png", Data: pngBytes}},
		},
	}}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(response.Text())
}

// Typed options distinguish explicit zero/false values from omission, and
// Model.With derives per-call settings without mutating the shared model.
func ExampleWithChatCompletionOptions() {
	provider := mistral.New(os.Getenv("MISTRAL_API_KEY"), nil)
	temperature, parallel := 0.0, false
	model, err := provider.TypedModel(mistral.MistralSmallLatest, mistral.WithChatCompletionOptions(mistral.ChatCompletionOptions{
		Temperature:       &temperature,
		ParallelToolCalls: &parallel,
		Stop:              []string{"\n\n"},
		Prediction:        &mistral.Prediction{Content: "func main() {}"},
	}))
	if err != nil {
		fmt.Println(err)
		return
	}
	key := "tenant-42"
	perCall := model.With(mistral.WithChatCompletionOptions(mistral.ChatCompletionOptions{
		Temperature:    &temperature,
		PromptCacheKey: &key,
	}))
	_ = perCall
}
