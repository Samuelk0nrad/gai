package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/ai/openai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

const defaultPrompt = "Where is order LACE-1042, and when should it arrive?"

type orderLookupResult struct {
	Found             bool   `json:"found"`
	OrderID           string `json:"order_id"`
	Status            string `json:"status,omitempty"`
	Carrier           string `json:"carrier,omitempty"`
	TrackingNumber    string `json:"tracking_number,omitempty"`
	EstimatedDelivery string `json:"estimated_delivery,omitempty"`
	LastUpdate        string `json:"last_update,omitempty"`
}

type lookupOrderArgs struct {
	OrderID string `json:"order_id"`
}

type lookupOrderTool struct {
	orders map[string]orderLookupResult
}

func newLookupOrderTool() *lookupOrderTool {
	return &lookupOrderTool{
		orders: map[string]orderLookupResult{
			"LACE-1042": {
				Found:             true,
				OrderID:           "LACE-1042",
				Status:            "in_transit",
				Carrier:           "Austrian Post",
				TrackingNumber:    "AT123456789",
				EstimatedDelivery: "within 2 business days",
				LastUpdate:        "Departed the Vienna logistics center",
			},
		},
	}
}

func (t *lookupOrderTool) Name() string {
	return "lookup_order"
}

func (t *lookupOrderTool) Description() string {
	return "Look up the current shipping status and estimated delivery date for an order ID."
}

func (t *lookupOrderTool) Params() ai.ToolParameters {
	return ai.ToolParameters{
		Strict: true,
		Properties: []ai.ToolParameter{
			{
				Name:        "order_id",
				Type:        ai.ToolParameterString,
				Description: "The order ID, for example LACE-1042.",
				Required:    true,
			},
		},
	}
}

func (t *lookupOrderTool) Function(ctx context.Context, req *ai.ToolCall) *loop.ToolResponse {
	select {
	case <-ctx.Done():
		return loop.NewToolError(ctx.Err())
	default:
	}

	var args lookupOrderArgs
	if err := loop.DecodeToolArgs(req, &args); err != nil {
		return loop.NewToolError(err)
	}

	orderID := strings.ToUpper(strings.TrimSpace(args.OrderID))
	if orderID == "" {
		return loop.NewToolError(fmt.Errorf("order_id must not be empty"))
	}

	result, found := t.orders[orderID]
	if !found {
		result = orderLookupResult{
			Found:   false,
			OrderID: orderID,
		}
	}

	payload, err := json.Marshal(result)
	if err != nil {
		return loop.NewToolError(fmt.Errorf("encode order lookup result: %w", err))
	}
	return loop.NewToolSuccess(string(payload))
}

func main() {
	log.SetFlags(0)
	if err := run(context.Background()); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context) error {
	apiKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	if apiKey == "" {
		return fmt.Errorf("OPENAI_API_KEY is required")
	}

	provider := openai.New(apiKey, nil)
	model, err := provider.Model(envOrDefault("OPENAI_MODEL", "gpt-4.1-mini"))
	if err != nil {
		return fmt.Errorf("create model: %w", err)
	}
	defer func() {
		if err := model.Close(); err != nil {
			log.Printf("close model: %v", err)
		}
	}()

	supportAgent := agent.New(agent.Definition{
		Name:  "order-support",
		Model: model,
		Tools: []loop.Tool{newLookupOrderTool()},
		Limits: agent.Limits{
			MaxLoopIterations: 4,
			MaxTokens:         500,
		},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{
				SystemInstructions: []gaictx.Part{
					gaictx.NewTextPart(
						"You are a concise ecommerce support agent. " +
							"Always use lookup_order for questions about an order. " +
							"Never invent shipping details. Explain the status and next step in one or two sentences.",
					),
				},
			}), nil
		},
	})

	userPrompt := promptFromArgs(os.Args[1:])
	workflow, err := supportAgent.NewRun(ctx, agent.RunInput{
		Prompt: gaictx.PromptInput{
			User: gaictx.NewTextContent(userPrompt),
		},
	})
	if err != nil {
		return fmt.Errorf("create workflow: %w", err)
	}

	fmt.Printf("User: %s\nAssistant: ", userPrompt)

	for event := range workflow.RunEvents(ctx) {
		switch event.Type {
		case agent.EventOutput:
			if text := visibleText(event.Output); text != "" {
				fmt.Print(text)
			}
		}
	}
	fmt.Println()

	if _, err := workflow.Wait(); err != nil {
		return fmt.Errorf("run workflow: %w", err)
	}
	return nil
}

func visibleText(output *agent.OutputPart) string {
	if output == nil || output.Kind != agent.OutputText {
		return ""
	}
	return output.Text
}

func promptFromArgs(args []string) string {
	prompt := strings.TrimSpace(strings.Join(args, " "))
	if prompt == "" {
		return defaultPrompt
	}
	return prompt
}

func envOrDefault(name, fallback string) string {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	return value
}
