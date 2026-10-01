package summary

import (
	"context"
	_ "embed"
	"fmt"
	"strings"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

//go:embed system.md
var DefaultSystemPrompt string

type Request struct {
	ID   string
	Text string
	// MaxTokens overrides the summary definition only when positive.
	// Zero and negative values inherit its default.
	MaxTokens int
	Required  bool
	Meta      map[string]any
}

type Config struct {
	SystemPrompt      string
	Tools             []loop.Tool
	MaxLoopIterations int
	MaxTokens         int
	TokenCounter      ai.TokenCounter
	RetryPolicy       *loop.RetryPolicy
}

type Option func(*Config)

func WithSystemPrompt(prompt string) Option {
	return func(config *Config) {
		config.SystemPrompt = prompt
	}
}

func WithTools(tools ...loop.Tool) Option {
	return func(config *Config) {
		config.Tools = tools
	}
}

func WithMaxLoopIterations(maxLoopIterations int) Option {
	return func(config *Config) {
		config.MaxLoopIterations = maxLoopIterations
	}
}

func WithMaxTokens(maxTokens int) Option {
	return func(config *Config) {
		config.MaxTokens = maxTokens
	}
}

// WithRetryPolicy configures opt-in classified retries for summary runs.
func WithRetryPolicy(policy loop.RetryPolicy) Option {
	return func(config *Config) {
		copy := policy
		config.RetryPolicy = &copy
	}
}

func WithTokenCounter(counter ai.TokenCounter) Option {
	return func(config *Config) {
		config.TokenCounter = counter
	}
}

func Definition(model ai.Model, opts ...Option) agent.Definition {
	config := Config{
		SystemPrompt:      DefaultSystemPrompt,
		MaxLoopIterations: 1,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&config)
		}
	}
	systemPrompt := strings.TrimSpace(config.SystemPrompt)
	return agent.Definition{
		Name:  "summary",
		Model: model,
		Tools: config.Tools,
		Prompt: func(ctx context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{
				Renderer:           &gaictx.XMLRenderer{},
				SystemInstructions: []gaictx.Part{gaictx.NewTextPart(systemPrompt)},
				TokenBudget:        -1,
				TokenCounter:       config.TokenCounter,
			}), nil
		},
		TokenCounter: config.TokenCounter,
		RetryPolicy:  config.RetryPolicy,
		Limits: agent.Limits{
			MaxLoopIterations: config.MaxLoopIterations,
			MaxTokens:         config.MaxTokens,
		},
	}
}

func New(model ai.Model, opts ...Option) Summarizer {
	return Summarizer{
		Definition: Definition(model, opts...),
	}
}

type Summarizer struct {
	Definition agent.Definition
}

type activeKey struct{}

func (s Summarizer) Summarize(ctx context.Context, req Request) (string, error) {
	if ctx.Value(activeKey{}) == true {
		return "", fmt.Errorf("%w: recursive summary agent call", gaictx.ErrPromptSource)
	}
	def := s.Definition
	if def.Limits.MaxLoopIterations == 0 {
		def.Limits.MaxLoopIterations = 1
	}
	input := agent.RunInput{
		ID:     req.ID,
		Prompt: gaictx.PromptInput{User: ai.TextParts(req.Text)},
		Meta:   req.Meta,
	}
	// Request.MaxTokens retains its existing non-positive-means-inherit API.
	if req.MaxTokens > 0 {
		input.Execution = &agent.ExecutionOverrides{
			Limits: agent.LimitsOverrides{MaxTokens: &req.MaxTokens},
		}
	}
	workflow, err := agent.New(def).NewRun(ctx, input)
	if err != nil {
		return "", err
	}

	runCtx := context.WithValue(ctx, activeKey{}, true)
	result, err := workflow.Run(runCtx)
	if err != nil {
		return "", err
	}
	if result.Text == "" {
		return "", fmt.Errorf("%w: summary agent produced no text", gaictx.ErrPromptSource)
	}
	return result.Text, nil
}
