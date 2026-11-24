package history

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput"
)

// DefaultContinuationInstructions is the summary guidance appended after the
// continued conversation. GAI adds the scope and summary-merge sentences.
const DefaultContinuationInstructions = `Preserve facts, decisions, constraints, names, IDs, results of tool calls that still matter, unresolved questions, and user intent.
Remove repetition and incidental wording.
Do not invent details.
Leave out system instructions, tool definitions, and reference context that is not part of the conversation.
Return only the summary text.`

const (
	quoteMinRunes  = 60
	quoteStepRunes = 20
	quoteMaxRunes  = 300
)

// ContinuationSummarizer summarizes history by continuing a request that
// already contains it. It sends the request unchanged (messages, tools, tool
// choice and reasoning) and appends one user message with the summary
// instruction, so the provider can serve the shared prefix from its prompt
// cache. Use the same model that produced the request: replayed provider state,
// such as reasoning signatures, belongs to that model.
//
// The summary call is one generation with no retries. A reply that calls a
// tool, is empty, or stops at the output limit is an error.
type ContinuationSummarizer struct {
	model        ai.Model
	instructions string
}

// ContinuationOption configures a ContinuationSummarizer.
type ContinuationOption func(*ContinuationSummarizer)

// WithContinuationInstructions replaces DefaultContinuationInstructions. GAI
// still adds which messages to cover and how to treat the earlier summary.
func WithContinuationInstructions(instructions string) ContinuationOption {
	return func(s *ContinuationSummarizer) { s.instructions = strings.TrimSpace(instructions) }
}

// NewContinuationSummarizer creates a summarizer that continues the request
// passed to Compact with WithPrefix.
func NewContinuationSummarizer(model ai.Model, opts ...ContinuationOption) *ContinuationSummarizer {
	s := &ContinuationSummarizer{model: model, instructions: DefaultContinuationInstructions}
	for _, opt := range opts {
		if opt != nil {
			opt(s)
		}
	}
	return s
}

// SummarizeHistory implements Summarizer.
//
// It first tries the full prefix with an instruction that quotes the start of
// the first kept user message, which reuses the whole cached prompt. When that
// message cannot be quoted unambiguously, or the request exceeds the request
// budget, it cuts the prefix after the last folded turn and summarizes all of
// it. When that still does not fit, it folds the most turns that fit, at least
// one.
func (s *ContinuationSummarizer) SummarizeHistory(ctx context.Context, input SummaryInput) (SummaryOutput, error) {
	if s == nil || s.model == nil {
		return SummaryOutput{}, ErrSummarizerMissing
	}
	if input.Prefix == nil {
		return SummaryOutput{}, ErrSummaryPrefixRequired
	}
	if len(input.Turns) == 0 {
		return SummaryOutput{}, fmt.Errorf("%w: no turns to fold", ErrInvalidSummaryPrefix)
	}
	if err := input.ResponseFormat.Validate(); err != nil {
		return SummaryOutput{}, err
	}
	base, err := s.baseRequest(*input.Prefix, input)
	if err != nil {
		return SummaryOutput{}, err
	}
	firstKept := firstTurn(input.Kept)
	layout, err := locateHistory(base.Messages, input.PreviousSummary, input.Turns, firstKept)
	if err != nil {
		return SummaryOutput{}, err
	}
	counter := input.Counter
	if counter == nil {
		counter = ai.TextTokenEstimator{}
	}
	merge := input.PreviousSummary != nil

	// Full prefix: the cache-friendly request.
	if quote, ok := boundaryQuote(base.Messages, layout.firstKeptUser); ok {
		request := withInstruction(base, len(base.Messages), s.instruction(quote, merge))
		fits, err := fitsBudget(ctx, request, input.RequestBudget, counter, input.MaxTokens > 0)
		if err != nil {
			return SummaryOutput{}, err
		}
		if fits {
			return s.generate(ctx, request, 0)
		}
	}
	// Cut prefixes: fold every turn, else the most turns that fit, oldest
	// first. A cut request grows with the turns it folds, so a binary search
	// finds that count with a logarithmic number of estimates.
	cut := func(folded int) (ai.AIRequest, bool, error) {
		request := withInstruction(base, layout.turnEnds[folded-1], s.instruction("", merge))
		fits, err := fitsBudget(ctx, request, input.RequestBudget, counter, input.MaxTokens > 0)
		return request, fits, err
	}
	request, fits, err := cut(len(input.Turns))
	if err != nil {
		return SummaryOutput{}, err
	}
	if fits {
		return s.generate(ctx, request, 0)
	}
	// Invariant: lo turns fit (or lo is zero) and hi turns do not.
	lo, hi := 0, len(input.Turns)
	var best ai.AIRequest
	for lo+1 < hi {
		mid := lo + (hi-lo)/2
		request, fits, err := cut(mid)
		if err != nil {
			return SummaryOutput{}, err
		}
		if fits {
			lo, best = mid, request
		} else {
			hi = mid
		}
	}
	if lo == 0 {
		return SummaryOutput{}, ErrSummaryRequestTooLarge
	}
	return s.generate(ctx, best, lo)
}

// baseRequest applies the summary settings that do not depend on the cut.
func (s *ContinuationSummarizer) baseRequest(prefix ai.AIRequest, input SummaryInput) (ai.AIRequest, error) {
	request := prefix.Copy()
	if len(request.Messages) == 0 {
		return ai.AIRequest{}, fmt.Errorf("%w: no messages", ErrInvalidSummaryPrefix)
	}
	if request.Messages[len(request.Messages)-1].Role == ai.RoleSystem {
		return ai.AIRequest{}, fmt.Errorf("%w: prefix ends with a system message", ErrInvalidSummaryPrefix)
	}
	if request.ToolChoice.Mode == ai.ToolChoiceRequired {
		// A required call cannot produce a text summary. A run normally ends
		// with the requirement satisfied and the choice relaxed to auto; one
		// whose provider ignored the requirement still has it. Changing the
		// choice costs the cached messages on some providers.
		request.ToolChoice = ai.ToolChoice{Mode: ai.ToolChoiceAuto}
	}
	structured := request.ResponseFormat.Type != "" && request.ResponseFormat.Type != ai.ResponseFormatText
	switch input.ResponseFormat {
	case SummaryResponseFormatOmit:
		request.ResponseFormat = ai.ResponseFormat{}
	case SummaryResponseFormatReject:
		if structured {
			return ai.AIRequest{}, fmt.Errorf("%w: prefix requests structured output", ErrInvalidSummaryPrefix)
		}
	}
	if input.MaxTokens > 0 {
		request.MaxTokens = input.MaxTokens
		if request.Reasoning.BudgetTokens > 0 {
			// Providers reject a thinking budget that is not below max tokens.
			// The summary limit applies to the visible text.
			request.MaxTokens += request.Reasoning.BudgetTokens
		}
	}
	return request, nil
}

func (s *ContinuationSummarizer) instruction(quote string, merge bool) string {
	var b strings.Builder
	b.WriteString("Stop the conversation here. Do not call any tool and do not reply to the last message.\n\n")
	b.WriteString("Write a summary that another agent will use in place of ")
	if quote != "" {
		b.WriteString("every message before the user message that begins: ")
		b.WriteString(strconv.Quote(quote))
		b.WriteString(". Do not summarize that message or anything after it.")
	} else {
		b.WriteString("the whole conversation above.")
	}
	if merge {
		b.WriteString(" The conversation starts with an earlier summary, the message beginning \"Conversation summary:\". Merge it into the new summary so nothing in it is lost.")
	}
	if s.instructions != "" {
		b.WriteString("\n\n")
		b.WriteString(s.instructions)
	}
	return b.String()
}

func (s *ContinuationSummarizer) generate(ctx context.Context, request ai.AIRequest, folded int) (SummaryOutput, error) {
	if err := request.Validate(); err != nil {
		return SummaryOutput{}, fmt.Errorf("%w: %w", ErrInvalidSummaryPrefix, err)
	}
	var response ai.AIResponse
	output := SummaryOutput{FoldedTurns: folded}
	var streamErr error
	for token := range s.model.GenerateStream(ctx, request) {
		if token.Err != nil {
			if streamErr == nil {
				streamErr = token.Err
			}
			continue
		}
		if token.Completion != nil && token.Completion.UsageReported {
			output.Usage = token.Completion.Usage
			output.UsageReported = true
		}
		response.AppendToken(token)
	}
	if streamErr != nil {
		return output, streamErr
	}
	if err := ctx.Err(); err != nil {
		return output, err
	}
	for _, part := range response.Message.Parts {
		if part.Kind == ai.ContentToolCall || part.Kind == ai.ContentToolResult {
			return output, fmt.Errorf("%w: the model called a tool", ErrInvalidSummaryReply)
		}
	}
	if truncatedFinish(response.FinishReason) {
		return output, fmt.Errorf("%w: summary stopped at the output limit (%s)", ErrInvalidSummaryReply, response.FinishReason)
	}
	// Text includes JSON parts, the reply of a kept structured format.
	output.Text = strings.TrimSpace(response.Message.Text())
	if output.Text == "" {
		return output, fmt.Errorf("%w: empty summary", ErrInvalidSummaryReply)
	}
	return output, nil
}

func truncatedFinish(reason string) bool {
	switch strings.ToLower(reason) {
	case "max_tokens", "length", "max_output_tokens":
		return true
	}
	return false
}

// withInstruction shares the messages with base, which the summarizer owns and
// never modifies, so trying several cuts copies no message content.
func withInstruction(base ai.AIRequest, end int, instruction string) ai.AIRequest {
	request := base
	messages := make([]ai.Message, end, end+1)
	copy(messages, base.Messages[:end])
	request.Messages = append(messages, ai.TextMessage(ai.RoleUser, instruction))
	return request
}

// boundaryQuote returns the shortest normalized start of the first kept user
// message, from quoteMinRunes, that no other user message in the request
// starts with. ok is false when there is no such quote.
func boundaryQuote(messages []ai.Message, index int) (string, bool) {
	if index < 0 {
		return "", false
	}
	target := []rune(normalizeQuoteText(messages[index].Text()))
	if len(target) == 0 {
		return "", false
	}
	var others []string
	for i, message := range messages {
		if i != index && message.Role == ai.RoleUser {
			others = append(others, normalizeQuoteText(message.Text()))
		}
	}
	for n := quoteMinRunes; ; n += quoteStepRunes {
		n = min(n, len(target), quoteMaxRunes)
		quote := string(target[:n])
		unique := true
		for _, other := range others {
			if strings.HasPrefix(other, quote) {
				unique = false
				break
			}
		}
		if unique {
			return quote, true
		}
		if n == len(target) || n == quoteMaxRunes {
			return "", false
		}
	}
}

func normalizeQuoteText(text string) string {
	return strings.Join(strings.Fields(text), " ")
}

// fitsBudget estimates the request locally. Accurate counting mode is not
// used for summary requests; a zero budget skips the check. With an explicit
// summary limit, only that limit is reserved for output, not the run's reserve.
func fitsBudget(ctx context.Context, request ai.AIRequest, budget ai.RequestBudgetConfig, counter ai.TokenCounter, ownLimit bool) (bool, error) {
	if budget.Limit == 0 && budget.InputLimit == 0 {
		return true, nil
	}
	estimate, err := ai.EstimateRequestTokens(ctx, request, counter)
	if err != nil {
		return false, err
	}
	input := estimate.InputTokens + budget.SafetyMargin
	reserve := max(budget.OutputReserve, request.MaxTokens)
	if ownLimit {
		reserve = request.MaxTokens
	}
	total := input + reserve
	return (budget.Limit == 0 || total <= budget.Limit) && (budget.InputLimit == 0 || input <= budget.InputLimit), nil
}

func firstTurn(turns []modelinput.Turn) *modelinput.Turn {
	if len(turns) == 0 {
		return nil
	}
	return &turns[0]
}
