package agent

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput"
	"github.com/Samuelk0nrad/gai/modelinput/history"
)

// ErrHistoryCompactionNotConfigurable means automatic history compaction was
// enabled with a prompt builder other than the standard *modelinput.Builder.
var ErrHistoryCompactionNotConfigurable = errors.New("automatic history compaction requires *context.Builder")

// HistoryCompactionFailure selects what a run does when run-start history
// compaction fails.
type HistoryCompactionFailure uint8

const (
	// HistoryCompactionStop ends the run with the compaction error. Remaining
	// pressure after a successful compaction is an error too.
	HistoryCompactionStop HistoryCompactionFailure = iota
	// HistoryCompactionContinue reports the failure and builds the prompt from
	// the uncompacted history, which selection still trims to its allocation.
	// Cancellation and configuration errors still end the run.
	HistoryCompactionContinue
)

// HistoryCompactionPolicy enables persistent history compaction at workflow
// start, before history selection. It requires a *modelinput.Builder and
// history sources backed by a writable history.HistoryStore. Compaction runs
// only under pressure, uses the effective run model and counter and the source's
// remaining allocation, and is skipped when budgets are disabled. It runs at
// most once per workflow and is never retried. Applications still own session
// ordering and accepted-turn writes.
//
// By default the summary continues the run's own prompt: the system prompt,
// the context before the history source, and the full stored history, with the
// run's tools, tool choice and reasoning, followed by one summary instruction.
// That request repeats the previous run's final request, so the provider can
// serve it from its prompt cache while the cache is still valid.
type HistoryCompactionPolicy struct {
	OnFailure HistoryCompactionFailure
	// Summarizer replaces the default continuation summarizer. It must
	// support concurrent use and cannot be set in durable run overrides.
	Summarizer history.Summarizer
	// Amount is the oldest fraction of turns to summarize, in (0,1]. Zero
	// selects the default 0.7.
	Amount float32
	// SummaryMaxTokens limits the summary output; zero inherits the run's.
	SummaryMaxTokens int
	// ResponseFormat selects how the summary treats the run's structured
	// output setting. The zero value omits it.
	ResponseFormat history.SummaryResponseFormat
}

func (p *HistoryCompactionPolicy) validate() error {
	if p == nil {
		return nil
	}
	if p.OnFailure > HistoryCompactionContinue {
		return fmt.Errorf("%w: unknown history compaction failure mode %d", ErrInvalidExecutionConfig, p.OnFailure)
	}
	if math.IsNaN(float64(p.Amount)) || math.IsInf(float64(p.Amount), 0) || p.Amount < 0 || p.Amount > 1 {
		return fmt.Errorf("%w: %w", ErrInvalidExecutionConfig, history.ErrInvalidSummaryAmount)
	}
	if p.SummaryMaxTokens < 0 {
		return fmt.Errorf("%w: %w", ErrInvalidExecutionConfig, history.ErrInvalidSummaryMaxTokens)
	}
	if err := p.ResponseFormat.Validate(); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidExecutionConfig, err)
	}
	if p.Summarizer != nil && nilDependency(p.Summarizer) {
		return fmt.Errorf("%w: history summarizer is a typed nil", ErrInvalidExecutionConfig)
	}
	return nil
}

// HistoryCompactionReport describes one run-start compaction of a history
// source.
type HistoryCompactionReport struct {
	// Source is the context source name.
	Source string
	// Result is the compaction result. When Changed is true, its Revision is
	// the revision compaction committed, which the run then built its prompt
	// from unless another writer changed it first: commit the accepted turn
	// against it. When Changed is false, compaction wrote nothing and the
	// revision from before the run still applies. Reports are not kept across
	// a durable pause.
	Result history.CompactionResult
	// Err is the failure, including history.ErrHistoryPressureRemaining.
	Err error
	// Continued is true when the run went on despite Err.
	Continued bool
}

// The structural capability keeps history sources replaceable in tests and
// adapters. Ordinary context sources and ordinary context builds remain read-only.
type historyCompactionSource interface {
	CompactHistory(context.Context, history.AutoCompaction) (history.CompactionResult, error)
}

// Embedding the concrete builder preserves the loop's budget/counter and
// request-building capabilities. The loop applies its resolved allocation and
// request template before this initial build, outside its model iteration and
// retry loops.
type compactingPromptBuilder struct {
	*modelinput.Builder
	model  ai.Model
	policy HistoryCompactionPolicy
	sink   gai.ObservationSink

	mu       sync.Mutex
	template ai.AIRequest
	budget   ai.RequestBudgetConfig
	reports  []HistoryCompactionReport
}

// SetRequestTemplate implements loop.RequestTemplateSetter.
func (b *compactingPromptBuilder) SetRequestTemplate(settings ai.AIRequest, budget ai.RequestBudgetConfig) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.template, b.budget = settings.Copy(), budget
}

func (b *compactingPromptBuilder) BuildContext(ctx context.Context) ([]modelinput.Part, error) {
	return b.Builder.BuildContextWithPreparation(ctx, func(ctx context.Context, prep modelinput.SourcePreparation) error {
		source, ok := prep.Source.(historyCompactionSource)
		if !ok {
			return nil
		}
		preceding, err := prep.Preceding(ctx)
		if err != nil {
			return err
		}
		b.mu.Lock()
		prefix, budget := b.template.Copy(), b.budget
		b.mu.Unlock()
		prefix.Messages = preceding
		result, err := source.CompactHistory(ctx, history.AutoCompaction{
			HistoryBudget: prep.Remaining, Model: b.model, Counter: prep.Counter,
			Summarizer: b.policy.Summarizer, Amount: b.policy.Amount,
			SummaryMaxTokens: b.policy.SummaryMaxTokens, ResponseFormat: b.policy.ResponseFormat,
			Prefix: prefix, RequestBudget: budget,
		})
		if err == nil && result.PressureRemaining {
			err = history.ErrHistoryPressureRemaining
		}
		report := HistoryCompactionReport{Source: prep.Source.Name(), Result: result, Err: err}
		if err != nil && b.policy.OnFailure == HistoryCompactionContinue && ctx.Err() == nil && !historyConfigurationError(err) {
			report.Continued = true
		}
		b.record(ctx, report)
		if err != nil && !report.Continued {
			return err
		}
		return nil
	})
}

func (b *compactingPromptBuilder) record(ctx context.Context, report HistoryCompactionReport) {
	b.mu.Lock()
	b.reports = append(b.reports, report)
	b.mu.Unlock()
	outcome := "unchanged"
	switch {
	case errors.Is(report.Err, history.ErrHistoryPressureRemaining):
		outcome = "pressure_remaining"
	case report.Err != nil:
		outcome = "failed"
	case report.Result.Changed:
		outcome = "compacted"
	}
	result := report.Result
	gai.EmitObservation(ctx, b.sink, gai.Observation{
		Name: "agent_history_compaction", Source: "agent:history_compaction", Err: report.Err,
		Fields: map[string]any{
			"source": report.Source, "outcome": outcome, "continued": report.Continued,
			"changed": result.Changed, "revision": string(result.Revision),
			"pressure_remaining": result.PressureRemaining, "summarized_turns": result.SummarizedTurns,
			"usage_reported": result.UsageReported, "input_tokens": result.Usage.InputTokens,
			"output_tokens": result.Usage.OutputTokens, "cached_tokens": result.Usage.CachedTokens,
			"cache_creation_tokens": result.Usage.CacheCreationTokens,
		},
	})
}

func (b *compactingPromptBuilder) compactionReports() []HistoryCompactionReport {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]HistoryCompactionReport(nil), b.reports...)
}

// historyConfigurationError reports errors that compaction would hit on every
// run. Continuing past them would silently disable compaction.
func historyConfigurationError(err error) bool {
	for _, target := range []error{
		history.ErrHistoryStoreRequired, history.ErrSummaryPrefixRequired, history.ErrSummarizerRequired,
		history.ErrInvalidSummaryAmount, history.ErrInvalidSummaryMaxTokens, history.ErrInvalidSummaryResponseFormat,
		history.ErrInvalidSummaryPrefix,
	} {
		if errors.Is(err, target) {
			return true
		}
	}
	return false
}
