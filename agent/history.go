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

// ErrHistoryCompactionTimeout ends a run whose compaction failed because the
// summarizer or store hit a deadline it set for itself. The run fails with it
// instead of context.DeadlineExceeded, which would report the run as canceled.
// The HistoryCompactionReport keeps the original error.
var ErrHistoryCompactionTimeout = errors.New("history compaction timed out")

// HistoryCompactionFailure selects what a run does when run-start history
// compaction fails.
type HistoryCompactionFailure uint8

// A compaction that started early, below the allocation, does not end the run
// when it fails without writing: the failure is reported and the run continues
// with the history, which still fits. Once the history changed, the mode
// applies: a committed summary that leaves pressure, or a revision conflict
// because another writer changed it. Under either mode,
// cancellation (the run's context ending, or a context.Canceled error) and
// configuration errors end the run. A deadline that the summarizer or store
// sets for itself is an ordinary failure; when it ends the run, the run fails
// with ErrHistoryCompactionTimeout.
const (
	// HistoryCompactionStop ends the run with the compaction error. Remaining
	// pressure after a successful compaction is an error too.
	HistoryCompactionStop HistoryCompactionFailure = iota
	// HistoryCompactionContinue reports the failure and builds the prompt from
	// the uncompacted history, which selection still trims to its allocation.
	HistoryCompactionContinue
)

// HistoryCompactionPolicy enables persistent history compaction at workflow
// start, before history selection. It requires a *modelinput.Builder and
// history sources backed by a writable history.HistoryStore. Compaction runs
// once the history exceeds Threshold of the source's remaining allocation,
// uses the effective run model and counter, and is skipped when budgets are
// disabled. It runs at most once per workflow and is never retried.
// Applications still own session ordering and accepted-turn writes.
//
// By default the summary continues the run's own prompt: the system prompt,
// the context before the history source, and the full stored history, with the
// run's tools, tool choice and reasoning, followed by one summary instruction.
// That request repeats the previous run's final request, so the provider can
// serve it from its prompt cache while the cache is still valid.
type HistoryCompactionPolicy struct {
	OnFailure HistoryCompactionFailure
	// Threshold is the fraction of the history allocation above which history
	// is compacted, in [0,1]. Zero selects history.DefaultCompactionThreshold
	// (0.85); 1 compacts only history that no longer fits. Compacting early
	// leaves the summary request, which repeats the previous run's prompt and
	// adds an instruction, room within the request budget, so it can reuse the
	// provider's prompt cache instead of falling back to a cut request.
	Threshold float32
	// Summarizer replaces the default continuation summarizer. It must
	// support concurrent use and cannot be set in durable run overrides.
	Summarizer history.Summarizer
	// Amount is the oldest fraction of turns to summarize, in (0,1]. Zero
	// selects the default 0.7.
	Amount float32
	// SummaryMaxTokens limits the summary output. Zero selects the
	// summarizer's default, history.DefaultSummaryMaxTokens for the
	// continuation summarizer; the run's answer limit does not apply.
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
	if math.IsNaN(float64(p.Threshold)) || p.Threshold < 0 || p.Threshold > 1 {
		return fmt.Errorf("%w: %w", ErrInvalidExecutionConfig, history.ErrInvalidCompactionThreshold)
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
	// Continued is true when the run went on despite Err: under
	// HistoryCompactionContinue, or after an early compaction (Result.Early)
	// that wrote nothing and did not conflict.
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
		b.mu.Lock()
		template, budget := b.template.Copy(), b.budget
		b.mu.Unlock()
		// The prefix is rendered only when compaction is needed.
		prefix := func(ctx context.Context) (ai.AIRequest, error) {
			preceding, err := prep.Preceding(ctx)
			if err != nil {
				return ai.AIRequest{}, err
			}
			request := template // a copy, and this runs at most once
			request.Messages = preceding
			return request, nil
		}
		summarizer := b.policy.Summarizer
		if media, ok := b.model.(mediaPolicyModel); ok && summarizer == nil {
			// Convert media before the summary's budget check, so it counts
			// the request the model receives.
			summarizer = history.NewContinuationSummarizer(b.model, history.WithContinuationMessages(media.media.Messages))
		}
		result, err := source.CompactHistory(ctx, history.AutoCompaction{
			HistoryBudget: prep.Remaining, Threshold: b.policy.Threshold, Model: b.model, Counter: prep.Counter,
			Summarizer: summarizer, Amount: b.policy.Amount,
			SummaryMaxTokens: b.policy.SummaryMaxTokens, ResponseFormat: b.policy.ResponseFormat,
			Prefix: prefix, RequestBudget: budget,
		})
		if err == nil && result.PressureRemaining {
			err = history.ErrHistoryPressureRemaining
		}
		report := HistoryCompactionReport{Source: prep.Source.Name(), Result: result, Err: err}
		if err != nil && (b.policy.OnFailure == HistoryCompactionContinue || earlyFailure(result, err)) && !canceled(ctx, err) && !historyConfigurationError(err) {
			report.Continued = true
		}
		b.record(ctx, report)
		if err != nil && !report.Continued {
			if !canceled(ctx, err) && errors.Is(err, context.DeadlineExceeded) {
				// The loop reports any deadline error as cancellation.
				return fmt.Errorf("%w: %s: %v", ErrHistoryCompactionTimeout, report.Source, err)
			}
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
			"pressure_remaining": result.PressureRemaining, "early": result.Early, "summarized_turns": result.SummarizedTurns,
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

// earlyFailure reports a failed compaction that started while the history still
// fit and wrote nothing, which the run can continue past. After a commit or a
// conflict, the history changed and may no longer fit: a committed summary can
// leave pressure, and a conflict means another writer changed it.
func earlyFailure(result history.CompactionResult, err error) bool {
	var conflict *history.RevisionConflictError
	return result.Early && !result.Changed && !errors.As(err, &conflict)
}

// canceled reports whether the run was canceled. A summarizer or store can
// return context.Canceled from a context of its own; a deadline it set for
// itself is a compaction failure, not a cancellation.
func canceled(ctx context.Context, err error) bool {
	return ctx.Err() != nil || errors.Is(err, context.Canceled)
}

// historyConfigurationError reports errors that compaction would hit on every
// run. Continuing past them would silently disable compaction.
func historyConfigurationError(err error) bool {
	for _, target := range []error{
		history.ErrHistoryStoreRequired, history.ErrSummaryPrefixRequired, history.ErrSummarizerRequired,
		history.ErrInvalidSummaryAmount, history.ErrInvalidSummaryMaxTokens, history.ErrInvalidSummaryResponseFormat,
		history.ErrInvalidSummaryPrefix, history.ErrInvalidCompactionThreshold,
	} {
		if errors.Is(err, target) {
			return true
		}
	}
	return false
}

// mediaPolicyModel applies the agent's media policy to requests that bypass
// the loop, such as history-compaction summaries. The default continuation
// summarizer converts media itself, before its budget check; this wrapper
// covers custom summarizers that call the model.
type mediaPolicyModel struct {
	ai.Model
	media *ai.MediaAdapter
}

func (m mediaPolicyModel) GenerateStream(ctx context.Context, request ai.AIRequest) <-chan ai.Token {
	messages, err := m.media.Messages(ctx, request.Messages)
	if err != nil {
		tokens := make(chan ai.Token, 1)
		tokens <- ai.Token{Err: err}
		close(tokens)
		return tokens
	}
	request.Messages = messages
	return m.Model.GenerateStream(ctx, request)
}
