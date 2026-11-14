package anthropic

import "github.com/Samuelk0nrad/gai/ai"

const (
	ClaudeFable5_2           = "claude-fable-5-2"
	ClaudeMythos5_2          = "claude-mythos-5-2"
	ClaudeFable5_1           = "claude-fable-5-1"
	ClaudeMythos5_1          = "claude-mythos-5-1"
	ClaudeFable5             = "claude-fable-5"
	ClaudeMythos5            = "claude-mythos-5"
	ClaudeOpus5_5            = "claude-opus-5-5"
	ClaudeOpus5              = "claude-opus-5"
	ClaudeOpus4_8            = "claude-opus-4-8"
	ClaudeSonnet5_5          = "claude-sonnet-5-5"
	ClaudeSonnet5            = "claude-sonnet-5"
	ClaudeHaiku5_5           = "claude-haiku-5-5"
	ClaudeOpus4_7            = "claude-opus-4-7"
	ClaudeOpus4_6            = "claude-opus-4-6"
	ClaudeSonnet4_6          = "claude-sonnet-4-6"
	ClaudeOpus4_5            = "claude-opus-4-5"
	ClaudeSonnet4_5          = "claude-sonnet-4-5"
	ClaudeHaiku4_5           = "claude-haiku-4-5"
	ClaudeOpus4_1            = "claude-opus-4-1"
	ClaudeOpus4_5_20251101   = "claude-opus-4-5-20251101"
	ClaudeSonnet4_5_20250929 = "claude-sonnet-4-5-20250929"
	ClaudeHaiku4_5_20251001  = "claude-haiku-4-5-20251001"
	ClaudeOpus4_1_20250805   = "claude-opus-4-1-20250805"
)

// thinkingMode is the thinking configuration a model accepts.
type thinkingMode int

const (
	// thinkingUnknown falls back to the descriptor's effort support.
	thinkingUnknown thinkingMode = iota
	// thinkingBudget accepts only {type: "enabled", budget_tokens: N}.
	thinkingBudget
	// thinkingAdaptiveOrBudget prefers adaptive thinking and still accepts a
	// budget (deprecated on Opus 4.6 and Sonnet 4.6).
	thinkingAdaptiveOrBudget
	// thinkingAdaptive accepts only adaptive thinking; a budget returns a 400.
	thinkingAdaptive
)

// modelFacts are the request rules GAI knows for a model without loading the
// live catalog.
type modelFacts struct {
	thinking thinkingMode
	// efforts lists the accepted output_config.effort values. Nil means the
	// model does not accept effort.
	efforts []ai.ReasoningEffort
	// forcedToolChoice reports whether tool_choice "any" and "tool" are accepted.
	forcedToolChoice bool
	// midConversationSystem reports whether role "system" entries are accepted
	// inside messages.
	midConversationSystem bool
	// serverFallbacks reports whether the provider-wide refusal fallbacks
	// option applies. Models without a documented server-side fallback skip it.
	serverFallbacks bool
}

var (
	effortsWithoutXHigh = []ai.ReasoningEffort{ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh, ai.ReasoningEffortMax}
	effortsAll          = []ai.ReasoningEffort{ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh, ai.ReasoningEffortXHigh, ai.ReasoningEffortMax}

	// Models newer than Opus 4.6 that accept forced tool choice.
	adaptiveForced       = modelFacts{thinking: thinkingAdaptive, efforts: effortsAll, forcedToolChoice: true}
	adaptiveForcedSystem = modelFacts{thinking: thinkingAdaptive, efforts: effortsAll, forcedToolChoice: true, midConversationSystem: true}
	// Models with server-side refusal fallbacks.
	adaptiveForcedSystemFallbacks = modelFacts{thinking: thinkingAdaptive, efforts: effortsAll, forcedToolChoice: true, midConversationSystem: true, serverFallbacks: true}
	// Models that return a 400 for tool_choice "any" and "tool".
	adaptiveUnforcedSystem = modelFacts{thinking: thinkingAdaptive, efforts: effortsAll, midConversationSystem: true, serverFallbacks: true}
	adaptiveOrBudget4_6    = modelFacts{thinking: thinkingAdaptiveOrBudget, efforts: effortsWithoutXHigh, forcedToolChoice: true}
	budgetOnly             = modelFacts{thinking: thinkingBudget, forcedToolChoice: true}
)

// knownModels is the maintained Anthropic Messages API catalog, newest first,
// with the request rules GAI knows for each model. It includes current aliases
// and still-supported dated/legacy IDs so callers can pin a model version
// deliberately. Dated IDs share their alias's facts.
var knownModels = []struct {
	id    string
	facts modelFacts
}{
	{ClaudeFable5_2, adaptiveUnforcedSystem},
	{ClaudeMythos5_2, adaptiveUnforcedSystem},
	{ClaudeFable5_1, adaptiveUnforcedSystem},
	{ClaudeMythos5_1, adaptiveUnforcedSystem},
	{ClaudeFable5, adaptiveForcedSystem},
	{ClaudeMythos5, adaptiveForcedSystem},
	{ClaudeOpus5_5, adaptiveUnforcedSystem},
	{ClaudeOpus5, adaptiveForcedSystemFallbacks},
	{ClaudeOpus4_8, adaptiveForcedSystem},
	{ClaudeSonnet5_5, adaptiveUnforcedSystem},
	{ClaudeSonnet5, adaptiveForced},
	{ClaudeHaiku5_5, adaptiveForcedSystem},
	{ClaudeOpus4_7, adaptiveForced},
	{ClaudeOpus4_6, adaptiveOrBudget4_6},
	{ClaudeSonnet4_6, adaptiveOrBudget4_6},
	// Opus 4.5 accepts low/medium/high effort, but GAI pairs effort with
	// thinking and Opus 4.5 has no adaptive thinking, so effort stays off here.
	{ClaudeOpus4_5, budgetOnly},
	{ClaudeSonnet4_5, budgetOnly},
	{ClaudeHaiku4_5, budgetOnly},
	{ClaudeOpus4_1, budgetOnly},
	{ClaudeOpus4_5_20251101, budgetOnly},
	{ClaudeSonnet4_5_20250929, budgetOnly},
	{ClaudeHaiku4_5_20251001, budgetOnly},
	{ClaudeOpus4_1_20250805, budgetOnly},
}

// models lists the catalog IDs in knownModels order.
var models = func() []string {
	ids := make([]string, 0, len(knownModels))
	for _, model := range knownModels {
		ids = append(ids, model.id)
	}
	return ids
}()

// knownModelFacts indexes knownModels by exact model ID.
var knownModelFacts = func() map[string]modelFacts {
	facts := make(map[string]modelFacts, len(knownModels))
	for _, model := range knownModels {
		facts[model.id] = model.facts
	}
	return facts
}()

func lookupModelFacts(model string) (modelFacts, bool) {
	facts, ok := knownModelFacts[model]
	return facts, ok
}
