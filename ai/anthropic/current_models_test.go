package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
	antropic "github.com/anthropics/anthropic-sdk-go"
)

func emptyMessage(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"content":[],"usage":{}}`))
}

func modelFor(t *testing.T, name string, handler http.HandlerFunc, options ...Option) *Model {
	t.Helper()
	m := testModel(t, handler)
	for _, option := range options {
		option(m.client)
	}
	m.name = name
	return m
}

func unexpectedRequest(t *testing.T) http.HandlerFunc {
	return func(http.ResponseWriter, *http.Request) { t.Error("unexpected provider request") }
}

func userText(text string) []ai.Message { return []ai.Message{ai.TextMessage(ai.RoleUser, text)} }

var searchTool = []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object"}`)}}

func TestKnownModelsHaveLocalFacts(t *testing.T) {
	if len(models) != len(knownModelFacts) {
		t.Errorf("catalog has %d models, facts table %d; duplicate ID?", len(models), len(knownModelFacts))
	}
	for _, model := range []string{ClaudeOpus5_5, ClaudeOpus5, ClaudeSonnet5_5, ClaudeHaiku5_5, ClaudeFable5_1, ClaudeFable5_2, ClaudeMythos5_1, ClaudeMythos5_2} {
		if _, ok := lookupModelFacts(model); !ok {
			t.Errorf("%s missing", model)
		}
	}
}

func TestLocalDescriptorsWithoutCatalog(t *testing.T) {
	all := effortsAll
	forced := []ai.ToolChoiceMode{ai.ToolChoiceAuto, ai.ToolChoiceNone, ai.ToolChoiceRequired}
	unforced := []ai.ToolChoiceMode{ai.ToolChoiceAuto, ai.ToolChoiceNone}
	tests := []struct {
		model   string
		effort  ai.FeatureSupport
		efforts []ai.ReasoningEffort
		choices []ai.ToolChoiceMode
	}{
		{ClaudeOpus5_5, ai.FeatureSupportSupported, all, unforced},
		{ClaudeSonnet5_5, ai.FeatureSupportSupported, all, unforced},
		{ClaudeFable5_1, ai.FeatureSupportSupported, all, unforced},
		{ClaudeFable5_2, ai.FeatureSupportSupported, all, unforced},
		{ClaudeMythos5_1, ai.FeatureSupportSupported, all, unforced},
		{ClaudeMythos5_2, ai.FeatureSupportSupported, all, unforced},
		{ClaudeOpus5, ai.FeatureSupportSupported, all, forced},
		{ClaudeHaiku5_5, ai.FeatureSupportSupported, all, forced},
		{ClaudeFable5, ai.FeatureSupportSupported, all, forced},
		{ClaudeMythos5, ai.FeatureSupportSupported, all, forced},
		{ClaudeOpus4_8, ai.FeatureSupportSupported, all, forced},
		{ClaudeSonnet5, ai.FeatureSupportSupported, all, forced},
		{ClaudeOpus4_7, ai.FeatureSupportSupported, all, forced},
		{ClaudeOpus4_6, ai.FeatureSupportSupported, effortsWithoutXHigh, forced},
		{ClaudeSonnet4_6, ai.FeatureSupportSupported, effortsWithoutXHigh, forced},
		{ClaudeSonnet4_5, ai.FeatureSupportUnsupported, nil, forced},
		{ClaudeHaiku4_5_20251001, ai.FeatureSupportUnsupported, nil, forced},
	}
	p := New("test-key", nil)
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			m, err := p.TypedModel(tt.model)
			if err != nil {
				t.Fatal(err)
			}
			d := m.Descriptor()
			if d.Reasoning != ai.FeatureSupportSupported || d.ReasoningEffort != tt.effort || !reflect.DeepEqual(d.ReasoningEfforts, tt.efforts) || !reflect.DeepEqual(d.ToolChoiceModes, tt.choices) {
				t.Fatalf("descriptor = %+v", d)
			}
		})
	}
}

func TestAdaptiveOnlyModelsSendAdaptiveThinking(t *testing.T) {
	for _, model := range []string{ClaudeOpus4_7, ClaudeOpus4_8, ClaudeSonnet5, ClaudeFable5, ClaudeMythos5, ClaudeOpus5_5, ClaudeOpus5, ClaudeSonnet5_5, ClaudeHaiku5_5, ClaudeFable5_2} {
		t.Run(model, func(t *testing.T) {
			var got map[string]any
			m := modelFor(t, model, func(w http.ResponseWriter, r *http.Request) {
				got = decodeRequest(t, r)
				emptyMessage(w)
			})
			if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi"), Reasoning: ai.ReasoningConfig{Enabled: true, IncludeThoughts: true}}); err != nil {
				t.Fatal(err)
			}
			if thinking := object(t, got["thinking"]); thinking["type"] != "adaptive" || thinking["display"] != "summarized" {
				t.Fatalf("thinking = %#v", thinking)
			}
			if _, ok := got["output_config"]; ok {
				t.Fatalf("output_config = %#v, want none without effort", got["output_config"])
			}
		})
	}
}

func TestBudgetTokensOnAdaptiveOnlyModelFailsLocally(t *testing.T) {
	for _, model := range []string{ClaudeOpus4_7, ClaudeOpus5_5, ClaudeFable5, ClaudeSonnet5_5} {
		m := modelFor(t, model, unexpectedRequest(t))
		_, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi"), MaxTokens: 8000, Reasoning: ai.ReasoningConfig{Enabled: true, BudgetTokens: 2000}})
		if !errors.Is(err, ai.ErrUnsupportedCapability) || !strings.Contains(err.Error(), "only adaptive thinking") {
			t.Fatalf("%s error = %v", model, err)
		}
	}
	// Opus 4.6 still accepts a budget.
	var got map[string]any
	m := modelFor(t, ClaudeOpus4_6, func(w http.ResponseWriter, r *http.Request) {
		got = decodeRequest(t, r)
		emptyMessage(w)
	})
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi"), MaxTokens: 8000, Reasoning: ai.ReasoningConfig{Enabled: true, BudgetTokens: 2000}}); err != nil {
		t.Fatal(err)
	}
	if thinking := object(t, got["thinking"]); thinking["type"] != "enabled" || thinking["budget_tokens"] != float64(2000) {
		t.Fatalf("Opus 4.6 thinking = %#v", thinking)
	}
	// Budget-only models need a budget.
	m = modelFor(t, ClaudeSonnet4_5, unexpectedRequest(t))
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi"), Reasoning: ai.ReasoningConfig{Enabled: true}}); !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("Sonnet 4.5 without budget error = %v", err)
	}
}

func TestEffortLevelsXHighAndMax(t *testing.T) {
	for _, effort := range []ai.ReasoningEffort{ai.ReasoningEffortXHigh, ai.ReasoningEffortMax} {
		var got map[string]any
		m := modelFor(t, ClaudeOpus5_5, func(w http.ResponseWriter, r *http.Request) {
			got = decodeRequest(t, r)
			emptyMessage(w)
		})
		if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi"), Reasoning: ai.ReasoningConfig{Effort: effort}}); err != nil {
			t.Fatal(err)
		}
		if object(t, got["output_config"])["effort"] != string(effort) || object(t, got["thinking"])["type"] != "adaptive" {
			t.Fatalf("%s request = %#v", effort, got)
		}
	}
	m := modelFor(t, ClaudeOpus4_6, unexpectedRequest(t))
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi"), Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortXHigh}}); !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("xhigh on Opus 4.6 error = %v", err)
	}
}

func TestForcedToolChoiceOnlyWhereAccepted(t *testing.T) {
	required := ai.ToolChoice{Mode: ai.ToolChoiceRequired}
	for _, model := range []string{ClaudeOpus5_5, ClaudeSonnet5_5, ClaudeFable5_1, ClaudeFable5_2, ClaudeMythos5_1, ClaudeMythos5_2} {
		m := modelFor(t, model, unexpectedRequest(t))
		if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi"), Tools: searchTool, ToolChoice: required}); !errors.Is(err, ai.ErrUnsupportedCapability) {
			t.Fatalf("%s forced tool choice error = %v", model, err)
		}
	}
	var got map[string]any
	m := modelFor(t, ClaudeOpus5, func(w http.ResponseWriter, r *http.Request) {
		got = decodeRequest(t, r)
		emptyMessage(w)
	})
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi"), Tools: searchTool, ToolChoice: required}); err != nil {
		t.Fatal(err)
	}
	if object(t, got["tool_choice"])["type"] != "any" {
		t.Fatalf("tool_choice = %#v", got["tool_choice"])
	}
}

func TestCatalogParserReadsAllEffortsAndThinkingTypes(t *testing.T) {
	var info antropic.ModelInfo
	raw := `{"id":"claude-next","capabilities":{"thinking":{"supported":true,"types":{"adaptive":{"supported":true},"enabled":{"supported":false},"disabled":{"supported":false},"between_tools":{"supported":false}}},"effort":{"supported":true,"low":{"supported":true},"medium":{"supported":true},"high":{"supported":true},"xhigh":{"supported":true},"max":{"supported":true}}}}`
	if err := json.Unmarshal([]byte(raw), &info); err != nil {
		t.Fatal(err)
	}
	facts := anthropicCatalogFacts(info.ID, info)
	if !reflect.DeepEqual(facts.ReasoningEfforts, effortsAll) {
		t.Fatalf("catalog efforts = %v", facts.ReasoningEfforts)
	}
	if mode := anthropicCatalogThinking(info); mode != thinkingAdaptive {
		t.Fatalf("catalog thinking = %v", mode)
	}
	// A catalog that predates xhigh and max keeps its low/medium/high list.
	older := `{"id":"claude-old","capabilities":{"effort":{"supported":true,"low":{"supported":true},"medium":{"supported":true},"high":{"supported":true}}}}`
	var old antropic.ModelInfo
	if err := json.Unmarshal([]byte(older), &old); err != nil {
		t.Fatal(err)
	}
	want := []ai.ReasoningEffort{ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh}
	if facts := anthropicCatalogFacts(old.ID, old); !reflect.DeepEqual(facts.ReasoningEfforts, want) || anthropicCatalogThinking(old) != thinkingUnknown {
		t.Fatalf("older catalog facts = %+v", facts)
	}
	// Without all of low, medium and high no restrictive list is published.
	partial := `{"id":"claude-partial","capabilities":{"effort":{"supported":true,"low":{"supported":true},"xhigh":{"supported":true}}}}`
	var part antropic.ModelInfo
	if err := json.Unmarshal([]byte(partial), &part); err != nil {
		t.Fatal(err)
	}
	if facts := anthropicCatalogFacts(part.ID, part); facts.ReasoningEfforts != nil {
		t.Fatalf("partial catalog facts = %+v", facts)
	}
	// The older list is enforced locally: xhigh never reaches the API.
	m := modelFor(t, old.ID, unexpectedRequest(t))
	m.client.catalog.Replace([]ai.ModelDescriptor{anthropicCatalogFacts(old.ID, old)})
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi"), Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortXHigh}}); !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("xhigh on older catalog model error = %v", err)
	}
}

func TestCatalogThinkingTypesSelectMode(t *testing.T) {
	var got map[string]any
	m := modelFor(t, "claude-dynamic", func(w http.ResponseWriter, r *http.Request) {
		got = decodeRequest(t, r)
		emptyMessage(w)
	})
	// The catalog says effort works but only budget thinking is accepted.
	m.client.catalog.Replace([]ai.ModelDescriptor{{Model: m.name, Reasoning: ai.FeatureSupportSupported, ReasoningEffort: ai.FeatureSupportSupported}})
	modes := map[string]thinkingMode{m.name: thinkingBudget}
	m.client.catalogThinking.Store(&modes)
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi"), Reasoning: ai.ReasoningConfig{Enabled: true}}); !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("budget-only catalog model error = %v", err)
	}
	// The descriptor agrees: effort is not advertised for a budget-only model,
	// even when the catalog reports it, local facts included.
	if d := m.Descriptor(); d.ReasoningEffort != ai.FeatureSupportUnsupported || d.ReasoningEfforts != nil {
		t.Fatalf("budget-only descriptor = %+v", d)
	}
	opus45 := modelFor(t, ClaudeOpus4_5, unexpectedRequest(t))
	opus45.client.catalog.Replace([]ai.ModelDescriptor{{Model: ClaudeOpus4_5, Reasoning: ai.FeatureSupportSupported, ReasoningEffort: ai.FeatureSupportSupported, ReasoningEfforts: []ai.ReasoningEffort{ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh}}})
	if d := opus45.Descriptor(); d.ReasoningEffort != ai.FeatureSupportUnsupported {
		t.Fatalf("Opus 4.5 descriptor = %+v", d)
	}
	modes = map[string]thinkingMode{m.name: thinkingAdaptive}
	m.client.catalogThinking.Store(&modes)
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi"), Reasoning: ai.ReasoningConfig{Enabled: true}}); err != nil {
		t.Fatal(err)
	}
	if object(t, got["thinking"])["type"] != "adaptive" {
		t.Fatalf("thinking = %#v", got["thinking"])
	}
}

func TestMidConversationSystemMessages(t *testing.T) {
	conversation := []ai.Message{
		ai.TextMessage(ai.RoleSystem, "rules"),
		ai.TextMessage(ai.RoleUser, "question"),
		ai.TextMessage(ai.RoleAssistant, "answer"),
		ai.TextMessage(ai.RoleUser, "follow-up"),
		ai.TextMessage(ai.RoleSystem, "It is 10:00."),
		{Role: ai.RoleSystem, Parts: []ai.ContentPart{{Kind: ai.ContentJSON, JSON: []byte(`{"tz":"UTC"}`)}}},
	}
	var got map[string]any
	m := modelFor(t, ClaudeOpus5_5, func(w http.ResponseWriter, r *http.Request) {
		got = decodeRequest(t, r)
		emptyMessage(w)
	})
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: conversation}); err != nil {
		t.Fatal(err)
	}
	system := array(t, got["system"])
	messages := array(t, got["messages"])
	if len(system) != 1 || object(t, system[0])["text"] != "rules" || len(messages) != 4 {
		t.Fatalf("request = %#v", got)
	}
	last := object(t, messages[3])
	content := array(t, last["content"])
	if last["role"] != "system" || len(content) != 2 || object(t, content[0])["text"] != "It is 10:00." || object(t, content[1])["text"] != `{"tz":"UTC"}` {
		t.Fatalf("system message = %#v", last)
	}

	// A system message between tool results and the assistant reply is valid.
	toolTurn := []ai.Message{
		ai.TextMessage(ai.RoleUser, "question"),
		{Role: ai.RoleAssistant, Parts: []ai.ContentPart{{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{ID: "toolu_1", Type: "function", Name: "search", Args: []byte(`{}`)}}}},
		{Role: ai.RoleTool, Parts: []ai.ContentPart{{Kind: ai.ContentToolResult, ToolResult: &ai.ToolResult{ToolCallID: "toolu_1", Name: "search", Parts: ai.TextParts("found")}}}},
		ai.TextMessage(ai.RoleSystem, "Be brief."),
		ai.TextMessage(ai.RoleAssistant, "done"),
	}
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: toolTurn}); err != nil {
		t.Fatal(err)
	}
	if roles := messageRoles(t, got); !reflect.DeepEqual(roles, []string{"user", "assistant", "user", "system", "assistant"}) {
		t.Fatalf("roles = %v", roles)
	}

	invalid := map[string][]ai.Message{
		"after assistant":  {ai.TextMessage(ai.RoleUser, "q"), ai.TextMessage(ai.RoleAssistant, "a"), ai.TextMessage(ai.RoleSystem, "late")},
		"before user turn": {ai.TextMessage(ai.RoleUser, "q"), ai.TextMessage(ai.RoleSystem, "late"), ai.TextMessage(ai.RoleUser, "again")},
	}
	strict := modelFor(t, ClaudeOpus5_5, unexpectedRequest(t))
	for name, messages := range invalid {
		if _, err := strict.Generate(t.Context(), ai.AIRequest{Messages: messages}); !errors.Is(err, ai.ErrUnsupportedCapability) {
			t.Errorf("%s error = %v", name, err)
		}
	}
	for _, model := range []string{ClaudeSonnet5, ClaudeOpus4_7, ClaudeSonnet4_6, "claude-unknown"} {
		unsupported := modelFor(t, model, unexpectedRequest(t))
		if _, err := unsupported.Generate(t.Context(), ai.AIRequest{Messages: conversation}); !errors.Is(err, ai.ErrUnsupportedCapability) || !strings.Contains(err.Error(), "leading system") {
			t.Errorf("%s error = %v", model, err)
		}
	}
}

func messageRoles(t *testing.T, body map[string]any) []string {
	t.Helper()
	var roles []string
	for _, message := range array(t, body["messages"]) {
		roles = append(roles, object(t, message)["role"].(string))
	}
	return roles
}

func TestPromptCachingOption(t *testing.T) {
	var got map[string]any
	m := modelFor(t, ClaudeOpus5_5, func(w http.ResponseWriter, r *http.Request) {
		got = decodeRequest(t, r)
		emptyMessage(w)
	}, WithPromptCaching(PromptCacheTTL1h))
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi")}); err != nil {
		t.Fatal(err)
	}
	if control := object(t, got["cache_control"]); control["type"] != "ephemeral" || control["ttl"] != "1h" {
		t.Fatalf("cache_control = %#v", control)
	}
	if err := New("key", nil, WithPromptCaching("2h")).Validate(); err == nil || !strings.Contains(err.Error(), "2h") {
		t.Fatalf("invalid TTL error = %v", err)
	}
	// A later valid option replaces an invalid one.
	if err := New("key", nil, WithPromptCaching("2h"), WithPromptCaching(PromptCacheTTL1h)).Validate(); err != nil {
		t.Fatalf("overridden TTL error = %v", err)
	}
	m = modelFor(t, ClaudeOpus5_5, func(w http.ResponseWriter, r *http.Request) {
		got = decodeRequest(t, r)
		emptyMessage(w)
	})
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi")}); err != nil {
		t.Fatal(err)
	}
	if _, ok := got["cache_control"]; ok {
		t.Fatalf("cache_control sent without the option: %#v", got)
	}
}

const refusalMessage = `{"id":"msg_1","model":"claude-opus-5-5","content":[],"stop_reason":"refusal","stop_details":{"type":"refusal","category":"cyber","explanation":"declined by policy","recommended_model":"claude-opus-4-8"},"usage":{"input_tokens":5,"output_tokens":0}}`

func assertRefusal(t *testing.T, err error) {
	t.Helper()
	var refusal *RefusalError
	var terminal *ai.TerminalError
	if !errors.As(err, &refusal) || refusal.Category != "cyber" || refusal.Explanation != "declined by policy" || refusal.RecommendedModel != "claude-opus-4-8" {
		t.Fatalf("error = %#v, want refusal details", err)
	}
	if !errors.Is(err, ai.ErrUnsuccessfulGeneration) || !errors.As(err, &terminal) || terminal.Reason != "refusal" {
		t.Fatalf("refusal does not unwrap to a terminal error: %v", err)
	}
	if !strings.Contains(err.Error(), "cyber") || !strings.Contains(err.Error(), "declined by policy") {
		t.Fatalf("error text = %q", err.Error())
	}
}

func TestRefusalReportsStopDetails(t *testing.T) {
	m := modelFor(t, ClaudeOpus5_5, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(refusalMessage))
	})
	_, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi")})
	assertRefusal(t, err)

	stream := modelFor(t, ClaudeOpus5_5, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-5-5\",\"content\":[],\"usage\":{\"input_tokens\":5}}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\",\"stop_details\":{\"type\":\"refusal\",\"category\":\"cyber\",\"explanation\":\"declined by policy\",\"recommended_model\":\"claude-opus-4-8\"}},\"usage\":{\"output_tokens\":0}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
	})
	var streamErr error
	for token := range stream.GenerateStream(t.Context(), ai.AIRequest{Messages: userText("hi")}) {
		if token.Err != nil {
			streamErr = token.Err
		}
	}
	assertRefusal(t, streamErr)

	// A refusal without stop_details is still a RefusalError.
	bare := (&RefusalError{}).Error()
	if bare != "anthropic generation ended unsuccessfully: refusal" {
		t.Fatalf("bare refusal = %q", bare)
	}
}

func TestRefusalFallbacksRequestShape(t *testing.T) {
	type captured struct {
		beta string
		body map[string]any
	}
	var got captured
	handler := func(w http.ResponseWriter, r *http.Request) {
		got = captured{beta: r.Header.Get("anthropic-beta"), body: decodeRequest(t, r)}
		emptyMessage(w)
	}
	m := modelFor(t, ClaudeOpus5_5, handler, WithRefusalFallbacks())
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi")}); err != nil {
		t.Fatal(err)
	}
	if got.beta != fallbackBetaHeader || got.body["fallbacks"] != "default" {
		t.Fatalf("default fallbacks request = %+v", got)
	}

	WithModelRefusalFallbacks(ClaudeOpus4_8)(m)
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi")}); err != nil {
		t.Fatal(err)
	}
	list := array(t, got.body["fallbacks"])
	if len(list) != 1 || object(t, list[0])["model"] != ClaudeOpus4_8 {
		t.Fatalf("model fallbacks = %#v", got.body["fallbacks"])
	}

	// A hook that replaces extra fields does not drop fallbacks; one that sets
	// fallbacks itself wins.
	m.messageHooks = append(m.messageHooks, func(params *antropic.MessageNewParams) error {
		params.SetExtraFields(map[string]any{"fallbacks": []map[string]string{{"model": "claude-opus-5"}}})
		return nil
	})
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi")}); err != nil {
		t.Fatal(err)
	}
	if list := array(t, got.body["fallbacks"]); len(list) != 1 || object(t, list[0])["model"] != "claude-opus-5" {
		t.Fatalf("hook fallbacks = %#v", got.body["fallbacks"])
	}

	// Without the option no beta header or field is sent.
	plain := modelFor(t, ClaudeOpus5_5, handler)
	if _, err := plain.Generate(t.Context(), ai.AIRequest{Messages: userText("hi")}); err != nil {
		t.Fatal(err)
	}
	if _, ok := got.body["fallbacks"]; ok || got.beta != "" {
		t.Fatalf("plain request = %+v", got)
	}
}

const fallbackBlock = `{"type":"fallback","from":{"model":"claude-opus-5-5"},"to":{"model":"claude-opus-4-8"},"trigger":{"type":"refusal","category":"cyber"}}`

func TestFallbackBlocksRoundTrip(t *testing.T) {
	var got map[string]any
	m := modelFor(t, ClaudeOpus5_5, func(w http.ResponseWriter, r *http.Request) {
		got = decodeRequest(t, r)
		w.Header().Set("Content-Type", "application/json")
		// The declined attempt thinks, calls a tool and emits a malformed
		// tool call; only the fallback model's answer counts.
		_, _ = w.Write([]byte(`{"id":"msg_1","model":"claude-opus-4-8","content":[{"type":"thinking","thinking":"plan","signature":"sig"},{"type":"tool_use","id":"toolu_old","name":"search","input":{}},{"type":"tool_use","id":"toolu_bad","name":"","input":{}},` + fallbackBlock + `,{"type":"text","text":"answer"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3}}`))
	}, WithRefusalFallbacks())
	response, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi"), Tools: searchTool})
	if err != nil {
		t.Fatal(err)
	}
	// Thinking stays in the response, as in a stream; tool calls do not.
	parts := response.Message.Parts
	if len(parts) != 3 || parts[0].Kind != ai.ContentReasoning || parts[1].Kind != ai.ContentExtension || parts[2].Text != "answer" || len(response.Message.ToolCalls()) != 0 {
		t.Fatalf("parts = %#v", parts)
	}

	// Replay keeps the fallback block in place and drops the declined
	// attempt's thinking before it.
	history := []ai.Message{
		ai.TextMessage(ai.RoleUser, "hi"),
		{Role: ai.RoleAssistant, Parts: append([]ai.ContentPart{{Kind: ai.ContentReasoning, Text: "", Extensions: anthropicExtensions("thinking_signature", "sig")}, {Kind: ai.ContentText, Text: "partial "}}, parts...)},
		ai.TextMessage(ai.RoleUser, "next"),
	}
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: history}); err != nil {
		t.Fatal(err)
	}
	content := array(t, object(t, array(t, got["messages"])[1])["content"])
	var types []string
	for _, block := range content {
		types = append(types, object(t, block)["type"].(string))
	}
	if !reflect.DeepEqual(types, []string{"text", "fallback", "text"}) {
		t.Fatalf("replayed blocks = %v", types)
	}
	if replayed := object(t, content[1]); object(t, replayed["to"])["model"] != "claude-opus-4-8" {
		t.Fatalf("replayed fallback = %#v", replayed)
	}
}

func TestGenerateRejectsMalformedToolCallWithoutFallback(t *testing.T) {
	m := modelFor(t, ClaudeOpus5_5, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","model":"claude-opus-5-5","content":[{"type":"tool_use","id":"toolu_bad","name":"","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":5,"output_tokens":3}}`))
	}, WithRefusalFallbacks())
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi"), Tools: searchTool}); err == nil || !strings.Contains(err.Error(), "tool use name empty") {
		t.Fatalf("error = %v", err)
	}
}

// TestStreamReleasesHeldPartsWithoutMessageDelta checks that a stream ending
// cleanly before message_delta still delivers the parts held for a fallback.
func TestStreamReleasesHeldPartsWithoutMessageDelta(t *testing.T) {
	m := modelFor(t, ClaudeOpus5_5, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		events := []string{
			`{"type":"message_start","message":{"id":"msg_1","model":"claude-opus-5-5","content":[],"usage":{"input_tokens":5}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"search","input":{}}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"after"}}`,
			`{"type":"content_block_stop","index":1}`,
		}
		var b strings.Builder
		for _, event := range events {
			b.WriteString("event: x\ndata: " + event + "\n\n")
		}
		_, _ = w.Write([]byte(b.String()))
	}, WithRefusalFallbacks())
	var kinds []ai.ContentKind
	for token := range m.GenerateStream(t.Context(), ai.AIRequest{Messages: userText("hi"), Tools: searchTool}) {
		if token.Err != nil {
			t.Fatal(token.Err)
		}
		if token.Part != nil {
			kinds = append(kinds, token.Part.Kind)
		}
	}
	if !reflect.DeepEqual(kinds, []ai.ContentKind{ai.ContentToolCall, ai.ContentText}) {
		t.Fatalf("parts = %v", kinds)
	}
}

func TestStreamDropsToolCallsBeforeFallback(t *testing.T) {
	for _, hookOnly := range []bool{false, true} {
		t.Run(map[bool]string{false: "option", true: "hook"}[hookOnly], func(t *testing.T) { streamDropsToolCallsBeforeFallback(t, hookOnly) })
	}
}

func streamDropsToolCallsBeforeFallback(t *testing.T, hookOnly bool) {
	var requests atomic.Int32
	m := modelFor(t, ClaudeOpus5_5, func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		if r.Header.Get("anthropic-beta") != fallbackBetaHeader {
			t.Errorf("anthropic-beta = %q", r.Header.Get("anthropic-beta"))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		events := []string{
			`{"type":"message_start","message":{"id":"msg_1","model":"claude-opus-5-5","content":[],"usage":{"input_tokens":5}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial "}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_old","name":"search","input":{}}}`,
			`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
			`{"type":"content_block_stop","index":1}`,
			`{"type":"content_block_start","index":2,"content_block":` + fallbackBlock + `}`,
			`{"type":"content_block_stop","index":2}`,
			`{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"toolu_new","name":"search","input":{}}}`,
			`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"x\"}"}}`,
			`{"type":"content_block_stop","index":3}`,
			`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}`,
			`{"type":"message_stop"}`,
		}
		var b strings.Builder
		for _, event := range events {
			b.WriteString("event: x\ndata: " + event + "\n\n")
		}
		_, _ = w.Write([]byte(b.String()))
	})
	if hookOnly {
		m.messageHooks = append(m.messageHooks, func(params *antropic.MessageNewParams) error {
			params.SetExtraFields(map[string]any{"fallbacks": "default"})
			return nil
		})
	} else {
		WithRefusalFallbacks()(m.client)
	}
	var completion *ai.Completion
	var kinds []ai.ContentKind
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Messages: userText("hi"), Tools: searchTool}) {
		if token.Err != nil {
			t.Fatal(token.Err)
		}
		if token.Part != nil {
			kinds = append(kinds, token.Part.Kind)
		}
		if token.Completion != nil {
			completion = token.Completion
			if len(kinds) != 3 {
				t.Fatalf("completion arrived before held parts: %v", kinds)
			}
		}
	}
	if !reflect.DeepEqual(kinds, []ai.ContentKind{ai.ContentText, ai.ContentExtension, ai.ContentToolCall}) {
		t.Fatalf("parts = %v", kinds)
	}
	if completion == nil || completion.Model != ClaudeOpus4_8 || requests.Load() != 1 {
		t.Fatalf("completion = %#v", completion)
	}
}

func TestFallbacksScopeAndBetaHeader(t *testing.T) {
	type captured struct {
		beta string
		body map[string]any
	}
	var got captured
	handler := func(w http.ResponseWriter, r *http.Request) {
		got = captured{beta: r.Header.Get("anthropic-beta"), body: decodeRequest(t, r)}
		emptyMessage(w)
	}
	// The provider-wide option skips models without server-side fallbacks.
	for _, model := range []string{ClaudeHaiku5_5, ClaudeSonnet4_6, "claude-unknown"} {
		m := modelFor(t, model, handler, WithRefusalFallbacks())
		if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi")}); err != nil {
			t.Fatal(err)
		}
		if _, ok := got.body["fallbacks"]; ok || got.beta != "" {
			t.Fatalf("%s request = %+v", model, got)
		}
		// A model option still applies.
		WithModelRefusalFallbacks()(m)
		if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi")}); err != nil {
			t.Fatal(err)
		}
		if got.body["fallbacks"] != "default" || got.beta != fallbackBetaHeader {
			t.Fatalf("%s with model option = %+v", model, got)
		}
	}

	// A hook that sets fallbacks without any option still gets the header.
	m := modelFor(t, ClaudeOpus5_5, handler)
	m.messageHooks = append(m.messageHooks, func(params *antropic.MessageNewParams) error {
		params.SetExtraFields(map[string]any{"fallbacks": "default"})
		return nil
	})
	if _, err := m.Generate(t.Context(), ai.AIRequest{Messages: userText("hi")}); err != nil {
		t.Fatal(err)
	}
	if got.body["fallbacks"] != "default" || got.beta != fallbackBetaHeader {
		t.Fatalf("hook-only fallbacks = %+v", got)
	}

	// A hook that sets fallbacks to nil turns them off: no field, no header.
	off := modelFor(t, ClaudeOpus5_5, handler, WithRefusalFallbacks())
	off.messageHooks = append(off.messageHooks, func(params *antropic.MessageNewParams) error {
		params.SetExtraFields(map[string]any{"fallbacks": nil})
		return nil
	})
	if _, err := off.Generate(t.Context(), ai.AIRequest{Messages: userText("hi")}); err != nil {
		t.Fatal(err)
	}
	if _, ok := got.body["fallbacks"]; ok || got.beta != "" {
		t.Fatalf("hook opt-out request = %+v", got)
	}

	// WithoutRefusalFallbacks turns the provider-wide option off for one model.
	provider := New("key", nil, WithRefusalFallbacks())
	if without, err := provider.TypedModel(ClaudeOpus5_5, WithoutRefusalFallbacks()); err != nil || without.fallbacksValue() != nil {
		t.Fatalf("WithoutRefusalFallbacks = %v, %v", without, err)
	}
	if with, err := provider.TypedModel(ClaudeOpus5_5); err != nil || with.fallbacksValue() != defaultFallbacks {
		t.Fatalf("provider fallbacks = %v, %v", with, err)
	}
	// An empty fallback model name fails at construction.
	if _, err := provider.TypedModel(ClaudeOpus5_5, WithModelRefusalFallbacks(ClaudeOpus4_8, " ")); err == nil {
		t.Fatal("empty fallback model name accepted")
	}

	// Replaying a fallback block needs the header even with fallbacks off,
	// for generation and token counting.
	history := []ai.Message{
		ai.TextMessage(ai.RoleUser, "hi"),
		{Role: ai.RoleAssistant, Parts: []ai.ContentPart{fallbackPart(fallbackBlock), {Kind: ai.ContentText, Text: "answer"}}},
		ai.TextMessage(ai.RoleUser, "next"),
	}
	plain := modelFor(t, ClaudeOpus5_5, handler)
	if _, err := plain.Generate(t.Context(), ai.AIRequest{Messages: history}); err != nil {
		t.Fatal(err)
	}
	if _, ok := got.body["fallbacks"]; ok || got.beta != fallbackBetaHeader {
		t.Fatalf("fallback history request = %+v", got)
	}
	counter := modelFor(t, ClaudeOpus5_5, func(w http.ResponseWriter, r *http.Request) {
		got = captured{beta: r.Header.Get("anthropic-beta"), body: decodeRequest(t, r)}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"input_tokens":12}`))
	})
	if _, err := counter.CountInputTokens(t.Context(), ai.AIRequest{Messages: history}); err != nil {
		t.Fatal(err)
	}
	if _, ok := got.body["fallbacks"]; ok || got.beta != fallbackBetaHeader {
		t.Fatalf("count_tokens with fallback history = %+v", got)
	}
}
