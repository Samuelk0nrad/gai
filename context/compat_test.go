package context_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/lace-ai/gai/ai"
	legacy "github.com/lace-ai/gai/context"
	legacyhistory "github.com/lace-ai/gai/context/history"
	legacytools "github.com/lace-ai/gai/context/tooldefinitions"
	"github.com/lace-ai/gai/modelinput"
	"github.com/lace-ai/gai/modelinput/history"
	"github.com/lace-ai/gai/modelinput/tooldefinitions"
)

// The return types intentionally use the old paths, as existing application
// implementations do. They must satisfy the canonical interfaces unchanged.
type legacySource struct{}

func (legacySource) Name() string { return "legacy" }

func (legacySource) Function(context.Context, int) (legacy.Part, error) {
	return legacy.NewMessagePart(ai.TextMessage(ai.RoleAssistant, "previous answer")), nil
}

type legacyReader struct{ state *legacyhistory.HistoryState }

func (r legacyReader) LoadHistory(context.Context, string) (legacyhistory.HistorySnapshot, error) {
	return legacyhistory.HistorySnapshot{Revision: "legacy-revision", State: r.state.Clone()}, nil
}

type canonicalReader struct{ state *history.HistoryState }

func (r canonicalReader) LoadHistory(context.Context, string) (history.HistorySnapshot, error) {
	return history.HistorySnapshot{Revision: "canonical-revision", State: r.state.Clone()}, nil
}

var (
	_ modelinput.ContextSource    = legacySource{}
	_ history.HistoryReader       = legacyReader{}
	_ legacyhistory.HistoryReader = canonicalReader{}
)

func TestCompatibilityMixedBuilderConfiguration(t *testing.T) {
	t.Parallel()

	oldDefinition := legacy.Definition{
		Renderer:           &modelinput.SimpleRenderer{},
		SystemInstructions: []legacy.Part{modelinput.NewTextPart("system instructions")},
		ContextSources:     []legacy.ContextSource{legacySource{}},
		PromptInput: modelinput.PromptInput{
			User:    ai.TextParts("current request"),
			Context: []legacy.Part{modelinput.NewTextPart("current context")},
		},
		TokenBudget: -1,
	}
	newDefinition := modelinput.Definition{
		Renderer:           &legacy.SimpleRenderer{},
		SystemInstructions: []modelinput.Part{legacy.NewTextPart("system instructions")},
		ContextSources:     []modelinput.ContextSource{legacySource{}},
		PromptInput: legacy.PromptInput{
			User:    ai.TextParts("current request"),
			Context: []modelinput.Part{legacy.NewTextPart("current context")},
		},
		TokenBudget: -1,
	}
	for name, builder := range map[string]*modelinput.Builder{
		"old constructor with new configuration": legacy.New(newDefinition),
		"new constructor with old configuration": modelinput.New(oldDefinition),
	} {
		t.Run(name, func(t *testing.T) {
			var oldPointer *legacy.Builder = builder
			var newPointer *modelinput.Builder = oldPointer
			if oldPointer != builder || newPointer != builder {
				t.Fatal("builder aliases changed pointer identity")
			}
			var oldContract legacy.PromptBuilder = newPointer
			var newContract modelinput.PromptBuilder = oldContract
			if _, err := newContract.BuildContext(t.Context()); err != nil {
				t.Fatal(err)
			}
			request, err := oldContract.BuildRequest(t.Context(), nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(request.Messages) != 4 {
				t.Fatalf("messages = %+v, want system, prior answer, context, user", request.Messages)
			}
			if request.Messages[0].Role != ai.RoleSystem || !strings.Contains(request.Messages[0].Parts[0].Text, "system instructions") {
				t.Fatalf("system instructions lost: %+v", request.Messages[0])
			}
			if !reflect.DeepEqual(request.Messages[1], ai.TextMessage(ai.RoleAssistant, "previous answer")) {
				t.Fatalf("old ConversationPart was rendered instead of retaining its role: %+v", request.Messages[1])
			}
			if !strings.Contains(request.Messages[2].Parts[0].Text, "current context") || !reflect.DeepEqual(request.Messages[3], ai.TextMessage(ai.RoleUser, "current request")) {
				t.Fatalf("mixed input lost content: %+v", request.Messages[2:])
			}
			oldPointer.SetInput(modelinput.PromptInput{User: ai.TextParts("replacement")})
			if got := newPointer.Input().User[0].Text; got != "replacement" {
				t.Fatalf("changes through old pointer did not reach new pointer: %q", got)
			}
		})
	}
}

func TestCompatibilityHistoryReadersAndStates(t *testing.T) {
	t.Parallel()

	oldState := &legacyhistory.HistoryState{
		Turns: []modelinput.Turn{{
			ID: "turn", Count: 1,
			UserMessage: &legacy.StoredMessage{Message: ai.TextMessage(ai.RoleUser, "question")},
			Messages:    []modelinput.StoredMessage{{Message: ai.TextMessage(ai.RoleAssistant, "answer")}},
		}},
	}
	var newState *history.HistoryState = oldState
	if newState != oldState {
		t.Fatal("history state aliases changed pointer identity")
	}
	for name, source := range map[string]*history.HistorySource{
		"new source with old reader": history.NewHistory("session", legacyReader{oldState}),
		"old source with new reader": legacyhistory.NewHistory("session", canonicalReader{newState}),
	} {
		t.Run(name, func(t *testing.T) {
			var oldSource legacy.ContextSourceWithTokenCount = source
			var newSource modelinput.ContextSourceWithTokenCount = oldSource
			newSource.SetTokenCounter(ai.TextTokenEstimator{})
			part, count, err := newSource.FunctionWithTokens(t.Context(), 1000)
			if err != nil || count <= 0 {
				t.Fatalf("history selection: count=%d, error=%v", count, err)
			}
			newPart, ok := part.(*history.Part)
			if !ok {
				t.Fatalf("history part has type %T", part)
			}
			oldPart, ok := part.(*legacyhistory.Part)
			if !ok || oldPart != newPart {
				t.Fatal("history part does not have the same old and new identity")
			}
			want := []ai.Message{ai.TextMessage(ai.RoleUser, "question"), ai.TextMessage(ai.RoleAssistant, "answer")}
			if !reflect.DeepEqual(oldPart.ConversationMessages(), want) {
				t.Fatalf("mixed history content = %+v, want %+v", oldPart.ConversationMessages(), want)
			}
		})
	}
}

func TestCompatibilityErrorIdentity(t *testing.T) {
	t.Parallel()

	_, oldPartError := legacy.NewNamedPart("", "text")
	_, newPartError := modelinput.NewNamedPart("", "text")
	var oldCompactor *legacyhistory.Compactor
	_, oldCompactorError := oldCompactor.Compact(t.Context(), 1)
	var newCompactor *history.Compactor
	_, newCompactorError := newCompactor.Compact(t.Context(), 1)
	_, oldToolsError := legacytools.New(nil, nil, nil)
	_, newToolsError := tooldefinitions.New(nil, nil, nil)
	for _, test := range []struct {
		name                          string
		err, oldSentinel, newSentinel error
	}{
		{"old named part", oldPartError, legacy.ErrPromptPartName, modelinput.ErrPromptPartName},
		{"new named part", newPartError, legacy.ErrPromptPartName, modelinput.ErrPromptPartName},
		{"old compactor", oldCompactorError, legacyhistory.ErrCompactorNil, history.ErrCompactorNil},
		{"new compactor", newCompactorError, legacyhistory.ErrCompactorNil, history.ErrCompactorNil},
		{"old tool definitions", oldToolsError, legacytools.ErrToolsEmpty, tooldefinitions.ErrToolsEmpty},
		{"new tool definitions", newToolsError, legacytools.ErrToolsEmpty, tooldefinitions.ErrToolsEmpty},
	} {
		t.Run(test.name, func(t *testing.T) {
			if !errors.Is(test.err, test.oldSentinel) || !errors.Is(test.err, test.newSentinel) {
				t.Fatalf("error %v does not match both import paths", test.err)
			}
		})
	}

	newConflict := &history.RevisionConflictError{SessionID: "session", Expected: legacyhistory.Revision("r1"), Actual: "r2"}
	var oldTarget *legacyhistory.RevisionConflictError
	if !errors.As(fmt.Errorf("write rejected: %w", newConflict), &oldTarget) || oldTarget != newConflict {
		t.Fatal("new revision conflict does not match the old error type")
	}
	oldConflict := &legacyhistory.RevisionConflictError{SessionID: "session", Expected: history.Revision("r2"), Actual: "r3"}
	var newTarget *history.RevisionConflictError
	if !errors.As(fmt.Errorf("write rejected: %w", oldConflict), &newTarget) || newTarget != oldConflict {
		t.Fatal("old revision conflict does not match the new error type")
	}
}

func TestCompatibilityPersistenceRoundTrip(t *testing.T) {
	t.Parallel()

	oldState := legacyhistory.HistoryState{
		SchemaVersion: history.HistorySchemaVersion,
		Summary:       history.NewSummary("summary", "t0", "t0", 0, 0, ai.ContentPart{Kind: ai.ContentText, Text: "earlier conversation"}),
		Turns: []legacy.Turn{{
			ID: "t1", Count: 1,
			UserMessage: &legacy.StoredMessage{
				SchemaVersion: modelinput.MessageSchemaVersion, ID: "user", SessionID: "session", TurnID: "t1",
				Message: ai.TextMessage(ai.RoleUser, "question"),
			},
			Messages: []modelinput.StoredMessage{{
				SchemaVersion: legacy.MessageSchemaVersion, ID: "answer", SessionID: "session", TurnID: "t1",
				Message: ai.TextMessage(ai.RoleAssistant, "answer"),
			}},
		}},
	}
	oldJSON, err := json.Marshal(oldState)
	if err != nil {
		t.Fatal(err)
	}
	var newState history.HistoryState
	if err := json.Unmarshal(oldJSON, &newState); err != nil {
		t.Fatal(err)
	}
	newJSON, err := json.Marshal(newState)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip legacyhistory.HistoryState
	if err := json.Unmarshal(newJSON, &roundTrip); err != nil {
		t.Fatal(err)
	}
	if string(oldJSON) != string(newJSON) || !reflect.DeepEqual(oldState, roundTrip) {
		t.Fatalf("history changed across import paths:\nold %s\nnew %s", oldJSON, newJSON)
	}

	messageJSON, err := json.Marshal(*oldState.Turns[0].UserMessage)
	if err != nil {
		t.Fatal(err)
	}
	var newMessage modelinput.StoredMessage
	if err := json.Unmarshal(messageJSON, &newMessage); err != nil {
		t.Fatal(err)
	}
	newMessageJSON, err := json.Marshal(newMessage)
	if err != nil {
		t.Fatal(err)
	}
	var oldMessage legacy.StoredMessage
	if err := json.Unmarshal(newMessageJSON, &oldMessage); err != nil {
		t.Fatal(err)
	}
	if string(messageJSON) != string(newMessageJSON) || !reflect.DeepEqual(*oldState.Turns[0].UserMessage, oldMessage) {
		t.Fatalf("stored message changed across import paths: %+v", oldMessage)
	}
}

type compatibilityTool struct{}

func (compatibilityTool) Name() string        { return "search" }
func (compatibilityTool) Description() string { return "Searches documentation." }
func (compatibilityTool) Params() ai.ToolParameters {
	return ai.ToolParameters{Properties: []ai.ToolParameter{{Name: "query", Type: ai.ToolParameterString}}}
}

func TestCompatibilityMixedToolDefinitions(t *testing.T) {
	t.Parallel()

	choice := ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"search"}}
	oldOptions := []legacytools.Option{tooldefinitions.WithUsageProtocol("Call search for documentation."), legacytools.WithToolChoice(choice)}
	newOptions := []tooldefinitions.Option{legacytools.WithUsageProtocol("Call search for documentation."), tooldefinitions.WithToolChoice(choice)}
	newSource, err := tooldefinitions.New(&legacy.SimpleRenderer{}, []legacy.ToolSignature{compatibilityTool{}}, nil, oldOptions...)
	if err != nil {
		t.Fatal(err)
	}
	oldSource, err := legacytools.New(&modelinput.SimpleRenderer{}, []modelinput.ToolSignature{compatibilityTool{}}, nil, newOptions...)
	if err != nil {
		t.Fatal(err)
	}
	for name, source := range map[string]*tooldefinitions.Source{"new source with old types": newSource, "old source with new types": oldSource} {
		t.Run(name, func(t *testing.T) {
			var oldPointer *legacytools.Source = source
			if oldPointer != source {
				t.Fatal("tool source aliases changed pointer identity")
			}
			var oldContract legacy.ContextSource = source
			var newContract modelinput.ContextSource = oldContract
			part, err := newContract.Function(t.Context(), 1000)
			if err != nil {
				t.Fatal(err)
			}
			oldRendered, err := (legacy.SimpleRenderer{}).Render(t.Context(), []modelinput.Part{part})
			if err != nil {
				t.Fatal(err)
			}
			newRendered, err := (modelinput.SimpleRenderer{}).Render(t.Context(), []legacy.Part{part})
			if err != nil {
				t.Fatal(err)
			}
			if oldRendered != newRendered {
				t.Fatalf("renderers differ:\nold %s\nnew %s", oldRendered, newRendered)
			}
			for _, fragment := range []string{"Call search for documentation.", "selected tools: search", "tool: search", "query"} {
				if !strings.Contains(newRendered, fragment) {
					t.Fatalf("mixed options or signatures lost %q:\n%s", fragment, newRendered)
				}
			}
		})
	}
}
