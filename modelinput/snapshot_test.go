package modelinput_test

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput"
	"github.com/Samuelk0nrad/gai/testutil/mocks"
)

func snapshotForTest(t *testing.T) (modelinput.Snapshot, ai.AIRequest) {
	t.Helper()
	named, err := modelinput.NewNamedPart("tenant", "ACME")
	if err != nil {
		t.Fatal(err)
	}
	history := modelinput.NewMessagePart(ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("earlier turn")})
	b := modelinput.New(modelinput.Definition{
		SystemInstructions: []modelinput.Part{modelinput.NewTextPart("system")},
		PromptInput:        modelinput.PromptInput{User: ai.TextParts("question"), Context: []modelinput.Part{named, history}},
	})
	ctx := context.Background()
	if _, err := b.BuildContext(ctx); err != nil {
		t.Fatal(err)
	}
	request, err := b.BuildRequest(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := b.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	return snapshot, request
}

func TestSnapshotRestoreReproducesRequestWithoutSources(t *testing.T) {
	snapshot, request := snapshotForTest(t)
	restored, err := modelinput.Restore(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if parts, err := restored.BuildContext(ctx); err != nil || len(parts) != 0 {
		t.Fatal("frozen builder selected context again", parts, err)
	}
	conversation := []ai.Message{{Role: ai.RoleAssistant, Parts: ai.TextParts("answer")}}
	got, err := restored.BuildRequest(ctx, testConversation(conversation))
	if err != nil {
		t.Fatal(err)
	}
	want := append(ai.CloneMessages(request.Messages), conversation...)
	if !reflect.DeepEqual(got.Messages, want) {
		t.Fatalf("restored request\n got %#v\nwant %#v", got.Messages, want)
	}
}

func TestSnapshotInputRestoresFrozenContextParts(t *testing.T) {
	snapshot, _ := snapshotForTest(t)
	input := snapshot.Input()
	if len(input.Context) != 2 || input.Context[0].Name() != "tenant" || input.Context[1].Name() != "message" {
		t.Fatal(input.Context)
	}
	ctx := context.Background()
	node, err := input.Context[0].Render(ctx)
	if err != nil || !containsValue(node, "ACME") {
		t.Fatal(node, err)
	}
	if n, err := input.Context[0].Tokens(ctx, &mocks.MockTokenCounter{Count: 7}); err != nil || n != 7 {
		t.Fatal(n, err)
	}
	// Restored parts count the same text as the parts they were saved from.
	named, _ := modelinput.NewNamedPart("tenant", "ACME")
	history := modelinput.NewMessagePart(ai.Message{Role: ai.RoleUser, Parts: ai.TextParts("earlier turn")})
	for i, original := range []modelinput.Part{named, history} {
		want, err := original.Tokens(ctx, charCounter{})
		if err != nil {
			t.Fatal(err)
		}
		if got, err := input.Context[i].Tokens(ctx, charCounter{}); err != nil || got != want {
			t.Fatalf("%s tokens = %d, want %d (%v)", original.Name(), got, want, err)
		}
		if _, err := input.Context[i].Tokens(ctx, nil); !errors.Is(err, modelinput.ErrTokenCounterNotFound) {
			t.Fatal("missing counter accepted", err)
		}
	}
	conversation, ok := input.Context[1].(modelinput.ConversationPart)
	if !ok {
		t.Fatal("history part lost its conversation messages")
	}
	messages := conversation.ConversationMessages()
	if len(messages) != 1 || messages[0].Text() != "earlier turn" {
		t.Fatal(messages)
	}
	messages[0].Parts[0].Text = "changed"
	if conversation.ConversationMessages()[0].Text() != "earlier turn" {
		t.Fatal("restored conversation part is not detached")
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := input.Context[0].Render(canceled); err == nil {
		t.Fatal("render ignored cancellation")
	}
}

func TestSnapshotCloneAndValidate(t *testing.T) {
	snapshot, _ := snapshotForTest(t)
	clone := snapshot.Clone()
	clone.Messages[0].Parts[0].Text = "changed"
	clone.User[0].Text = "changed"
	clone.Context[0].Node.Fields = append(clone.Context[0].Node.Fields, modelinput.RenderField{Key: "k", Value: "v"})
	clone.Context[0].PartName = "changed"
	if reflect.DeepEqual(clone, snapshot) || snapshot.Context[0].PartName != "tenant" || snapshot.User[0].Text != "question" {
		t.Fatal("clone shares state with the snapshot")
	}
	if err := snapshot.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := (modelinput.Snapshot{}).Validate(); err == nil {
		t.Fatal("empty snapshot accepted")
	}
	if _, err := modelinput.Restore(modelinput.Snapshot{}); err == nil {
		t.Fatal("restore accepted an empty snapshot")
	}
	invalid := snapshot.Clone()
	invalid.Budget.Limit = -1
	if err := invalid.Validate(); err == nil {
		t.Fatal("invalid budget accepted")
	}
}

func containsValue(node modelinput.RenderNode, value string) bool {
	if node.Value == value {
		return true
	}
	for _, field := range node.Fields {
		if field.Value == value {
			return true
		}
	}
	for _, child := range node.Children {
		if containsValue(child, value) {
			return true
		}
	}
	return false
}

type testConversation []ai.Message

func (c testConversation) Messages() []ai.Message { return ai.CloneMessages(c) }

// charCounter counts characters, so token counts reflect exactly what text a part contributes.
type charCounter struct{}

func (charCounter) CountTokens(_ context.Context, text string) (int, error) { return len(text), nil }
func (charCounter) ID() string                                              { return "test.chars" }
func (charCounter) Fidelity() ai.TokenCountFidelity                         { return ai.TokenCountEstimated }
