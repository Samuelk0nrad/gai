// This credential-free example demonstrates deferred approval and restoration.
// Replace jsonStore with durable application storage before using crash recovery.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"sync"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/modelinput"
)

// jsonStore demonstrates atomic revision checks and a serialization boundary.
// This store is process-local; it intentionally makes no durability claim.
type jsonStore struct {
	mu       sync.Mutex
	revision uint64
	data     []byte
}

func (s *jsonStore) commit(_ context.Context, expected uint64, next agent.Checkpoint) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if expected != s.revision {
		return errors.New("checkpoint ownership lost")
	}
	data, err := json.Marshal(next)
	if err != nil {
		return err
	}
	s.data, s.revision = data, next.Revision
	return nil
}

func (s *jsonStore) load() (agent.Checkpoint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var checkpoint agent.Checkpoint
	err := json.Unmarshal(s.data, &checkpoint)
	return checkpoint, err
}

// scriptedModel uses the canonical transcript, so a fresh instance can continue.
type scriptedModel struct{}

func (scriptedModel) Descriptor() ai.ModelDescriptor {
	return ai.ModelDescriptor{NativeTools: ai.FeatureSupportSupported}
}

func (scriptedModel) GenerateStream(ctx context.Context, request ai.AIRequest) <-chan ai.Token {
	part := ai.ContentPart{Kind: ai.ContentToolCall, ToolCall: &ai.ToolCall{Type: "function", ID: "provider-call", Name: "save", Args: json.RawMessage(`{}`)}}
	for _, message := range request.Messages {
		if len(message.ToolResults()) > 0 {
			part = ai.ContentPart{Kind: ai.ContentText, Text: "Saved after approval."}
		}
	}
	out := make(chan ai.Token, 1)
	ai.SendToken(ctx, out, ai.Token{Part: &part})
	close(out)
	return out
}

func definition(effect func(string)) (agent.Definition, error) {
	tool, err := loop.NewTool("save", "Save the approved operation", ai.ToolParameters{}, func(ctx context.Context, _ ai.ToolCall) (string, error) {
		invocation, ok := loop.InvocationFromContext(ctx)
		if !ok {
			return "", errors.New("missing invocation identity")
		}
		// A real external service must atomically honor this idempotency key.
		effect(invocation.RunID + ":" + invocation.InvocationID)
		return "saved", nil
	})
	if err != nil {
		return agent.Definition{}, err
	}
	return agent.Definition{
		Name: "approval-example", Model: scriptedModel{}, Tools: []loop.Tool{tool},
		ToolPolicy: loop.ToolPolicyFunc(func(context.Context, loop.ToolPolicyInput) (loop.ToolDecision, error) {
			return loop.ToolDecision{Action: loop.ToolRequireApproval}, nil
		}),
		Prompt: func(context.Context, agent.RunInput) (modelinput.PromptBuilder, error) {
			return modelinput.New(modelinput.Definition{}), nil
		},
	}, nil
}

func run(ctx context.Context, out io.Writer) error {
	store := &jsonStore{}
	durability := agent.Durability{DefinitionRevision: "example-v1", DeferredApprovals: true, Commit: store.commit}
	effects := 0
	effect := func(string) { effects++ }
	def, err := definition(effect)
	if err != nil {
		return err
	}
	w, err := agent.New(def).NewRun(ctx, agent.RunInput{ID: "example-run", Prompt: modelinput.PromptInput{User: ai.TextParts("Save this")}, Durability: &durability})
	if err != nil {
		return err
	}
	paused, err := w.Run(ctx)
	if err != nil {
		return err
	}
	if paused.Status != agent.StatusPaused || effects != 0 {
		return errors.New("expected approval before any effect")
	}
	fmt.Fprintln(out, "Paused for approval; effects: 0")
	checkpoint, err := store.load()
	if err != nil {
		return err
	}
	request := checkpoint.Stages[0].Continuation.Pending.Invocations[0].Approval
	// Recreate all runtime dependencies, as a replacement worker would.
	def, err = definition(effect)
	if err != nil {
		return err
	}
	w, err = agent.New(def).Resume(ctx, checkpoint, agent.ResumeInput{
		CommandID: "approval-command-1",
		Approvals: []loop.ToolApprovalDecision{{RequestID: request.ID, Approved: true}},
	}, durability)
	if err != nil {
		return err
	}
	result, err := w.Run(ctx)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "%s Effects: %d\n", result.Text, effects)
	return nil
}

func main() {
	if err := run(context.Background(), os.Stdout); err != nil {
		log.Fatal(err)
	}
}
