package ai

import "context"

// Model is the minimal streaming capability used by the agent and loop.
// Diagnostic names, synchronous generation, counting, and descriptors are
// optional capabilities. Resource ownership stays with the caller; callers
// may close models implementing io.Closer after all runs have finished.
//
// Implementations must close the channel returned by GenerateStream when the
// request finishes or its context is canceled. Cancellation may be represented
// by closing the channel without emitting an error token.
type Model interface {
	// GenerateStream executes a request and emits response tokens incrementally.
	// Implementations should use SendToken so cancellation cannot block a sender.
	GenerateStream(ctx context.Context, req AIRequest) <-chan Token
}

// ModelGenerator is the optional synchronous generation capability. Concrete
// models may expose it directly; the loop only requires GenerateStream.
type ModelGenerator interface {
	Generate(context.Context, AIRequest) (*AIResponse, error)
}

// ModelNamer optionally supplies the provider-specific name used in diagnostics.
type ModelNamer interface {
	Name() string
}

// ModelName returns an optional diagnostic name, or an empty string when the
// model does not expose one. Naming is never required to execute a model.
func ModelName(model Model) string {
	if namer, ok := model.(ModelNamer); ok {
		return namer.Name()
	}
	return ""
}

// NativeToolModel is optionally implemented by legacy models that send tool
// definitions through their provider's native tool-calling API. New models
// should implement ModelDescriber and report ModelDescriptor.NativeTools
// instead. Agent consults NativeToolModel only when ModelDescriber is absent.
//
// It is deliberately separate from Model so existing custom Model
// implementations retain the text-based compatibility protocol by default.
//
// Deprecated: implement ModelDescriber instead.
type NativeToolModel interface {
	NativeTools() bool
}
