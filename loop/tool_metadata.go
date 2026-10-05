package loop

import "github.com/lace-ai/gai/ai"

// ToolEffect describes declared side effects, not an authorization guarantee.
type ToolEffect string

const (
	ToolEffectUnknown     ToolEffect = ""
	ToolEffectReadOnly    ToolEffect = "read_only"
	ToolEffectMutating    ToolEffect = "mutating"
	ToolEffectDestructive ToolEffect = "destructive"
)

// ToolTraits are execution-only declarations. Idempotent never enables retries.
// The zero value makes no safety or idempotency claim.
type ToolTraits struct {
	Effect     ToolEffect
	Idempotent bool
}

// ToolPolicyInput is a portable snapshot for authorization and result processing.
// Call is independently cloned for each callback. Application identity belongs
// in trusted context, never in model-supplied arguments.
type ToolPolicyInput struct {
	Call   ai.ToolCall
	Traits ToolTraits
}
