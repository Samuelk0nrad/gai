// Package context preserves the former model-input import path.
//
// This is a deprecated compatibility package; use github.com/lace-ai/gai/modelinput.
// It remains available until its removal in a future breaking release.
package context

import (
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/modelinput"
)

type (
	Builder                           = modelinput.Builder
	ContextSource                     = modelinput.ContextSource
	ContextSourceWithTokenCount       = modelinput.ContextSourceWithTokenCount
	ContextSourceWithBudgetProjection = modelinput.ContextSourceWithBudgetProjection
	TokenCounterSetter                = modelinput.TokenCounterSetter
	PromptBuilder                     = modelinput.PromptBuilder
	TokenBudget                       = modelinput.TokenBudget
	Definition                        = modelinput.Definition
	Part                              = modelinput.Part
	TextPart                          = modelinput.TextPart
	MessagePart                       = modelinput.MessagePart
	SystemPart                        = modelinput.SystemPart
	PromptInput                       = modelinput.PromptInput
	NamedPart                         = modelinput.NamedPart
	Conversation                      = modelinput.Conversation
	ConversationPart                  = modelinput.ConversationPart
	Turn                              = modelinput.Turn
	StoredMessage                     = modelinput.StoredMessage
	Renderer                          = modelinput.Renderer
	PreviewRenderer                   = modelinput.PreviewRenderer
	ToolSignature                     = modelinput.ToolSignature
	RenderResultCallback              = modelinput.RenderResultCallback
	RenderField                       = modelinput.RenderField
	RenderNode                        = modelinput.RenderNode
	XMLRenderer                       = modelinput.XMLRenderer
	SimpleRenderer                    = modelinput.SimpleRenderer
)

const MessageSchemaVersion = modelinput.MessageSchemaVersion

// These variables retain the canonical sentinel error identities. Applications
// should compare or wrap them, rather than reassign them.
var (
	ErrSessionNotFound      = modelinput.ErrSessionNotFound
	ErrInvalidSessionID     = modelinput.ErrInvalidSessionID
	ErrSessionStoreNotFound = modelinput.ErrSessionStoreNotFound
	ErrRAGStoreNotFound     = modelinput.ErrRAGStoreNotFound
	ErrPromptPathEmpty      = modelinput.ErrPromptPathEmpty
	ErrPromptFileType       = modelinput.ErrPromptFileType
	ErrPromptMissing        = modelinput.ErrPromptMissing
	ErrPromptBuilderNil     = modelinput.ErrPromptBuilderNil
	ErrPromptEntryID        = modelinput.ErrPromptEntryID
	ErrPromptSource         = modelinput.ErrPromptSource
	ErrPromptBudget         = modelinput.ErrPromptBudget
	ErrPromptPartName       = modelinput.ErrPromptPartName
	ErrTokenCounterNotFound = modelinput.ErrTokenCounterNotFound
	ErrMessageNotFound      = modelinput.ErrMessageNotFound
	ErrInvalidTokenCount    = modelinput.ErrInvalidTokenCount
	ErrInvalideTokenLimit   = modelinput.ErrInvalideTokenLimit
	ErrInvalidOutputReserve = modelinput.ErrInvalidOutputReserve
)

func New(def Definition) *Builder { return modelinput.New(def) }

func NewBuilder(renderer Renderer, tokenBudget int) *Builder {
	return modelinput.NewBuilder(renderer, tokenBudget)
}

func NewTextPart(content string) TextPart { return modelinput.NewTextPart(content) }

func NewMessagePart(message ai.Message) MessagePart { return modelinput.NewMessagePart(message) }

func NewSystemPart(instructions []Part) SystemPart { return modelinput.NewSystemPart(instructions) }

func NewNamedPart(name, value string) (*NamedPart, error) {
	return modelinput.NewNamedPart(name, value)
}

func NewJSONPart(name string, value any) (*NamedPart, error) {
	return modelinput.NewJSONPart(name, value)
}

func IsValidRole(role ai.Role) bool { return modelinput.IsValidRole(role) }

func LoadPromptFromFile(path string) (Part, error) { return modelinput.LoadPromptFromFile(path) }
