// Package tooldefinitions preserves the former tool-definition import path.
//
// Deprecated: Use github.com/Samuelk0nrad/gai/modelinput/tooldefinitions.
// It remains available until its removal in a future breaking release.
package tooldefinitions

import (
	"github.com/Samuelk0nrad/gai"
	"github.com/Samuelk0nrad/gai/ai"
	"github.com/Samuelk0nrad/gai/modelinput"
	"github.com/Samuelk0nrad/gai/modelinput/tooldefinitions"
)

type (
	ToolSignatureRenderer = tooldefinitions.ToolSignatureRenderer
	Source                = tooldefinitions.Source
	Option                = tooldefinitions.Option
)

var (
	ErrToolsEmpty  = tooldefinitions.ErrToolsEmpty
	ErrToolInvalid = tooldefinitions.ErrToolInvalid
)

func New(renderer ToolSignatureRenderer, tools []modelinput.ToolSignature, debug gai.ObservationSink, options ...Option) (*Source, error) {
	return tooldefinitions.New(renderer, tools, debug, options...)
}

func WithUsageProtocol(protocol string) Option { return tooldefinitions.WithUsageProtocol(protocol) }

func WithToolChoice(choice ai.ToolChoice) Option { return tooldefinitions.WithToolChoice(choice) }
