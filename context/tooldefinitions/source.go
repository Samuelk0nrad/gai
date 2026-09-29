// Package tooldefinitions provides prompt context for agent tool definitions.
package tooldefinitions

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

var (
	ErrToolsEmpty  = errors.New("tool definition source requires at least one tool")
	ErrToolInvalid = errors.New("invalid tool definition")
)

const toolUseProtocol = `When a tool is required, output each tool call as a standalone JSON object using exactly this shape:
{"type":"function","name":"<tool-name>","arguments":{...}}

The name must match a listed tool, type must be exactly "function" and arguments must match its signature. Do not include an id, do not wrap the JSON in Markdown, and separate multiple calls with a blank line. If no tool is needed, respond normally. Do not repeat a completed tool call when its result is already present.`

// Source renders tool signatures and their text-based invocation protocol as
// prompt context. Executable loop tools satisfy context.ToolSignature but are
// not required by this package.
type Source struct {
	tools         []gaictx.ToolSignature
	renderer      gaictx.Renderer
	debug         gai.ObservationSink
	usageProtocol string
	toolChoice    ai.ToolChoice
}

var _ gaictx.ContextSource = (*Source)(nil)

// Option configures a Source.
type Option func(*Source) error

// WithUsageProtocol overrides the default tool-call instruction prompt.
func WithUsageProtocol(protocol string) Option {
	return func(source *Source) error {
		protocol = strings.TrimSpace(protocol)
		if protocol == "" {
			return fmt.Errorf("%w: usage protocol is empty", ErrToolInvalid)
		}
		source.usageProtocol = protocol
		return nil
	}
}

// WithToolChoice configures text-transport tool-choice instructions.
func WithToolChoice(choice ai.ToolChoice) Option {
	return func(source *Source) error {
		source.toolChoice = choice
		source.toolChoice.Names = append([]string(nil), choice.Names...)
		return nil
	}
}

func toolChoiceInstruction(choice ai.ToolChoice) string {
	if choice.Mode != ai.ToolChoiceRequired {
		return ""
	}
	if len(choice.Names) == 0 {
		return "You must make at least one tool call before producing a normal response."
	}
	return "You must make at least one tool call before producing a normal response. You may call only these selected tools: " + strings.Join(choice.Names, ", ") + "."
}

// New creates a context source from tool signatures and a renderer. The slice is
// copied so callers can safely reuse or modify their input slice after construction.
func New(renderer gaictx.Renderer, tools []gaictx.ToolSignature, debug gai.ObservationSink, options ...Option) (*Source, error) {
	if len(tools) == 0 {
		return nil, ErrToolsEmpty
	}
	signatures := make([]gaictx.ToolSignature, len(tools))
	for index, signature := range tools {
		if isNilToolSignature(signature) {
			return nil, fmt.Errorf("%w: tool at index %d is nil", ErrToolInvalid, index)
		}
		signatures[index] = signature
	}
	source := &Source{
		tools:         signatures,
		renderer:      renderer,
		debug:         debug,
		usageProtocol: toolUseProtocol,
	}
	for _, option := range options {
		if option == nil {
			return nil, fmt.Errorf("%w: option is nil", ErrToolInvalid)
		}
		if err := option(source); err != nil {
			return nil, err
		}
	}
	return source, nil
}

func (s *Source) Name() string {
	return "tool_definitions"
}

func (s *Source) Function(ctx context.Context, tokenBudget int) (part gaictx.Part, err error) {
	ctx, observer := newObserver(ctx, s.debug, len(s.tools), tokenBudget)
	defer func() {
		observer.Finish(err)
	}()

	observer.Started(ctx)
	if err = ctx.Err(); err != nil {
		observer.Failed(ctx, "context", err)
		return nil, err
	}

	definitions := make([]definition, 0, len(s.tools))
	for index, tool := range s.tools {
		definition, definitionErr := definitionFromTool(tool, index)
		if definitionErr != nil {
			err = definitionErr
			observer.Failed(ctx, "validate_tool", err)
			return nil, err
		}
		definitions = append(definitions, definition)
	}
	sort.Slice(definitions, func(i, j int) bool {
		return definitions[i].name < definitions[j].name
	})

	renderer := s.renderer
	if renderer == nil {
		renderer = &gaictx.XMLRenderer{}
	}
	signatures, err := renderer.RenderToolSignatures(append([]gaictx.ToolSignature(nil), s.tools...))
	if err != nil {
		observer.Failed(ctx, "render_tool_signatures", err)
		return nil, err
	}
	usageProtocol := s.usageProtocol
	if instruction := toolChoiceInstruction(s.toolChoice); instruction != "" {
		usageProtocol += "\n\n" + instruction
	}
	part = newPart(definitions, usageProtocol, usageProtocol+signatures)
	observer.Succeeded(ctx, definitionNames(definitions))
	return part, nil
}

type definition struct {
	name        string
	description string
	parameters  string
}

func definitionFromTool(tool gaictx.ToolSignature, index int) (definition, error) {
	if isNilToolSignature(tool) {
		return definition{}, fmt.Errorf("%w: tool at index %d is nil", ErrToolInvalid, index)
	}
	result := definition{
		name:        strings.TrimSpace(tool.Name()),
		description: strings.TrimSpace(tool.Description()),
	}
	if result.name == "" {
		return definition{}, fmt.Errorf("%w: tool at index %d has an empty name", ErrToolInvalid, index)
	}
	if result.description == "" {
		return definition{}, fmt.Errorf("%w: tool %q has an empty description", ErrToolInvalid, result.name)
	}
	params, err := tool.Params().JSONSchema()
	if err != nil {
		return definition{}, fmt.Errorf("%w: tool %q has invalid parameters", ErrToolInvalid, result.name)
	}
	result.parameters = string(params)
	return result, nil
}

func isNilToolSignature(signature gaictx.ToolSignature) bool {
	if signature == nil {
		return true
	}
	value := reflect.ValueOf(signature)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

func definitionNames(definitions []definition) []string {
	names := make([]string, 0, len(definitions))
	for _, definition := range definitions {
		names = append(names, definition.name)
	}
	return names
}

type part struct {
	definitions   []definition
	usageProtocol string
	text          gaictx.TextPart
}

func newPart(definitions []definition, usageProtocol, tokenText string) *part {
	return &part{
		definitions:   definitions,
		usageProtocol: usageProtocol,
		text:          gaictx.NewTextPart(tokenText),
	}
}

func (p *part) Name() string {
	return "tool_definitions"
}

func (p *part) Tokens(ctx context.Context, counter ai.TokenCounter) (int, error) {
	return p.text.Tokens(ctx, counter)
}

func (p *part) Render(ctx context.Context) (gaictx.RenderNode, error) {
	if err := ctx.Err(); err != nil {
		return gaictx.RenderNode{}, err
	}
	node := gaictx.RenderNode{
		Type: "tools",
		Children: []gaictx.RenderNode{
			{Type: "tool_usage", Value: p.usageProtocol},
		},
	}
	for _, definition := range p.definitions {
		node.Children = append(node.Children, gaictx.RenderNode{
			Type:   "tool",
			Fields: []gaictx.RenderField{{Key: "name", Value: definition.name}},
			Children: []gaictx.RenderNode{
				{Type: "tool-name", Value: definition.name},
				{Type: "description", Value: definition.description},
				{Type: "signature", Value: definition.parameters},
			},
		})
	}
	return node, nil
}
