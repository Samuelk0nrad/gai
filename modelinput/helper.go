package modelinput

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// SystemPrompt loads a supported prompt file and appends it as a system instruction.
func (b *Builder) SystemPrompt(ctx context.Context, path string) error {
	part, err := LoadPromptFromFile(path)
	if err != nil {
		return err
	}

	b.AppendSystemInstructions(ctx, part)
	return nil
}

// LoadPromptFromFile loads a Markdown or text file as a trimmed TextPart.
func LoadPromptFromFile(path string) (Part, error) {
	cleanPath := strings.TrimSpace(path)
	if cleanPath == "" {
		return nil, ErrPromptPathEmpty
	}

	ext := strings.ToLower(filepath.Ext(cleanPath))
	if ext != ".md" && ext != ".txt" {
		return nil, fmt.Errorf("%w: %s", ErrPromptFileType, cleanPath)
	}

	content, err := os.ReadFile(cleanPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("%w: %s", ErrPromptMissing, cleanPath)
		}
		return nil, err
	}

	part := NewTextPart(strings.TrimSpace(string(content)))
	return part, nil
}
