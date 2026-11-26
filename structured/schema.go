// Package structured generates, validates, and decodes provider-neutral JSON Schema output.
package structured

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"

	invopop "github.com/invopop/jsonschema"
	validator "github.com/santhosh-tekuri/jsonschema/v6"
)

// Schema is a named JSON Schema snapshot. JSON must be treated as immutable.
type Schema struct {
	Name string
	JSON json.RawMessage

	compiled *validator.Schema
	source   json.RawMessage
}

// For generates and compiles a JSON Schema for T.
func For[T any](name string) (Schema, error) {
	reflected := (&invopop.Reflector{Anonymous: true}).ReflectFromType(reflect.TypeFor[T]())
	raw, err := json.Marshal(reflected)
	if err != nil {
		return Schema{}, fmt.Errorf("marshal generated schema: %w", err)
	}
	return Raw(name, raw)
}

// Raw validates, compiles, and copies an exact JSON Schema. Unsupported
// keywords are retained rather than projected through a smaller schema model.
func Raw(name string, raw json.RawMessage) (Schema, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return Schema{}, errors.New("structured schema name is empty")
	}
	if len(raw) == 0 {
		return Schema{}, errors.New("structured schema is empty")
	}
	copyRaw := append(json.RawMessage(nil), raw...)
	compiled, err := compile(copyRaw)
	if err != nil {
		return Schema{}, fmt.Errorf("compile structured schema: %w", err)
	}
	return Schema{
		Name:     name,
		JSON:     append(json.RawMessage(nil), copyRaw...),
		compiled: compiled,
		source:   copyRaw,
	}, nil
}

func compile(raw json.RawMessage) (*validator.Schema, error) {
	document, err := decodeDocument(raw)
	if err != nil {
		return nil, err
	}
	compiler := validator.NewCompiler()
	compiler.UseLoader(rejectRemoteLoader{})
	if err := compiler.AddResource("schema.json", document); err != nil {
		return nil, err
	}
	return compiler.Compile("schema.json")
}

type rejectRemoteLoader struct{}

func (rejectRemoteLoader) Load(url string) (any, error) {
	return nil, fmt.Errorf("remote schema reference %q is not allowed", url)
}

func (s Schema) validateConfiguration() error {
	if s.compiled == nil || len(s.source) == 0 {
		return errors.New("structured schema is not initialized")
	}
	if !bytes.Equal(s.JSON, s.source) {
		return errors.New("structured schema JSON was mutated after construction")
	}
	return nil
}

func decodeDocument(raw json.RawMessage) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return nil, errors.New("multiple JSON values")
		}
		return nil, err
	}
	return value, nil
}
