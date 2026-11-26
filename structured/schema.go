// Package structured generates, validates, and decodes provider-neutral JSON Schema output.
package structured

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strconv"
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
	value, err := decodeValue(decoder, nil)
	if err != nil {
		return nil, err
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		if err == nil {
			return nil, errors.New("multiple JSON values")
		}
		return nil, err
	}
	return value, nil
}

func decodeValue(decoder *json.Decoder, path []string) (any, error) {
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return token, nil
	}
	switch delim {
	case '{':
		object := make(map[string]any)
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return nil, err
			}
			key, ok := keyToken.(string)
			if !ok {
				return nil, fmt.Errorf("object key is %T, want string", keyToken)
			}
			keyPath := appendPath(path, key)
			if _, exists := object[key]; exists {
				return nil, &duplicateKeyError{path: pointer(keyPath), key: key}
			}
			value, err := decodeValue(decoder, keyPath)
			if err != nil {
				return nil, err
			}
			object[key] = value
		}
		if _, err := decoder.Token(); err != nil {
			return nil, err
		}
		return object, nil
	case '[':
		array := make([]any, 0)
		for decoder.More() {
			value, err := decodeValue(decoder, appendPath(path, strconv.Itoa(len(array))))
			if err != nil {
				return nil, err
			}
			array = append(array, value)
		}
		if _, err := decoder.Token(); err != nil {
			return nil, err
		}
		return array, nil
	default:
		return nil, fmt.Errorf("unexpected JSON delimiter %q", delim)
	}
}

func appendPath(path []string, part string) []string {
	result := make([]string, len(path)+1)
	copy(result, path)
	result[len(path)] = part
	return result
}

type duplicateKeyError struct {
	path string
	key  string
}

func (e *duplicateKeyError) Error() string       { return fmt.Sprintf("duplicate object key %q", e.key) }
func (e *duplicateKeyError) JSONPointer() string { return e.path }
