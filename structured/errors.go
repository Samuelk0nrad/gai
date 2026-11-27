package structured

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	validator "github.com/santhosh-tekuri/jsonschema/v6"
)

// ValidationError reports a JSON Schema or semantic validation failure.
// Path is an RFC 6901 JSON Pointer. The document root is the empty string.
type ValidationError struct {
	Path    string
	Keyword string
	JSON    json.RawMessage
	Err     error
}

func (e *ValidationError) Error() string {
	if e == nil {
		return "structured validation failed"
	}
	return fmt.Sprintf("structured validation failed at %s (%s): %v", normalizePath(e.Path), e.Keyword, e.Err)
}

func (e *ValidationError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// StructuredOutputErrorKind provides a stable low-cardinality retry class.
func (*ValidationError) StructuredOutputErrorKind() string { return "structured_validation" }

// DecodeError reports malformed JSON or a value that cannot decode into T.
// Path is an RFC 6901 JSON Pointer. The document root is the empty string.
type DecodeError struct {
	Path string
	JSON json.RawMessage
	Err  error
}

func (e *DecodeError) Error() string {
	if e == nil {
		return "structured decode failed"
	}
	return fmt.Sprintf("structured decode failed at %s: %v", normalizePath(e.Path), e.Err)
}

func (e *DecodeError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// StructuredOutputErrorKind provides a stable low-cardinality retry class.
func (*DecodeError) StructuredOutputErrorKind() string { return "structured_decode" }

func validationError(err error, raw json.RawMessage) error {
	var schemaErr *validator.ValidationError
	if !asValidationError(err, &schemaErr) {
		return &ValidationError{Path: "", Keyword: "schema", JSON: cloneJSON(raw), Err: err}
	}
	leaf := deepestValidationError(schemaErr)
	keyword := "schema"
	if leaf.ErrorKind != nil {
		path := leaf.ErrorKind.KeywordPath()
		if len(path) > 0 {
			keyword = path[len(path)-1]
		}
	}
	return &ValidationError{
		Path:    pointer(leaf.InstanceLocation),
		Keyword: keyword,
		JSON:    cloneJSON(raw),
		Err:     leaf,
	}
}

func asValidationError(err error, target **validator.ValidationError) bool {
	for err != nil {
		if typed, ok := err.(*validator.ValidationError); ok {
			*target = typed
			return true
		}
		type unwrapper interface{ Unwrap() error }
		u, ok := err.(unwrapper)
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}

func deepestValidationError(err *validator.ValidationError) *validator.ValidationError {
	best := err
	for _, cause := range err.Causes {
		candidate := deepestValidationError(cause)
		if len(candidate.InstanceLocation) > len(best.InstanceLocation) {
			best = candidate
		}
	}
	return best
}

func decodePath(raw json.RawMessage, err error) string {
	var typed *json.UnmarshalTypeError
	if !errors.As(err, &typed) || typed.Offset <= 0 {
		return ""
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	path, found, parseErr := pathAtOffset(decoder, nil, typed.Offset)
	if parseErr != nil || !found {
		return ""
	}
	return pointer(path)
}

func pathAtOffset(decoder *json.Decoder, path []string, offset int64) ([]string, bool, error) {
	start := decoder.InputOffset()
	token, err := decoder.Token()
	if err != nil {
		return nil, false, err
	}
	delim, isDelim := token.(json.Delim)
	if !isDelim {
		return path, offset > start && offset <= decoder.InputOffset(), nil
	}
	switch delim {
	case '{':
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return nil, false, err
			}
			key, ok := keyToken.(string)
			if !ok {
				return nil, false, fmt.Errorf("object key is %T, want string", keyToken)
			}
			if foundPath, found, err := pathAtOffset(decoder, appendPath(path, key), offset); err != nil || found {
				return foundPath, found, err
			}
		}
	case '[':
		for index := 0; decoder.More(); index++ {
			if foundPath, found, err := pathAtOffset(decoder, appendPath(path, strconv.Itoa(index)), offset); err != nil || found {
				return foundPath, found, err
			}
		}
	default:
		return nil, false, fmt.Errorf("unexpected JSON delimiter %q", delim)
	}
	if _, err := decoder.Token(); err != nil {
		return nil, false, err
	}
	return path, offset > start && offset <= decoder.InputOffset(), nil
}

func pointer(parts []string) string {
	if len(parts) == 0 {
		return ""
	}
	escaped := make([]string, len(parts))
	for i, part := range parts {
		part = strings.ReplaceAll(part, "~", "~0")
		escaped[i] = strings.ReplaceAll(part, "/", "~1")
	}
	return "/" + strings.Join(escaped, "/")
}

func normalizePath(path string) string {
	if path == "" {
		return "/"
	}
	return path
}

func cloneJSON(raw json.RawMessage) json.RawMessage {
	return append(json.RawMessage(nil), raw...)
}

func errorPath(err error) string {
	type pathError interface{ JSONPointer() string }
	var located pathError
	if errors.As(err, &located) {
		return located.JSONPointer()
	}
	return ""
}
