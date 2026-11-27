package structured_test

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/Samuelk0nrad/gai/structured"
)

type generatedAnswer struct {
	Name  string   `json:"name"`
	Score *int     `json:"score"`
	Tags  []string `json:"tags"`
}

func TestForGeneratesSchemaAndDecodesTypedValue(t *testing.T) {
	schema, err := structured.For[generatedAnswer]("answer")
	if err != nil {
		t.Fatalf("For: %v", err)
	}
	if schema.Name != "answer" || len(schema.JSON) == 0 {
		t.Fatalf("schema = %#v", schema)
	}

	score := 7
	want := generatedAnswer{Name: "Ada", Score: &score, Tags: []string{"go"}}
	result, err := (structured.Spec[generatedAnswer]{Schema: schema}).Decode(json.RawMessage(`{"name":"Ada","score":7,"tags":["go"]}`))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if result.Value.Name != want.Name || result.Value.Score == nil || *result.Value.Score != score || len(result.Value.Tags) != 1 || result.Value.Tags[0] != "go" {
		t.Fatalf("value = %#v, want %#v", result.Value, want)
	}
	if string(result.JSON) != `{"name":"Ada","score":7,"tags":["go"]}` {
		t.Fatalf("JSON = %s", result.JSON)
	}
}

func TestRawPreservesLargeNumericBounds(t *testing.T) {
	schema, err := structured.Raw("large integer", json.RawMessage(`{"type":"integer","maximum":9007199254740993}`))
	if err != nil {
		t.Fatalf("Raw: %v", err)
	}

	if _, err := (structured.Spec[json.Number]{Schema: schema}).Decode(json.RawMessage(`9007199254740993`)); err != nil {
		t.Fatalf("Decode at exact maximum: %v", err)
	}
}

func TestRawValidatesUnionNullableAndReportsJSONPointer(t *testing.T) {
	schema, err := structured.Raw("answer", json.RawMessage(`{
		"type":"object",
		"properties":{"items":{"type":"array","items":{"type":"object","properties":{"value":{"anyOf":[{"type":"integer","minimum":1},{"type":"null"}]}},"required":["value"]}}},
		"required":["items"]
	}`))
	if err != nil {
		t.Fatalf("Raw: %v", err)
	}
	type item struct {
		Value *int `json:"value"`
	}
	type answer struct {
		Items []item `json:"items"`
	}
	spec := structured.Spec[answer]{Schema: schema}

	if _, err := spec.Decode(json.RawMessage(`{"items":[{"value":null}]}`)); err != nil {
		t.Fatalf("nullable Decode: %v", err)
	}
	_, err = spec.Decode(json.RawMessage(`{"items":[{"value":0}]}`))
	var validation *structured.ValidationError
	if !errors.As(err, &validation) {
		t.Fatalf("error = %T %v, want ValidationError", err, err)
	}
	if validation.Path != "/items/0/value" {
		t.Fatalf("path = %q, want /items/0/value", validation.Path)
	}
	if validation.Keyword == "" {
		t.Fatal("validation keyword is empty")
	}
}

func TestDecodeErrorIncludesPathAndRejectsTrailingJSON(t *testing.T) {
	schema, err := structured.Raw("answer", json.RawMessage(`{"type":"object","properties":{"count":{}},"required":["count"]}`))
	if err != nil {
		t.Fatal(err)
	}
	type answer struct {
		Count int `json:"count"`
	}

	_, err = (structured.Spec[answer]{Schema: schema}).Decode(json.RawMessage(`{"count":"many"}`))
	var decode *structured.DecodeError
	if !errors.As(err, &decode) || decode.Path != "/count" {
		t.Fatalf("error = %T %#v, want DecodeError at /count", err, err)
	}

	_, err = (structured.Spec[answer]{Schema: schema}).Decode(json.RawMessage(`{"count":1} {"count":2}`))
	if !errors.As(err, &decode) || decode.Path != "" {
		t.Fatalf("trailing error = %T %#v, want DecodeError at document root", err, err)
	}
}

func TestDecodeErrorUsesInputLocationForDottedKeysAndArrayElements(t *testing.T) {
	schema, err := structured.Raw("answer", json.RawMessage(`{"type":"object"}`))
	if err != nil {
		t.Fatal(err)
	}

	t.Run("dotted key", func(t *testing.T) {
		type answer struct {
			Value int `json:"a.b"`
		}
		_, err := (structured.Spec[answer]{Schema: schema}).Decode(json.RawMessage(`{"a.b":"many"}`))
		var decode *structured.DecodeError
		if !errors.As(err, &decode) || decode.Path != "/a.b" {
			t.Fatalf("error = %T %#v, want DecodeError at /a.b", err, err)
		}
	})

	t.Run("array element", func(t *testing.T) {
		type item struct {
			Count int `json:"count"`
		}
		type answer struct {
			Items []item `json:"items"`
		}
		_, err := (structured.Spec[answer]{Schema: schema}).Decode(json.RawMessage(`{"items":[{"count":"many"}]}`))
		var decode *structured.DecodeError
		if !errors.As(err, &decode) || decode.Path != "/items/0/count" {
			t.Fatalf("error = %T %#v, want DecodeError at /items/0/count", err, err)
		}
	})
}

func TestDecodeRejectsDuplicateObjectKeysBeforeTypedDecode(t *testing.T) {
	schema, err := structured.Raw("answer", json.RawMessage(`{
		"type":"object",
		"properties":{"payload":{"type":"object","properties":{"count":{"type":"integer"},"extra":{"type":"string","maxLength":3}},"required":["count"]}},
		"required":["payload"]
	}`))
	if err != nil {
		t.Fatal(err)
	}
	type payload struct {
		Count int    `json:"count"`
		Extra string `json:"extra"`
	}
	type answer struct {
		Payload payload `json:"payload"`
	}

	_, err = (structured.Spec[answer]{Schema: schema}).Decode(json.RawMessage(`{"payload":{"extra":"too long"},"payload":{"count":1}}`))
	var decode *structured.DecodeError
	if !errors.As(err, &decode) {
		t.Fatalf("error = %T %v, want DecodeError", err, err)
	}
	if decode.Path != "/payload" || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("error = %#v, want duplicate-key error at /payload", err)
	}

	_, err = (structured.Spec[any]{Schema: schema}).Decode(json.RawMessage(`{"payload":{"items":[{"a/b":1,"a/b":2}]}}`))
	if !errors.As(err, &decode) || decode.Path != "/payload/items/0/a~1b" {
		t.Fatalf("nested duplicate error = %T %#v, want DecodeError at /payload/items/0/a~1b", err, err)
	}
}

func TestDecodeBoundsNestingDepth(t *testing.T) {
	schema, err := structured.Raw("any", json.RawMessage(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	spec := structured.Spec[any]{Schema: schema}

	const maxDepth = 1000
	atLimit := json.RawMessage(strings.Repeat("[", maxDepth) + "0" + strings.Repeat("]", maxDepth))
	if _, err := spec.Decode(atLimit); err != nil {
		t.Fatalf("Decode at nesting limit: %v", err)
	}

	aboveLimit := json.RawMessage(strings.Repeat("[", maxDepth+1) + "0" + strings.Repeat("]", maxDepth+1))
	_, err = spec.Decode(aboveLimit)
	var decode *structured.DecodeError
	if !errors.As(err, &decode) || !strings.Contains(err.Error(), "maximum nesting depth") {
		t.Fatalf("error = %T %v, want maximum-nesting-depth DecodeError", err, err)
	}
}

func TestValidationPathsUseRFC6901WithoutMutatingValidatorError(t *testing.T) {
	schema, err := structured.Raw("answer", json.RawMessage(`{"type":"object","properties":{"a/b":{"type":"integer"}}}`))
	if err != nil {
		t.Fatal(err)
	}
	_, err = (structured.Spec[map[string]any]{Schema: schema}).Decode(json.RawMessage(`{"a/b":"many"}`))
	var validation *structured.ValidationError
	if !errors.As(err, &validation) {
		t.Fatalf("error = %T %v, want ValidationError", err, err)
	}
	if validation.Path != "/a~1b" {
		t.Fatalf("path = %q, want /a~1b", validation.Path)
	}
	if got := validation.Err.Error(); !strings.Contains(got, "/a~1b") || strings.Contains(got, "/a~01b") {
		t.Fatalf("wrapped validator error path was mutated: %q", got)
	}

	rootSchema, err := structured.Raw("root", json.RawMessage(`{"type":"integer"}`))
	if err != nil {
		t.Fatal(err)
	}
	_, err = (structured.Spec[int]{Schema: rootSchema}).Decode(json.RawMessage(`"many"`))
	if !errors.As(err, &validation) || validation.Path != "" {
		t.Fatalf("root validation error = %T %#v, want empty RFC 6901 pointer", err, err)
	}
}

func TestSemanticValidationErrorIsCopiedPerDecode(t *testing.T) {
	schema, err := structured.Raw("answer", json.RawMessage(`{"type":"object"}`))
	if err != nil {
		t.Fatal(err)
	}
	reused := &structured.ValidationError{Path: "/value", Keyword: "semantic", Err: errors.New("invalid")}
	spec := structured.Spec[map[string]any]{Schema: schema, Validate: func(map[string]any) error { return reused }}

	_, firstErr := spec.Decode(json.RawMessage(`{"value":1}`))
	_, secondErr := spec.Decode(json.RawMessage(`{"value":2}`))
	var first, second *structured.ValidationError
	if !errors.As(firstErr, &first) || !errors.As(secondErr, &second) {
		t.Fatalf("errors = %T, %T; want ValidationError", firstErr, secondErr)
	}
	if first == reused || second == reused || first == second {
		t.Fatal("semantic validation error was not copied")
	}
	if reused.JSON != nil || string(first.JSON) != `{"value":1}` || string(second.JSON) != `{"value":2}` {
		t.Fatalf("JSON snapshots: reused=%s first=%s second=%s", reused.JSON, first.JSON, second.JSON)
	}
}

func TestRawAndResultOwnTheirJSON(t *testing.T) {
	rawSchema := json.RawMessage(`{"type":"object"}`)
	schema, err := structured.Raw("answer", rawSchema)
	if err != nil {
		t.Fatal(err)
	}
	rawSchema[0] = '['
	if schema.JSON[0] != '{' {
		t.Fatal("schema retained caller bytes")
	}

	rawValue := json.RawMessage(`{}`)
	result, err := (structured.Spec[map[string]any]{Schema: schema}).Decode(rawValue)
	if err != nil {
		t.Fatal(err)
	}
	rawValue[0] = '['
	if result.JSON[0] != '{' {
		t.Fatal("result retained caller bytes")
	}
}
