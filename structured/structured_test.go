package structured_test

import (
	"encoding/json"
	"errors"
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
	if !errors.As(err, &decode) || decode.Path != "/" {
		t.Fatalf("trailing error = %T %#v, want DecodeError at /", err, err)
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
