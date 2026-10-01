package gemini

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync/atomic"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestSystemInstructionsIgnoreOptionalExtensions(t *testing.T) {
	for _, placement := range []string{"message", "part", "both"} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/streaming_%t", placement, streaming), func(t *testing.T) {
				requests := make(chan []byte, 1)
				model := emptyPartModel(t, func(w http.ResponseWriter, r *http.Request) {
					body, err := io.ReadAll(r.Body)
					if err != nil {
						t.Error(err)
						w.WriteHeader(http.StatusBadRequest)
						return
					}
					requests <- body
					response := `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`
					if streaming {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = fmt.Fprintf(w, "data: %s\n\n", response)
					} else {
						w.Header().Set("Content-Type", "application/json")
						_, _ = io.WriteString(w, response)
					}
				})
				req := systemExtensionRequest(placement, false)
				if streaming {
					for token := range model.GenerateStream(t.Context(), req) {
						if token.Err != nil {
							t.Fatal(token.Err)
						}
					}
				} else if _, err := model.Generate(t.Context(), req); err != nil {
					t.Fatal(err)
				}
				var raw []byte
				select {
				case raw = <-requests:
				default:
					t.Fatal("optional extensions prevented the provider request")
				}
				var request struct {
					SystemInstruction struct {
						Parts []struct{ Text string } `json:"parts"`
					} `json:"systemInstruction"`
					Contents []struct {
						Role  string
						Parts []struct{ Text string }
					} `json:"contents"`
				}
				if err := json.Unmarshal(raw, &request); err != nil {
					t.Fatal(err)
				}
				system := request.SystemInstruction.Parts
				if len(system) != 2 || system[0].Text != "Follow these rules." || system[1].Text != `{"style":"concise"}` {
					t.Fatalf("system instructions = %s", raw)
				}
				if len(request.Contents) != 1 || request.Contents[0].Role != "user" || len(request.Contents[0].Parts) != 1 || request.Contents[0].Parts[0].Text != "Question" {
					t.Fatalf("system instructions were not kept separate from user content: %s", raw)
				}
				if bytes.Contains(raw, []byte("optional-metadata")) {
					t.Fatalf("unsupported optional extension leaked into the transport: %s", raw)
				}
			})
		}
	}
}

func TestSystemInstructionsRejectRequiredExtensionsBeforeTransport(t *testing.T) {
	for _, placement := range []string{"message", "part"} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/streaming_%t", placement, streaming), func(t *testing.T) {
				var requests atomic.Int32
				model := emptyPartModel(t, func(w http.ResponseWriter, _ *http.Request) {
					requests.Add(1)
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, `{}`)
				})
				req := systemExtensionRequest(placement, true)
				var err error
				if streaming {
					for token := range model.GenerateStream(t.Context(), req) {
						if token.Err != nil {
							err = token.Err
						}
					}
				} else {
					_, err = model.Generate(t.Context(), req)
				}
				if !errors.Is(err, ai.ErrUnsupportedCapability) || requests.Load() != 0 {
					t.Fatalf("error = %v, requests = %d; want unsupported capability before transport", err, requests.Load())
				}
			})
		}
	}
}

func systemExtensionRequest(placement string, required bool) ai.AIRequest {
	extension := ai.Extension{Namespace: "example", Type: "unknown", Data: json.RawMessage(`"optional-metadata"`), Required: required}
	system := ai.Message{Role: ai.RoleSystem, Parts: []ai.ContentPart{
		{Kind: ai.ContentText, Text: "Follow these rules."},
		{Kind: ai.ContentJSON, JSON: json.RawMessage(`{"style":"concise"}`)},
	}}
	if placement == "message" || placement == "both" {
		system.Extensions = []ai.Extension{extension}
	}
	if placement == "part" || placement == "both" {
		for i := range system.Parts {
			system.Parts[i].Extensions = []ai.Extension{extension}
		}
	}
	return ai.AIRequest{Messages: []ai.Message{system, ai.TextMessage(ai.RoleUser, "Question")}}
}

func TestSystemInstructionsOmitOptionalExtensionOnlyParts(t *testing.T) {
	extension := ai.Extension{Namespace: "example", Type: "unknown", Data: json.RawMessage(`"optional"`)}
	req := ai.AIRequest{Messages: []ai.Message{
		{Role: ai.RoleSystem, Parts: []ai.ContentPart{{Kind: ai.ContentExtension, Extensions: []ai.Extension{extension}}}},
		ai.TextMessage(ai.RoleUser, "Question"),
	}}
	config, err := buildGenerateContentConfig(req)
	if err != nil {
		t.Fatal(err)
	}
	if config != nil && config.SystemInstruction != nil {
		t.Fatalf("ignored optional metadata created empty system instructions: %#v", config.SystemInstruction)
	}
	contents, err := nativeContents(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(contents) != 1 || contents[0].Role != "user" || len(contents[0].Parts) != 1 || contents[0].Parts[0].Text != "Question" {
		t.Fatalf("contents = %#v", contents)
	}
}

func TestSystemInstructionsRejectRequiredThoughtSignatures(t *testing.T) {
	for _, placement := range []string{"message", "part"} {
		t.Run(placement, func(t *testing.T) {
			req := systemExtensionRequest(placement, true)
			signature := thoughtExtensions([]byte("signed-state"))
			if placement == "message" {
				req.Messages[0].Extensions = signature
			} else {
				req.Messages[0].Parts[0].Extensions = signature
			}
			_, err := buildGenerateContentConfig(req)
			if !errors.Is(err, ai.ErrUnsupportedCapability) {
				t.Fatalf("error = %v, want unsupported capability for system thought signature", err)
			}
		})
	}
}
