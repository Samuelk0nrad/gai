package gemini

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
	"google.golang.org/genai"
)

func TestGenerateIgnoresEmptyPartsAndPreservesCompletion(t *testing.T) {
	for _, empty := range []struct{ name, raw string }{{"object", `{}`}, {"text", `{"text":""}`}, {"nil", `null`}, {"empty_metadata", `{"partMetadata":{}}`}} {
		for position := 0; position <= 2; position++ {
			t.Run(fmt.Sprintf("%s/position_%d", empty.name, position), func(t *testing.T) {
				parts := emptyPartFixture(position, json.RawMessage(empty.raw))
				model := emptyPartModel(t, func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write(emptyPartResponse(t, parts, true))
				})
				response, err := model.Generate(t.Context(), ai.AIRequest{Prompt: "hello"})
				if err != nil {
					t.Fatal(err)
				}
				if response.Text != "hello world" || response.Message.Text() != "hello world" || len(response.Message.Parts) != 2 || response.FinishReason != "STOP" || response.InputTokens != 11 || response.OutputTokens != 7 || response.ReasoningTokens != 3 {
					t.Fatalf("response = %#v", response)
				}
				var raw struct {
					ResponseID string `json:"responseId"`
				}
				if err := json.Unmarshal(response.Raw, &raw); err != nil || raw.ResponseID != "empty-part-response" {
					t.Fatalf("raw response = %s, error = %v", response.Raw, err)
				}
			})
		}
	}
}

func TestGenerateStreamIgnoresEmptyPartsAndPreservesCompletion(t *testing.T) {
	for _, empty := range []struct{ name, raw string }{{"object", `{}`}, {"text", `{"text":""}`}, {"nil", `null`}, {"empty_metadata", `{"partMetadata":{}}`}} {
		for position := 0; position <= 2; position++ {
			t.Run(fmt.Sprintf("%s/position_%d", empty.name, position), func(t *testing.T) {
				parts := emptyPartFixture(position, json.RawMessage(empty.raw))
				model := emptyPartModel(t, func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "text/event-stream")
					for i, part := range parts {
						_, _ = fmt.Fprintf(w, "data: %s\n\n", emptyPartResponse(t, []json.RawMessage{part}, i == len(parts)-1))
					}
				})
				var text strings.Builder
				var completion *ai.Completion
				var completions, textTokens int
				for token := range model.GenerateStream(t.Context(), ai.AIRequest{Prompt: "hello"}) {
					if token.Err != nil {
						t.Fatal(token.Err)
					}
					switch token.Type {
					case ai.TokenTypeText:
						textTokens++
						text.WriteString(token.Text)
					case ai.TokenTypeCompletion:
						completions++
						completion = token.Completion
					default:
						t.Fatalf("unexpected token: %#v", token)
					}
				}
				if text.String() != "hello world" || textTokens != 2 || completions != 1 || completion == nil {
					t.Fatalf("text = %q, text tokens = %d, completions = %d, completion = %#v", text.String(), textTokens, completions, completion)
				}
				if completion.RequestID != "empty-part-response" || completion.Model != "gemini-test" || completion.FinishReason != "STOP" || !completion.UsageReported || completion.Usage.InputTokens != 11 || completion.Usage.OutputTokens != 7 || completion.Usage.ReasoningTokens != 3 || completion.Usage.CachedTokens != 2 || !json.Valid(completion.Raw) {
					t.Fatalf("completion = %#v", completion)
				}
			})
		}
	}
}

func TestUnsupportedPartPayloadsRemainErrors(t *testing.T) {
	cases := []struct {
		name string
		part *genai.Part
	}{
		{"media_resolution", &genai.Part{MediaResolution: &genai.PartMediaResolution{Level: genai.PartMediaResolutionLevelMediaResolutionHigh}}},
		{"code_result", &genai.Part{CodeExecutionResult: &genai.CodeExecutionResult{Output: "output"}}},
		{"executable_code", &genai.Part{ExecutableCode: &genai.ExecutableCode{Code: "print(1)"}}},
		{"file", &genai.Part{FileData: &genai.FileData{FileURI: "gs://bucket/image.png", MIMEType: "image/png"}}},
		{"function_response", &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "call_1", Name: "lookup", Response: map[string]any{"output": "result"}}}},
		{"inline_data", &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1, 2, 3}}}},
		{"video_metadata", &genai.Part{VideoMetadata: &genai.VideoMetadata{EndOffset: time.Second}}},
		{"server_tool_call", &genai.Part{ToolCall: &genai.ToolCall{ID: "server_call"}}},
		{"server_tool_response", &genai.Part{ToolResponse: &genai.ToolResponse{ID: "server_call", Response: map[string]any{"output": "result"}}}},
		{"part_metadata", &genai.Part{PartMetadata: map[string]any{"source": "document"}}},
		{"audio_transcription", &genai.Part{AudioTranscription: &genai.Transcription{Text: "transcript"}}},
		{"media_processing", &genai.Part{MediaProcessing: genai.MediaProcessingAgentic}},
		{"thought_without_content", &genai.Part{Thought: true}},
		{"text_with_inline_data", &genai.Part{Text: "caption", InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}}}},
		{"function_call_with_file", &genai.Part{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookup", Args: map[string]any{}}, FileData: &genai.FileData{FileURI: "gs://bucket/image.png", MIMEType: "image/png"}}},
		{"signature_with_transcription", &genai.Part{ThoughtSignature: []byte("signature"), AudioTranscription: &genai.Transcription{Text: "transcript"}}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			_, err := mapCanonicalResponse(&genai.GenerateContentResponse{Candidates: []*genai.Candidate{{Content: &genai.Content{Parts: []*genai.Part{test.part}}}}})
			if !errors.Is(err, ai.ErrUnsupportedCapability) {
				t.Fatalf("canonical error = %v, want unsupported capability", err)
			}
			part, err := json.Marshal(test.part)
			if err != nil {
				t.Fatal(err)
			}
			model := emptyPartModel(t, func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = fmt.Fprintf(w, "data: %s\n\n", emptyPartResponse(t, []json.RawMessage{part}, true))
			})
			var streamError error
			for token := range model.GenerateStream(t.Context(), ai.AIRequest{Prompt: "hello"}) {
				if token.Type == ai.TokenTypeErr {
					streamError = token.Err
				}
			}
			if !errors.Is(streamError, ai.ErrUnsupportedCapability) {
				t.Fatalf("stream error = %v, want unsupported capability for %s", streamError, part)
			}
		})
	}
}

func TestSignatureOnlyPartsRemainCanonical(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprintf("streaming_%t", streaming), func(t *testing.T) {
			signature := []byte("opaque-signature")
			part, err := json.Marshal(&genai.Part{ThoughtSignature: signature})
			if err != nil {
				t.Fatal(err)
			}
			model := emptyPartModel(t, func(w http.ResponseWriter, _ *http.Request) {
				response := emptyPartResponse(t, []json.RawMessage{part}, true)
				if streaming {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = fmt.Fprintf(w, "data: %s\n\n", response)
				} else {
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write(response)
				}
			})
			var message ai.Message
			if streaming {
				message.Role = ai.RoleAssistant
				for token := range model.GenerateStream(t.Context(), ai.AIRequest{Prompt: "hello"}) {
					if token.Err != nil {
						t.Fatal(token.Err)
					}
					message.AppendToken(token)
				}
			} else {
				response, err := model.Generate(t.Context(), ai.AIRequest{Prompt: "hello"})
				if err != nil {
					t.Fatal(err)
				}
				message = response.Message
			}
			if len(message.Parts) != 1 || message.Parts[0].Kind != ai.ContentExtension || len(message.Parts[0].Extensions) != 1 {
				t.Fatalf("signature-only message = %#v", message)
			}
			ext := message.Parts[0].Extensions[0]
			var restored []byte
			if err := json.Unmarshal(ext.Data, &restored); err != nil || string(restored) != string(signature) || ext.Namespace != "google" || ext.Type != "thought_signature" || !ext.Required {
				t.Fatalf("signature extension = %#v, error = %v", ext, err)
			}
		})
	}
}

func emptyPartFixture(position int, empty json.RawMessage) []json.RawMessage {
	parts := []json.RawMessage{json.RawMessage(`{"text":"hello "}`), json.RawMessage(`{"text":"world"}`)}
	parts = append(parts, nil)
	copy(parts[position+1:], parts[position:])
	parts[position] = empty
	return parts
}

func emptyPartResponse(t *testing.T, parts []json.RawMessage, completion bool) []byte {
	t.Helper()
	candidate := map[string]any{"content": map[string]any{"role": "model", "parts": parts}}
	response := map[string]any{"candidates": []any{candidate}}
	if completion {
		candidate["finishReason"] = "STOP"
		response["responseId"] = "empty-part-response"
		response["modelVersion"] = "gemini-test"
		response["usageMetadata"] = map[string]int{"promptTokenCount": 11, "candidatesTokenCount": 7, "thoughtsTokenCount": 3, "cachedContentTokenCount": 2}
	}
	raw, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func emptyPartModel(t *testing.T, handler http.HandlerFunc) *Model {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	provider := New("test-api-key", nil)
	provider.baseURL = server.URL
	provider.httpClient = server.Client()
	model, err := provider.Model("gemini-test")
	if err != nil {
		t.Fatal(err)
	}
	return model.(*Model)
}
