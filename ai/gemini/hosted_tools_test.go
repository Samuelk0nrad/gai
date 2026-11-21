package gemini

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
	"google.golang.org/genai"
)

const (
	serverCallPart     = `{"toolCall":{"id":"call_s1","toolType":"GOOGLE_SEARCH_WEB","args":{"query":"go 1.26 release"}},"thoughtSignature":"c2ln"}`
	serverResponsePart = `{"toolResponse":{"id":"call_s1","toolType":"GOOGLE_SEARCH_WEB","response":{"results":"opaque"}}}`
	answerPart         = `{"text":"Go 1.26 is out."}`
	grounding          = `"groundingMetadata":{"webSearchQueries":["go 1.26 release"],"groundingChunks":[{"web":{"uri":"https://go.dev/doc/go1.26","title":"go.dev"}}],"groundingSupports":[{"segment":{"startIndex":0,"endIndex":15,"text":"Go 1.26 is out."},"groundingChunkIndices":[0]}]}`
)

func hostedTestModel(t *testing.T, handler http.HandlerFunc) ai.Model {
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
	return model
}

func assertGeminiSearch(t *testing.T, message ai.Message, wantCalls int) {
	t.Helper()
	calls := message.HostedToolCalls()
	if len(calls) != wantCalls || calls[0].Provider != "google" || calls[0].Kind != ai.HostedToolWebSearch || calls[0].Query != "go 1.26 release" {
		t.Fatalf("hosted tool calls = %#v", calls)
	}
	citations := message.Citations()
	if len(citations) != 1 || citations[0].URL != "https://go.dev/doc/go1.26" || citations[0].Text != "Go 1.26 is out." {
		t.Fatalf("citations = %#v", citations)
	}
	if message.Text() != "Go 1.26 is out." {
		t.Fatalf("text = %q", message.Text())
	}
	if err := message.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestGenerateKeepsServerToolPartsAndGrounding(t *testing.T) {
	model := hostedTestModel(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[` + serverCallPart + `,` + serverResponsePart + `,` + answerPart + `]},"finishReason":"STOP",` + grounding + `}]}`))
	})
	response, err := model.(ai.ModelGenerator).Generate(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")}})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	// The search came back as a server-side call, so the grounding queries
	// are not counted a second time.
	assertGeminiSearch(t, response.Message, 1)
	if calls := response.Message.HostedToolCalls(); calls[0].ID != "call_s1" || calls[0].Name != "GOOGLE_SEARCH_WEB" {
		t.Fatalf("call = %#v", calls[0])
	}
}

func TestGenerateStreamCountsGroundingQueriesWithoutServerParts(t *testing.T) {
	model := hostedTestModel(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Go 1.26 \"}]}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"is out.\"}]},\"finishReason\":\"STOP\"," + grounding + "}]}\n\n"))
	})
	message := ai.Message{Role: ai.RoleAssistant}
	for token := range model.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")}}) {
		if token.Err != nil {
			t.Fatalf("stream error: %v", token.Err)
		}
		message.AppendToken(token)
	}
	assertGeminiSearch(t, message, 1)
	if calls := message.HostedToolCalls(); calls[0].Name != "google_search" || len(calls[0].Raw) != 0 {
		t.Fatalf("query call = %#v", calls[0])
	}
}

func TestGenerateStreamKeepsServerToolParts(t *testing.T) {
	model := hostedTestModel(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[" + serverCallPart + "," + serverResponsePart + "]}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[" + answerPart + "]},\"finishReason\":\"STOP\"," + grounding + "}]}\n\n"))
	})
	message := ai.Message{Role: ai.RoleAssistant}
	for token := range model.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")}}) {
		if token.Err != nil {
			t.Fatalf("stream error: %v", token.Err)
		}
		message.AppendToken(token)
	}
	assertGeminiSearch(t, message, 1)
}

func TestServerToolPartsReplayInOrder(t *testing.T) {
	var body map[string]any
	requests := 0
	model := hostedTestModel(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		w.Header().Set("Content-Type", "application/json")
		if requests == 1 {
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[` + serverCallPart + `,` + serverResponsePart + `,` + answerPart + `]},"finishReason":"STOP",` + grounding + `}]}`))
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode: %v", err)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},"finishReason":"STOP"}]}`))
	})
	request := ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")}}
	first, err := model.(ai.ModelGenerator).Generate(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	stored, _ := json.Marshal(first.Message)
	var restored ai.Message
	if err := json.Unmarshal(stored, &restored); err != nil {
		t.Fatal(err)
	}
	request.Messages = append(request.Messages, restored, ai.TextMessage(ai.RoleUser, "more"))
	if _, err := model.(ai.ModelGenerator).Generate(t.Context(), request); err != nil {
		t.Fatal(err)
	}
	contents := body["contents"].([]any)
	parts := contents[1].(map[string]any)["parts"].([]any)
	var kinds []string
	for _, part := range parts {
		object := part.(map[string]any)
		switch {
		case object["toolCall"] != nil:
			kinds = append(kinds, "toolCall")
			if object["thoughtSignature"] != "c2ln" {
				t.Fatalf("server call lost its thought signature: %#v", object)
			}
		case object["toolResponse"] != nil:
			kinds = append(kinds, "toolResponse")
		case object["text"] != nil:
			kinds = append(kinds, "text")
		default:
			kinds = append(kinds, "other")
		}
	}
	// The grounding part is metadata only and is not sent back.
	if strings.Join(kinds, ",") != "toolCall,toolResponse,text" {
		t.Fatalf("replayed parts = %v (%#v)", kinds, parts)
	}
}

func TestCodeExecutionPartsAreKept(t *testing.T) {
	message, err := mapCanonicalResponseJSON(t, `{"candidates":[{"content":{"role":"model","parts":[{"executableCode":{"language":"PYTHON","code":"print(1)"}},{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1"}},{"text":"1"}]}}]}`)
	if err != nil {
		t.Fatal(err)
	}
	calls := message.HostedToolCalls()
	if len(calls) != 1 || calls[0].Kind != ai.HostedToolCodeExecution || len(message.Parts) != 3 {
		t.Fatalf("message = %#v", message)
	}
	contents, err := nativeContents(ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "run"), message}})
	if err != nil {
		t.Fatal(err)
	}
	parts := contents[1].Parts
	if len(parts) != 3 || parts[0].ExecutableCode == nil || parts[1].CodeExecutionResult == nil || parts[2].Text != "1" {
		t.Fatalf("replayed parts = %#v", parts)
	}
}

func mapCanonicalResponseJSON(t *testing.T, raw string) (ai.Message, error) {
	t.Helper()
	var response genai.GenerateContentResponse
	if err := json.Unmarshal([]byte(raw), &response); err != nil {
		t.Fatal(err)
	}
	return mapCanonicalResponse(&response)
}

func groundingChunk(t *testing.T, raw string) string {
	t.Helper()
	return "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\".\"}]}," + raw + "}]}\n\n"
}

func TestGenerateStreamMergesGroundingAcrossChunks(t *testing.T) {
	first := `"groundingMetadata":{"webSearchQueries":["a"],"groundingChunks":[{"web":{"uri":"https://a.test","title":"A"}}],"groundingSupports":[{"segment":{"text":"one"},"groundingChunkIndices":[0]}]}`
	// Repeats the first chunk's data and adds one query, source and support.
	// The new support refers to its source by this chunk's own index.
	second := `"groundingMetadata":{"webSearchQueries":["a","b"],"groundingChunks":[{"web":{"uri":"https://b.test","title":"B"}},{"web":{"uri":"https://a.test","title":"A"}}],"groundingSupports":[{"segment":{"text":"one"},"groundingChunkIndices":[1]},{"segment":{"text":"two"},"groundingChunkIndices":[0]}]}`
	// Carries only new data.
	third := `"groundingMetadata":{"webSearchQueries":["c"],"groundingChunks":[{"web":{"uri":"https://c.test","title":"C"}}],"groundingSupports":[{"segment":{"text":"three"},"groundingChunkIndices":[0]}]}`
	model := hostedTestModel(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(groundingChunk(t, first) + groundingChunk(t, second) + groundingChunk(t, `"finishReason":"STOP",`+third)))
	})
	message := ai.Message{Role: ai.RoleAssistant}
	for token := range model.GenerateStream(t.Context(), ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "news")}}) {
		if token.Err != nil {
			t.Fatalf("stream error: %v", token.Err)
		}
		message.AppendToken(token)
	}
	var queries []string
	for _, call := range message.HostedToolCalls() {
		queries = append(queries, call.Query)
	}
	if strings.Join(queries, ",") != "a,b,c" {
		t.Fatalf("queries = %v", queries)
	}
	var citations []string
	for _, citation := range message.Citations() {
		citations = append(citations, citation.Text+"="+citation.URL)
	}
	if strings.Join(citations, " ") != "one=https://a.test two=https://b.test three=https://c.test" {
		t.Fatalf("citations = %v", citations)
	}
}

func TestHostedPartWithRequiredSiblingIsRejected(t *testing.T) {
	message, err := mapCanonicalResponseJSON(t, `{"candidates":[{"content":{"role":"model","parts":[`+serverCallPart+`]}}]}`)
	if err != nil {
		t.Fatal(err)
	}
	message.Parts[0].Extensions = append(message.Parts[0].Extensions, ai.Extension{Namespace: "other", Type: "state", Data: json.RawMessage(`"s"`), Required: true})
	_, err = nativeContents(ai.AIRequest{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "x"), message}})
	if !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("error = %v", err)
	}
}

// Generate and GenerateStream count repeated queries the same way.
func TestGenerateCountsRepeatedGroundingQueriesOnce(t *testing.T) {
	message, err := mapCanonicalResponseJSON(t, `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},"groundingMetadata":{"webSearchQueries":["x","x"]}}]}`)
	if err != nil {
		t.Fatal(err)
	}
	if calls := message.HostedToolCalls(); len(calls) != 1 {
		t.Fatalf("calls = %#v", calls)
	}
}
