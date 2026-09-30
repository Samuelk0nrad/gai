package openai

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/lace-ai/gai/ai"
	tiktoken "github.com/tiktoken-go/tokenizer"
)

func TestNewTokenizerResolvesKnownOpenAIModels(t *testing.T) {
	tests := []struct {
		model  string
		wantID string
		text   string
		want   int
	}{
		{model: "gpt-5", wantID: "openai.tiktoken-go/v0.8.1:blocks-10000-runes-v1:o200k_base", text: "hello", want: 1},
		{model: "gpt-5.1", wantID: "openai.tiktoken-go/v0.8.1:blocks-10000-runes-v1:o200k_base", text: "hello", want: 1},
		{model: "gpt-5.1-codex", wantID: "openai.tiktoken-go/v0.8.1:blocks-10000-runes-v1:o200k_base", text: "hello", want: 1},
		{model: "o1", wantID: "openai.tiktoken-go/v0.8.1:blocks-10000-runes-v1:o200k_base", text: "hello", want: 1},
		{model: GPT41, wantID: "openai.tiktoken-go/v0.8.1:blocks-10000-runes-v1:o200k_base", text: "hello", want: 1},
		{model: GPT4oMini, wantID: "openai.tiktoken-go/v0.8.1:blocks-10000-runes-v1:o200k_base", text: "hello", want: 1},
		{model: "gpt-4-0125-preview", wantID: "openai.tiktoken-go/v0.8.1:blocks-10000-runes-v1:cl100k_base", text: `{"tool":"weather","city":"München"}`, want: 10},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			tokenizer, err := NewTokenizer(tt.model)
			if err != nil {
				t.Fatalf("NewTokenizer(%q) error: %v", tt.model, err)
			}
			if tokenizer.ID() != tt.wantID {
				t.Fatalf("ID() = %q, want %q", tokenizer.ID(), tt.wantID)
			}
			got, err := tokenizer.CountTokens(context.Background(), tt.text)
			if err != nil || got != tt.want {
				t.Fatalf("CountTokens(%q) = %d, %v; want %d, nil", tt.text, got, err, tt.want)
			}
			tokens, err := tokenizer.Tokenize(context.Background(), tt.text)
			if err != nil {
				t.Fatalf("Tokenize(%q) error: %v", tt.text, err)
			}
			if len(tokens) != got {
				t.Fatalf("len(Tokenize(%q)) = %d, want CountTokens result %d", tt.text, len(tokens), got)
			}
		})
	}
}

func TestNewTokenizerRejectsUnknownModelWithTypedUnavailability(t *testing.T) {
	_, err := NewTokenizer("gpt-future-999")
	if err == nil {
		t.Fatal("NewTokenizer() error = nil, want unavailable error")
	}
	var unavailable *TokenizerUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("NewTokenizer() error = %T %v, want *TokenizerUnavailableError", err, err)
	}
	if unavailable.Model != "gpt-future-999" {
		t.Fatalf("unavailable model = %q, want %q", unavailable.Model, "gpt-future-999")
	}
	if !errors.Is(err, ai.ErrTokenizerUnsupported) {
		t.Fatalf("NewTokenizer() error = %v, want ai.ErrTokenizerUnsupported", err)
	}
}

func TestModelTokenizerUsesExplicitResolverAndPreservesUnavailableFallback(t *testing.T) {
	if tokenizer := (&Model{name: GPT41}).Tokenizer(); tokenizer == nil {
		t.Fatal("Tokenizer() = nil for a supported model")
	}
	if tokenizer := (&Model{name: "gpt-future-999"}).Tokenizer(); tokenizer != nil {
		t.Fatalf("Tokenizer() = %T for an unsupported model, want nil", tokenizer)
	}
}

func TestTokenizerHonorsCanceledContextWithoutProviderIO(t *testing.T) {
	tokenizer, err := NewTokenizer(GPT41)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := tokenizer.CountTokens(ctx, "hello"); !errors.Is(err, context.Canceled) {
		t.Fatalf("CountTokens() error = %v, want context.Canceled", err)
	}
	if _, err := tokenizer.Tokenize(ctx, "hello"); !errors.Is(err, context.Canceled) {
		t.Fatalf("Tokenize() error = %v, want context.Canceled", err)
	}
}

var realisticToolSchemaJSONFixtures = []struct {
	name string
	json string
	want int
}{
	{
		name: "web search with nested filters",
		json: `{"name":"web_search","description":"Search the public web for current information and return source URLs.","parameters":{"type":"object","additionalProperties":false,"properties":{"query":{"type":"string","description":"Natural-language search query"},"recency_days":{"type":"integer","minimum":1,"maximum":30},"domains":{"type":"array","items":{"type":"string","format":"hostname"}}},"required":["query"]}}`,
		want: 85,
	},
	{
		name: "calendar event with attendees",
		json: `{"name":"create_calendar_event","description":"Create a calendar event after the user confirms the proposed time.","parameters":{"type":"object","additionalProperties":false,"properties":{"title":{"type":"string"},"starts_at":{"type":"string","format":"date-time"},"duration_minutes":{"type":"integer","minimum":15,"maximum":480},"attendees":{"type":"array","items":{"type":"object","additionalProperties":false,"properties":{"email":{"type":"string","format":"email"},"optional":{"type":"boolean"}},"required":["email"]}},"send_updates":{"type":"boolean","default":true}},"required":["title","starts_at"]}}`,
		want: 133,
	},
	{
		name: "customer lookup with message context",
		json: `{"messages":[{"role":"system","content":"Use this tool only for authorized support requests."},{"role":"user","content":"Find the subscription for ada@example.com and include the latest invoice status."}],"tool":{"name":"lookup_customer","description":"Retrieve a customer profile and recent invoices by a verified identifier.","parameters":{"type":"object","additionalProperties":false,"properties":{"email":{"type":"string","format":"email"},"include_invoices":{"type":"boolean","default":false},"invoice_limit":{"type":"integer","minimum":1,"maximum":10}},"required":["email"]}}}`,
		want: 121,
	},
}

func TestTokenizerCountsRealisticToolSchemaJSONFixtures(t *testing.T) {
	tokenizer, err := NewTokenizer(GPT41)
	if err != nil {
		t.Fatal(err)
	}
	for _, fixture := range realisticToolSchemaJSONFixtures {
		t.Run(fixture.name, func(t *testing.T) {
			if !json.Valid([]byte(fixture.json)) {
				t.Fatal("fixture is not valid JSON")
			}
			first, err := tokenizer.Tokenize(t.Context(), fixture.json)
			if err != nil {
				t.Fatal(err)
			}
			second, err := tokenizer.Tokenize(t.Context(), fixture.json)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(first, second) {
				t.Fatalf("Tokenize() is not deterministic: first=%q second=%q", first, second)
			}
			count, err := tokenizer.CountTokens(t.Context(), fixture.json)
			if err != nil || count != fixture.want {
				t.Fatalf("CountTokens() = %d, %v; want %d, nil", count, err, fixture.want)
			}
			if len(first) != count {
				t.Fatalf("len(Tokenize()) = %d, want CountTokens result %d", len(first), count)
			}
		})
	}
}

func TestAutomaticCounterMatchesLocalEncodingAndPreservesFailures(t *testing.T) {
	counter := (&Model{name: GPT41}).TokenCounter()
	if counter.Fidelity() != ai.TokenCountEstimated {
		t.Fatalf("fidelity = %v", counter.Fidelity())
	}
	for _, fixture := range realisticToolSchemaJSONFixtures {
		count, err := counter.CountTokens(t.Context(), fixture.json)
		if err != nil || count != fixture.want {
			t.Fatalf("counter %s = %d, %v; want %d", fixture.name, count, err, fixture.want)
		}
	}
	if got := (&Model{name: "unknown-model"}).TokenCounter(); got != nil {
		t.Fatalf("unknown model counter = %T", got)
	}
	broken := &modelTokenCounter{encoding: "unknown-encoding"}
	if _, err := broken.CountTokens(t.Context(), "text"); err == nil {
		t.Fatal("invalid encoding was silently estimated")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := counter.CountTokens(ctx, "text"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled count = %v", err)
	}
}

// These reference encodings independently catch the fallback's former Unicode
// undercount. The estimator remains a heuristic, not a bound for all inputs.
func TestGenericEstimatorCoversMultilingualReferenceCounts(t *testing.T) {
	fixtures := []string{
		"这是一个用于测试多语言令牌计数的中文句子。",
		"これは多言語のトークン数を確認するための日本語の文章です。",
		"다국어 토큰 수를 확인하기 위한 한국어 문장입니다.",
		"🙂🚀👨‍👩‍👧‍👦🎉",
		"Please summarize 中文内容、日本語、한국어 and 🙂.",
	}
	for _, encoding := range []tiktoken.Encoding{tiktoken.Cl100kBase, tiktoken.O200kBase} {
		reference, err := tiktoken.Get(encoding)
		if err != nil {
			t.Fatal(err)
		}
		for _, text := range fixtures {
			exact, err := reference.Count(text)
			if err != nil {
				t.Fatal(err)
			}
			estimate, err := (ai.TextTokenEstimator{}).CountTokens(t.Context(), text)
			if err != nil {
				t.Fatal(err)
			}
			if estimate < exact {
				t.Fatalf("%s: estimate %d below reference %d for %q", encoding, estimate, exact, text)
			}
		}
	}
}
