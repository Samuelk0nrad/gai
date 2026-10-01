package openai

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/lace-ai/gai/ai"
	tiktoken "github.com/tiktoken-go/tokenizer"
)

// Only Count is implemented: block counting must not decode or re-encode text.
type recordingBlockCodec struct {
	tiktoken.Codec
	blocks []string
	count  func(call int, text string) (int, error)
}

func (c *recordingBlockCodec) Count(text string) (int, error) {
	c.blocks = append(c.blocks, text)
	if c.count != nil {
		return c.count(len(c.blocks), text)
	}
	return utf8.RuneCountInString(text), nil
}

func TestCountTokenBlocksPreservesUnicodeBoundariesAndBytes(t *testing.T) {
	for _, tc := range []struct {
		name string
		text string
	}{
		{"empty", ""},
		{"short", "hello 世界🙂"},
		{"one exact block", strings.Repeat("a", tokenCountBlockRunes)},
		{"two exact blocks", strings.Repeat("中", 2*tokenCountBlockRunes)},
		{"unicode boundary", strings.Repeat("a", tokenCountBlockRunes-1) + "🙂中"},
		{"mixed unicode", strings.Repeat("🙂中é", tokenCountBlockRunes)},
		{"invalid UTF8", strings.Repeat("a", tokenCountBlockRunes-1) + "🙂\xff\xfe中"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			codec := &recordingBlockCodec{}
			got, err := countTokenBlocks(t.Context(), codec, tc.text)
			want := utf8.RuneCountInString(tc.text)
			if err != nil || got != want {
				t.Fatalf("count = %d, %v; fake codec total = %d", got, err, want)
			}
			if len(codec.blocks) != (want+tokenCountBlockRunes-1)/tokenCountBlockRunes {
				t.Fatalf("unexpected blocks: %d for %d code points", len(codec.blocks), want)
			}
			for i, block := range codec.blocks {
				if size := utf8.RuneCountInString(block); size == 0 || size > tokenCountBlockRunes {
					t.Fatalf("block %d has %d code points", i, size)
				}
				if utf8.ValidString(tc.text) && !utf8.ValidString(block) {
					t.Fatalf("block %d split a UTF-8 sequence", i)
				}
			}
			if strings.Join(codec.blocks, "") != tc.text {
				t.Fatal("block splitting changed the original bytes")
			}
		})
	}
}

func TestCountTokenBlocksHonorsCancellationBeforeAndAfterCount(t *testing.T) {
	for _, tc := range []struct {
		name   string
		text   string
		cancel int
	}{
		{"before counting", "hello", 0},
		{"before empty input", "", 0},
		{"after first block", strings.Repeat("a", 2*tokenCountBlockRunes+1), 1},
		{"after final block", strings.Repeat("a", tokenCountBlockRunes+1), 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			if tc.cancel == 0 {
				cancel()
			}
			codec := &recordingBlockCodec{count: func(call int, _ string) (int, error) {
				if call == tc.cancel {
					cancel()
				}
				// The current upstream call completes even when it cancels ctx.
				return 7, nil
			}}
			got, err := countTokenBlocks(ctx, codec, tc.text)
			if got != 0 || !errors.Is(err, context.Canceled) || len(codec.blocks) != tc.cancel {
				t.Fatalf("count = %d, %v, calls = %d; want cancellation after %d calls", got, err, len(codec.blocks), tc.cancel)
			}
		})
	}
}

func TestCountTokenBlocksReturnsSourceFailureWithoutPartialCount(t *testing.T) {
	failure := errors.New("upstream count failed")
	for _, failAt := range []int{1, 2} {
		codec := &recordingBlockCodec{count: func(call int, _ string) (int, error) {
			if call == failAt {
				return 0, failure
			}
			return 7, nil
		}}
		got, err := countTokenBlocks(t.Context(), codec, strings.Repeat("a", 2*tokenCountBlockRunes+1))
		if got != 0 || !errors.Is(err, failure) || len(codec.blocks) != failAt {
			t.Fatalf("failure at %d: count = %d, %v, calls = %d", failAt, got, err, len(codec.blocks))
		}
	}
}

func TestCountTokenBlocksStopsAfterCurrentCallOnDeadline(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()
	codec := &recordingBlockCodec{count: func(int, string) (int, error) {
		<-ctx.Done()
		return 7, nil
	}}
	got, err := countTokenBlocks(ctx, codec, strings.Repeat("a", tokenCountBlockRunes+1))
	if got != 0 || !errors.Is(err, context.DeadlineExceeded) || len(codec.blocks) != 1 {
		t.Fatalf("count = %d, %v, calls = %d; want deadline after current block", got, err, len(codec.blocks))
	}
}

func TestPublicBlockCountsMatchIndependentUpstreamChunks(t *testing.T) {
	first := strings.Repeat("a", tokenCountBlockRunes-2) + "he"
	text := first + "llo"
	for _, tc := range []struct {
		name     string
		model    string
		encoding tiktoken.Encoding
	}{
		{"cl100k", "gpt-4", tiktoken.Cl100kBase},
		{"o200k", GPT41, tiktoken.O200kBase},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reference, err := tiktoken.Get(tc.encoding)
			if err != nil {
				t.Fatal(err)
			}
			want := 0
			for _, block := range []string{first, "llo"} {
				count, err := reference.Count(block)
				if err != nil {
					t.Fatal(err)
				}
				want += count
			}
			whole, err := reference.Count(text)
			if err != nil || whole == want {
				t.Fatalf("fixture must distinguish whole-text and block counts: %d/%d, %v", whole, want, err)
			}
			tokenizer, err := NewTokenizer(tc.model)
			if err != nil {
				t.Fatal(err)
			}
			counter := (&Model{name: tc.model}).TokenCounter()
			explicit, ok := tokenizer.(ai.TokenCounter)
			if !ok {
				t.Fatal("explicit tokenizer does not expose counter fidelity")
			}
			if explicit.Fidelity() != ai.TokenCountEstimated || counter.Fidelity() != ai.TokenCountEstimated || explicit.ID() != counter.ID() || counter.ID() == "openai.tiktoken-go/v0.8.1:"+string(tc.encoding) {
				t.Fatalf("block counters need matching estimated identities distinct from whole-text counts: %q/%q", explicit.ID(), counter.ID())
			}
			for name, count := range map[string]func(context.Context, string) (int, error){
				"tokenizer": tokenizer.CountTokens,
				"model":     counter.CountTokens,
			} {
				got, err := count(t.Context(), text)
				if err != nil || got != want {
					t.Fatalf("%s count = %d, %v; independent block total = %d", name, got, err, want)
				}
			}
		})
	}
}

func TestAutomaticBlockCounterSharesUpstreamCodecConcurrently(t *testing.T) {
	for _, tc := range []struct {
		model    string
		encoding tiktoken.Encoding
	}{{"gpt-4", tiktoken.Cl100kBase}, {GPT41, tiktoken.O200kBase}} {
		t.Run(tc.model, func(t *testing.T) {
			reference, err := tiktoken.Get(tc.encoding)
			if err != nil {
				t.Fatal(err)
			}
			texts := []string{"hello world", "中文 日本語 🙂", "{\"answer\":42}\n"}
			wants := make([]int, len(texts))
			for i, text := range texts {
				wants[i], err = reference.Count(text)
				if err != nil {
					t.Fatal(err)
				}
			}
			start := make(chan struct{})
			var workers sync.WaitGroup
			for range 4 {
				workers.Add(1)
				go func() {
					defer workers.Done()
					counter := (&Model{name: tc.model}).TokenCounter()
					<-start
					for range 3 {
						for i, text := range texts {
							if got, err := counter.CountTokens(t.Context(), text); err != nil || got != wants[i] {
								t.Errorf("concurrent count = %d, %v; want %d", got, err, wants[i])
								return
							}
						}
					}
				}()
			}
			close(start)
			workers.Wait()
		})
	}
}
