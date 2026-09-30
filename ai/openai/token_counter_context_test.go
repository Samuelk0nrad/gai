package openai

import (
	"context"
	"errors"
	"math/rand"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	tiktoken "github.com/tiktoken-go/tokenizer"
)

var counterEncodings = []struct {
	name     string
	model    string
	encoding tiktoken.Encoding
}{
	{"cl100k", "gpt-4", tiktoken.Cl100kBase},
	{"o200k", GPT41, tiktoken.O200kBase},
}

// Compare with the dependency's original implementation, rather than with the
// context-aware adapter through another public entry point.
func TestContextCounterMatchesPinnedCodec(t *testing.T) {
	fixtures := []string{
		"", "hello", "HelloWorld HTTPServer XMLParser fooBAR Baz",
		"I'm I'M we're WE'RE they've She'll it'S can't isn't shouldn't",
		"01234567890123456789 123.456 999,000 -42 1e100",
		" \t\t\n\r\n  trailing spaces   \n\n\t \r next",
		"Grüße über Österreich — e\u0301 versus é; 中文 日本語 한국어 العربية",
		"🙂👩🏽‍💻👨‍👩‍👧‍👦🇦🇹\u200d\ufe0f",
		"<|endoftext|> <|fim_prefix|> <|im_start|>",
		"nul\x00control\x01\x7f byte\xff\xfe truncated\xe2\x82",
		strings.Repeat("a", 4097),
		strings.Repeat("abcdefghij", 211),
		strings.Repeat("中", 301),
	}
	random := rand.New(rand.NewSource(138))
	alphabet := []rune("aAZz'019-_/ .\t\n\r中界あ한éö\u0301🙂\u200d")
	for range 80 {
		text := make([]rune, random.Intn(256)+1)
		for i := range text {
			text[i] = alphabet[random.Intn(len(alphabet))]
		}
		fixtures = append(fixtures, string(text))
	}
	for range 20 {
		text := make([]byte, random.Intn(128)+1)
		if _, err := random.Read(text); err != nil {
			t.Fatal(err)
		}
		fixtures = append(fixtures, string(text))
	}
	for _, encoding := range counterEncodings {
		t.Run(encoding.name, func(t *testing.T) {
			reference, err := tiktoken.Get(encoding.encoding)
			if err != nil {
				t.Fatal(err)
			}
			tokenizer, err := NewTokenizer(encoding.model)
			if err != nil {
				t.Fatal(err)
			}
			counter := (&Model{name: encoding.model}).TokenCounter()
			for index, text := range fixtures {
				want, err := reference.Count(text)
				if err != nil {
					t.Fatalf("fixture %d reference Count: %v", index, err)
				}
				ids, _, err := reference.Encode(text)
				if err != nil || len(ids) != want {
					t.Fatalf("fixture %d reference Encode = %d, %v; Count = %d", index, len(ids), err, want)
				}
				for name, count := range map[string]func(context.Context, string) (int, error){
					"tokenizer": tokenizer.CountTokens,
					"model":     counter.CountTokens,
				} {
					got, err := count(t.Context(), text)
					if err != nil || got != want {
						t.Fatalf("%s fixture %d (%q) = %d, %v; pinned Count/Encode = %d", name, index, text, got, err, want)
					}
				}
			}
		})
	}
}

func TestContextCounterPreservesWholeTextMergeBoundaries(t *testing.T) {
	for _, encoding := range counterEncodings {
		t.Run(encoding.name, func(t *testing.T) {
			reference, err := tiktoken.Get(encoding.encoding)
			if err != nil {
				t.Fatal(err)
			}
			whole, err := reference.Count("hello")
			if err != nil {
				t.Fatal(err)
			}
			split := 0
			for _, text := range []string{"he", "llo"} {
				count, err := reference.Count(text)
				if err != nil {
					t.Fatal(err)
				}
				split += count
			}
			if whole == split {
				t.Fatal("fixture does not distinguish whole-text and arbitrary chunk counting")
			}
			got, err := (&Model{name: encoding.model}).TokenCounter().CountTokens(t.Context(), "hello")
			if err != nil || got != whole {
				t.Fatalf("whole count = %d, %v; want %d, not chunk total %d", got, err, whole, split)
			}
		})
	}
}

// This context begins live and cancels only after work polls it repeatedly.
// Unlike a pre-canceled context, it reaches the long-piece counting loops.
type cancelAfterCounterChecks struct {
	context.Context
	cancel context.CancelFunc
	after  int64
	calls  atomic.Int64
}

func newCancelAfterCounterChecks(t *testing.T, after int64) *cancelAfterCounterChecks {
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	return &cancelAfterCounterChecks{Context: ctx, cancel: cancel, after: after}
}

func (c *cancelAfterCounterChecks) Err() error {
	if c.calls.Add(1) >= c.after {
		c.cancel()
	}
	return c.Context.Err()
}

func TestPublicCountersCancelDuringSingleLongPiece(t *testing.T) {
	text := strings.Repeat("a", 64*1024)
	for _, encoding := range counterEncodings {
		t.Run(encoding.name, func(t *testing.T) {
			tokenizer, err := NewTokenizer(encoding.model)
			if err != nil {
				t.Fatal(err)
			}
			counter := (&Model{name: encoding.model}).TokenCounter()
			for name, count := range map[string]func(context.Context, string) (int, error){
				"tokenizer": tokenizer.CountTokens,
				"model":     counter.CountTokens,
			} {
				t.Run(name, func(t *testing.T) {
					// Warm the cache so cancellation must interrupt actual counting.
					if _, err := count(t.Context(), "warmup"); err != nil {
						t.Fatal(err)
					}
					ctx := newCancelAfterCounterChecks(t, 200)
					started := time.Now()
					got, err := count(ctx, text)
					if !errors.Is(err, context.Canceled) || got != 0 {
						t.Fatalf("mid-count cancellation = %d, %v; checks=%d", got, err, ctx.calls.Load())
					}
					if elapsed := time.Since(started); elapsed > 2*time.Second {
						t.Fatalf("cancellation took %s", elapsed)
					}
					checks := ctx.calls.Load()
					time.Sleep(10 * time.Millisecond)
					if after := ctx.calls.Load(); after != checks {
						t.Fatalf("counting continued polling after return: %d -> %d", checks, after)
					}
					deadline, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
					defer cancel()
					started = time.Now()
					got, err = count(deadline, text)
					if !errors.Is(err, context.DeadlineExceeded) || got != 0 {
						t.Fatalf("mid-count deadline = %d, %v", got, err)
					}
					if elapsed := time.Since(started); elapsed > 2*time.Second {
						t.Fatalf("deadline response took %s", elapsed)
					}
				})
			}
		})
	}
}

func TestCountPieceCancellationInsideMergeScan(t *testing.T) {
	codec, err := localCountCodec(t.Context(), tiktoken.Cl100kBase)
	if err != nil {
		t.Fatal(err)
	}
	// For 8192 bytes: entry + 9 initialization polls + 8 rank polls +
	// first merge entry consume 19 checks; check 22 is inside its minimum scan.
	ctx := newCancelAfterCounterChecks(t, 22)
	got, err := codec.countPiece(ctx, strings.Repeat("a", 8192))
	if got != 0 || !errors.Is(err, context.Canceled) || ctx.calls.Load() != 22 {
		t.Fatalf("merge-scan cancellation = %d, %v; checks=%d", got, err, ctx.calls.Load())
	}
}

func TestCountCodecCacheRetriesCanceledInitialization(t *testing.T) {
	for _, encoding := range counterEncodings {
		t.Run(encoding.name, func(t *testing.T) {
			var cache countCodecCache
			ctx := newCancelAfterCounterChecks(t, 5)
			codec, err := cache.load(ctx, encoding.encoding)
			if codec != nil || !errors.Is(err, context.Canceled) || cache.codec != nil {
				t.Fatalf("canceled initialization = %p, %v; published codec=%p", codec, err, cache.codec)
			}
			codec, err = cache.load(t.Context(), encoding.encoding)
			if err != nil || codec == nil {
				t.Fatalf("retry initialization = %p, %v", codec, err)
			}
			got, err := codec.count(t.Context(), "hello")
			if err != nil || got != 1 {
				t.Fatalf("retried codec count = %d, %v", got, err)
			}
			reused, err := cache.load(t.Context(), encoding.encoding)
			if err != nil || reused != codec {
				t.Fatalf("cache reuse = %p, %v; want %p", reused, err, codec)
			}
		})
	}
}

func TestCountCodecCacheWaitHonorsDeadline(t *testing.T) {
	codec, err := localCountCodec(t.Context(), tiktoken.Cl100kBase)
	if err != nil {
		t.Fatal(err)
	}
	cache := countCodecCache{codec: codec}
	if err := cache.mu.Lock(t.Context()); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()
	started := time.Now()
	got, err := cache.load(ctx, tiktoken.Cl100kBase)
	cache.mu.Unlock()
	if got != nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("blocked cache = %p, %v", got, err)
	}
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("cache wait took %s", elapsed)
	}
	got, err = cache.load(t.Context(), tiktoken.Cl100kBase)
	if err != nil || got != codec {
		t.Fatalf("canceled waiter damaged cache: %p, %v", got, err)
	}
}

func TestPublicCountersShareCodecConcurrently(t *testing.T) {
	for _, encoding := range counterEncodings {
		t.Run(encoding.name, func(t *testing.T) {
			reference, err := tiktoken.Get(encoding.encoding)
			if err != nil {
				t.Fatal(err)
			}
			texts := []string{"I'm comparing 中文 and 日本語 🙂.\n", strings.Repeat("abcdef", 43), " 1234567\t ABCdef\r\n"}
			wants := make([]int, len(texts))
			for i, text := range texts {
				wants[i], err = reference.Count(text)
				if err != nil {
					t.Fatal(err)
				}
			}
			tokenizer, err := NewTokenizer(encoding.model)
			if err != nil {
				t.Fatal(err)
			}
			counter := (&Model{name: encoding.model}).TokenCounter()
			counts := []func(context.Context, string) (int, error){tokenizer.CountTokens, counter.CountTokens}
			start := make(chan struct{})
			var workers sync.WaitGroup
			for worker := range 8 {
				workers.Add(1)
				go func() {
					defer workers.Done()
					<-start
					for range 8 {
						for index, text := range texts {
							got, err := counts[worker%len(counts)](t.Context(), text)
							if err != nil || got != wants[index] {
								t.Errorf("worker %d fixture %d = %d, %v; want %d", worker, index, got, err, wants[index])
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

func TestCountCodecCachePublishesOneCompleteCodecConcurrently(t *testing.T) {
	for _, encoding := range counterEncodings {
		t.Run(encoding.name, func(t *testing.T) {
			var cache countCodecCache
			var codecs [8]*countCodec
			var loadErrors [8]error
			var counts [8]int
			start := make(chan struct{})
			var workers sync.WaitGroup
			for worker := range len(codecs) {
				workers.Add(1)
				go func() {
					defer workers.Done()
					<-start
					codecs[worker], loadErrors[worker] = cache.load(t.Context(), encoding.encoding)
					if loadErrors[worker] == nil {
						counts[worker], loadErrors[worker] = codecs[worker].count(t.Context(), "hello")
					}
				}()
			}
			close(start)
			workers.Wait()
			for worker, codec := range codecs {
				if loadErrors[worker] != nil || codec == nil || codec != codecs[0] || counts[worker] != 1 {
					t.Errorf("worker %d codec=%p count=%d error=%v; want shared complete codec %p", worker, codec, counts[worker], loadErrors[worker], codecs[0])
				}
			}
		})
	}
}
