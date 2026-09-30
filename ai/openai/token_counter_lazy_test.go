package openai

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	tiktoken "github.com/tiktoken-go/tokenizer"
)

type observedCountCodec struct {
	tiktoken.Codec
	counts  atomic.Int64
	onCount func(int64)
	onPiece func(string)
}

func (c *observedCountCodec) Count(text string) (int, error) {
	count := c.counts.Add(1)
	if c.onCount != nil {
		c.onCount(count)
	}
	if c.onPiece != nil {
		c.onPiece(text)
	}
	return c.Codec.Count(text)
}

func TestLazyCounterEmptySkipsSharedCache(t *testing.T) {
	for _, encoding := range counterEncodings {
		t.Run(encoding.name, func(t *testing.T) {
			tokenizer, err := NewTokenizer(encoding.model)
			if err != nil {
				t.Fatal(err)
			}
			counter := (&Model{name: encoding.model}).TokenCounter()
			cache := &cl100kCounter
			if encoding.encoding == tiktoken.O200kBase {
				cache = &o200kCounter
			}
			// No cache resets: holding its lock makes a needless lookup fail by
			// deadline whether the shared codec is already initialized or not.
			if err := cache.mu.Lock(t.Context()); err != nil {
				t.Fatal(err)
			}
			defer cache.mu.Unlock()
			for name, count := range map[string]func(context.Context, string) (int, error){
				"tokenizer": tokenizer.CountTokens,
				"model":     counter.CountTokens,
			} {
				t.Run(name, func(t *testing.T) {
					ctx, cancel := context.WithTimeout(t.Context(), time.Second)
					defer cancel()
					got, err := count(ctx, "")
					if got != 0 || err != nil {
						t.Fatalf("empty count waited for codec initialization: %d, %v", got, err)
					}
					cancel()
					if _, err := count(ctx, ""); !errors.Is(err, context.Canceled) {
						t.Fatalf("empty count ignored cancellation: %v", err)
					}
				})
			}
		})
	}
}

func TestLazyCounterReusesSourceWithoutRanksForOrdinaryDocuments(t *testing.T) {
	for _, encoding := range counterEncodings {
		t.Run(encoding.name, func(t *testing.T) {
			tokenizer, err := NewTokenizer(encoding.model)
			if err != nil {
				t.Fatal(err)
			}
			existing := tokenizer.(*Tokenizer).codec
			source := &observedCountCodec{Codec: existing}
			var cache countCodecCache
			codec, err := cache.load(t.Context(), encoding.encoding, source)
			if err != nil {
				t.Fatal(err)
			}
			if codec.source != source || codec.ranks.ranks != nil {
				t.Fatal("codec did not reuse its supplied source without reconstructing ranks")
			}
			for name, text := range map[string]string{
				"short": "Hello world!",
				"prose": strings.Repeat("I'm comparing 中文 and 日本語, with small words.\n", 512),
				"json":  strings.Repeat(realisticToolSchemaJSONFixtures[0].json+"\n", 32),
			} {
				want, err := existing.Count(text)
				if err != nil {
					t.Fatal(err)
				}
				before := source.counts.Load()
				got, err := codec.count(t.Context(), text)
				if err != nil || got != want {
					t.Fatalf("%s count = %d, %v; upstream = %d", name, got, err, want)
				}
				if codec.ranks.ranks != nil || source.counts.Load() <= before {
					t.Fatalf("%s reconstructed ranks or bypassed the existing source", name)
				}
			}
		})
	}
}

func TestLazyCounterThresholdPreservesExactPieceCounts(t *testing.T) {
	// Invalid input is normalized by regexp Match.String. The threshold must
	// use that matched UTF-8 representation, not the original raw byte count.
	families := []struct {
		name  string
		piece func(int) string
	}{
		{"ascii", func(bytes int) string { return strings.Repeat("a", bytes) }},
		{"unicode", func(bytes int) string { return strings.Repeat("中", bytes/3) + strings.Repeat("a", bytes%3) }},
		{"emoji", func(bytes int) string { return strings.Repeat("🙂", bytes/4) + strings.Repeat("!", bytes%4) }},
		{"invalid_utf8", func(bytes int) string { return strings.Repeat("\xff", bytes/3) + strings.Repeat(":", bytes%3) }},
	}
	for _, encoding := range counterEncodings {
		t.Run(encoding.name, func(t *testing.T) {
			reference, err := tiktoken.Get(encoding.encoding)
			if err != nil {
				t.Fatal(err)
			}
			for _, family := range families {
				t.Run(family.name, func(t *testing.T) {
					codec, err := newCountCodec(t.Context(), encoding.encoding, nil)
					if err != nil {
						t.Fatal(err)
					}
					for _, size := range []int{countDirectLimit - 1, countDirectLimit, countDirectLimit + 1} {
						text := family.piece(size)
						match, err := codec.split.FindStringMatch(text)
						if err != nil || match == nil || len(match.String()) != size {
							t.Fatalf("fixture is not a %d-byte matched piece: %v", size, err)
						}
						if next, err := codec.split.FindNextMatch(match); err != nil || next != nil {
							t.Fatalf("fixture unexpectedly contains multiple regex pieces: %v", err)
						}
						// Force piece-level dispatch even when invalid raw bytes are
						// shorter than their normalized matched representation.
						input := strings.Repeat("x\n", countDirectLimit) + text
						want, err := reference.Count(input)
						if err != nil {
							t.Fatal(err)
						}
						got, err := codec.count(t.Context(), input)
						if err != nil || got != want {
							t.Fatalf("%d-byte piece = %d, %v; upstream = %d", size, got, err, want)
						}
						if initialized := codec.ranks.ranks != nil; initialized != (size > countDirectLimit) {
							t.Fatalf("%d-byte piece initialized ranks = %v", size, initialized)
						}
					}
				})
			}
		})
	}
}

func TestLazyCounterCancelsBetweenSmallPiecesWithoutRanks(t *testing.T) {
	for _, encoding := range counterEncodings {
		t.Run(encoding.name, func(t *testing.T) {
			reference, err := tiktoken.Get(encoding.encoding)
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			source := &observedCountCodec{Codec: reference, onCount: func(calls int64) {
				if calls == 10 {
					cancel()
				}
			}}
			codec, err := newCountCodec(t.Context(), encoding.encoding, source)
			if err != nil {
				t.Fatal(err)
			}
			got, err := codec.count(ctx, strings.Join(lazyCounterDistinctWords(128), "\n"))
			if got != 0 || !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation between pieces = %d, %v", got, err)
			}
			if source.counts.Load() != 10 || codec.ranks.ranks != nil {
				t.Fatalf("counting continued or built ranks after cancellation: calls=%d", source.counts.Load())
			}
		})
	}
}

func lazyCounterDistinctWords(count int) []string {
	words := make([]string, count)
	for i := range words {
		// Letters stay together as one regex piece in both encodings.
		words[i] = string([]rune{'a' + rune(i/676), 'a' + rune(i/26%26), 'a' + rune(i%26)})
	}
	return words
}

func TestLazyCounterMemoReusesWholePiecesOnlyWithinOneCall(t *testing.T) {
	for _, encoding := range counterEncodings {
		t.Run(encoding.name, func(t *testing.T) {
			reference, err := tiktoken.Get(encoding.encoding)
			if err != nil {
				t.Fatal(err)
			}
			text := strings.Repeat(" hello world", 512)
			want, err := reference.Count(text)
			if err != nil {
				t.Fatal(err)
			}
			source := &observedCountCodec{Codec: reference}
			codec, err := newCountCodec(t.Context(), encoding.encoding, source)
			if err != nil {
				t.Fatal(err)
			}
			var firstCalls int64
			for run := range 2 {
				before := source.counts.Load()
				got, err := codec.count(t.Context(), text)
				calls := source.counts.Load() - before
				if err != nil || got != want {
					t.Fatalf("run %d count = %d, %v; upstream = %d", run, got, err, want)
				}
				if calls <= 0 || calls >= 512 {
					t.Fatalf("run %d upstream calls = %d; want reused pieces without shared retention", run, calls)
				}
				if run == 0 {
					firstCalls = calls
				} else if calls != firstCalls {
					t.Fatalf("second call retained the first call's memo: upstream calls %d then %d", firstCalls, calls)
				}
			}
			if codec.ranks.ranks != nil {
				t.Fatal("repeated short pieces initialized ranks")
			}
		})
	}
}

func TestLazyCounterMemoStopsRetainingNewPiecesAtLimit(t *testing.T) {
	for _, encoding := range counterEncodings {
		t.Run(encoding.name, func(t *testing.T) {
			reference, err := tiktoken.Get(encoding.encoding)
			if err != nil {
				t.Fatal(err)
			}
			words := lazyCounterDistinctWords(countMemoLimit + 2)
			first, last := words[0], words[len(words)-1]
			text := strings.Join(words, "\n") + "\n" + strings.Repeat(first+"\n", 5) + strings.Repeat(last+"\n", 5)
			want, err := reference.Count(text)
			if err != nil {
				t.Fatal(err)
			}
			pieces := make(map[string]int)
			source := &observedCountCodec{Codec: reference, onPiece: func(piece string) { pieces[piece]++ }}
			codec, err := newCountCodec(t.Context(), encoding.encoding, source)
			if err != nil {
				t.Fatal(err)
			}
			got, err := codec.count(t.Context(), text)
			if err != nil || got != want {
				t.Fatalf("bounded memo count = %d, %v; upstream = %d", got, err, want)
			}
			if pieces[first] != 1 || pieces[last] != 6 {
				t.Fatalf("upstream calls for early/overflow pieces = %d/%d; want 1/6", pieces[first], pieces[last])
			}
			if codec.ranks.ranks != nil {
				t.Fatal("many distinct short pieces initialized ranks")
			}
		})
	}
}

func TestLazyCounterMemoHitsHonorCancellation(t *testing.T) {
	for _, encoding := range counterEncodings {
		t.Run(encoding.name, func(t *testing.T) {
			reference, err := tiktoken.Get(encoding.encoding)
			if err != nil {
				t.Fatal(err)
			}
			source := &observedCountCodec{Codec: reference}
			codec, err := newCountCodec(t.Context(), encoding.encoding, source)
			if err != nil {
				t.Fatal(err)
			}
			ctx := newCancelAfterCounterChecks(t, 20)
			got, err := codec.count(ctx, strings.Repeat(" hello", 1000))
			if got != 0 || !errors.Is(err, context.Canceled) {
				t.Fatalf("memo-hit cancellation = %d, %v", got, err)
			}
			if calls := source.counts.Load(); calls != 1 {
				t.Fatalf("expected cancellation while reusing the first piece; upstream calls = %d", calls)
			}
			if codec.ranks.ranks != nil {
				t.Fatal("memo-hit cancellation initialized ranks")
			}
		})
	}
}

func TestLazyCounterSharesRanksForConcurrentLongPieces(t *testing.T) {
	for _, encoding := range counterEncodings {
		t.Run(encoding.name, func(t *testing.T) {
			codec, err := newCountCodec(t.Context(), encoding.encoding, nil)
			if err != nil {
				t.Fatal(err)
			}
			text := strings.Repeat("a", countDirectLimit+1)
			want, err := codec.source.Count(text)
			if err != nil {
				t.Fatal(err)
			}
			if codec.ranks.ranks != nil {
				t.Fatal("fresh codec initialized ranks before long-piece counting")
			}
			start := make(chan struct{})
			var workers sync.WaitGroup
			var rankMaps [8]uintptr
			for worker := range len(rankMaps) {
				workers.Add(1)
				go func() {
					defer workers.Done()
					<-start
					got, err := codec.count(t.Context(), text)
					if err != nil || got != want {
						t.Errorf("worker %d count = %d, %v; want %d", worker, got, err, want)
						return
					}
					ranks, err := codec.ranks.load(t.Context(), encoding.encoding)
					if err != nil {
						t.Errorf("worker %d ranks: %v", worker, err)
						return
					}
					rankMaps[worker] = reflect.ValueOf(ranks).Pointer()
				}()
			}
			close(start)
			workers.Wait()
			for worker, pointer := range rankMaps {
				if pointer == 0 || pointer != rankMaps[0] {
					t.Errorf("worker %d received a different or missing rank map", worker)
				}
			}
			if got, err := codec.count(t.Context(), text); err != nil || got != want {
				t.Fatalf("repeat count = %d, %v; want %d", got, err, want)
			}
			if reflect.ValueOf(codec.ranks.ranks).Pointer() != rankMaps[0] {
				t.Fatal("repeat long-piece count reconstructed the rank map")
			}
		})
	}
}
