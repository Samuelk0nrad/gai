package openai

import (
	"context"
	"fmt"
	"math"

	"github.com/dlclark/regexp2/v2"
	"github.com/lace-ai/gai/ai"
	tiktoken "github.com/tiktoken-go/tokenizer"
)

// The split patterns and pair-merging algorithm below follow tiktoken-go
// tokenizer v0.8.1, with cooperative cancellation added to its work loops.
// See TIKTOKEN_LICENSE. Keep boundaries, rank ordering and leftmost tie-breaking
// aligned with that version: counting arbitrary text chunks is not equivalent.
const (
	cl100kSplit       = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	o200kSplit        = `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	countPollInterval = 1024
	// Bound synchronous calls to the upstream codec. Longer text is split at
	// encoding boundaries; oversized pieces use cancellation-aware pair merging.
	countDirectLimit = 256
	countMemoLimit   = 256
)

type countCodec struct {
	encoding tiktoken.Encoding
	source   tiktoken.Codec
	split    *regexp2.Regexp
	ranks    countRanksCache
}

type countCodecCache struct {
	mu    ai.ContextMutex
	codec *countCodec
}

var cl100kCounter, o200kCounter countCodecCache

func localCountCodec(ctx context.Context, encoding tiktoken.Encoding, source tiktoken.Codec) (*countCodec, error) {
	var cache *countCodecCache
	switch encoding {
	case tiktoken.Cl100kBase:
		cache = &cl100kCounter
	case tiktoken.O200kBase:
		cache = &o200kCounter
	default:
		return nil, tiktoken.ErrEncodingNotSupported
	}
	return cache.load(ctx, encoding, source)
}

func (c *countCodecCache) load(ctx context.Context, encoding tiktoken.Encoding, source tiktoken.Codec) (*countCodec, error) {
	if err := c.mu.Lock(ctx); err != nil {
		return nil, err
	}
	defer c.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if c.codec != nil {
		return c.codec, nil
	}
	codec, err := newCountCodec(ctx, encoding, source)
	if err != nil {
		return nil, err
	}
	// Publish only a complete wrapper. A canceled initializer may be retried.
	c.codec = codec
	return codec, nil
}

func newCountCodec(ctx context.Context, encoding tiktoken.Encoding, source tiktoken.Codec) (*countCodec, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var pattern string
	switch encoding {
	case tiktoken.Cl100kBase:
		pattern = cl100kSplit
	case tiktoken.O200kBase:
		pattern = o200kSplit
	default:
		return nil, tiktoken.ErrEncodingNotSupported
	}
	if source == nil {
		var err error
		source, err = tiktoken.Get(encoding)
		if err != nil {
			return nil, err
		}
	}
	split := regexp2.MustCompile(pattern, regexp2.None)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &countCodec{encoding: encoding, source: source, split: split}, nil
}

type countRanksCache struct {
	mu    ai.ContextMutex
	ranks map[string]uint
}

func (c *countRanksCache) load(ctx context.Context, encoding tiktoken.Encoding) (map[string]uint, error) {
	if err := c.mu.Lock(ctx); err != nil {
		return nil, err
	}
	defer c.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if c.ranks != nil {
		return c.ranks, nil
	}
	ranks, err := newCountRanks(ctx, encoding)
	if err != nil {
		return nil, err
	}
	// Readers share this immutable map only after successful initialization.
	c.ranks = ranks
	return ranks, nil
}

func newCountRanks(ctx context.Context, encoding tiktoken.Encoding) (map[string]uint, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var size uint
	// These are the contiguous ordinary-token ranks in the pinned v0.8.1 data.
	// Special tokens are not recognized by that codec's Count/Encode methods.
	switch encoding {
	case tiktoken.Cl100kBase:
		size = 100256
	case tiktoken.O200kBase:
		size = 199998
	default:
		return nil, tiktoken.ErrEncodingNotSupported
	}
	source, err := tiktoken.Get(encoding)
	if err != nil {
		return nil, err
	}
	ranks := make(map[string]uint, size)
	// Decode is the dependency's public access to its embedded vocabulary. Use a
	// private codec here: its reverse vocabulary is initialized lazily. This avoids
	// unsafe access or duplicating the embedded multi-megabyte data tables.
	for rank := uint(0); rank < size; rank++ {
		if rank%countPollInterval == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		piece, err := source.Decode([]uint{rank})
		if err != nil {
			return nil, fmt.Errorf("read %s token rank %d: %w", encoding, rank, err)
		}
		ranks[piece] = rank
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return ranks, nil
}

func (c *countCodec) count(ctx context.Context, text string) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if len(text) <= countDirectLimit {
		count, err := c.source.Count(text)
		if canceled := ctx.Err(); canceled != nil {
			return 0, canceled
		}
		return count, err
	}
	match, err := c.split.FindStringMatch(text)
	if err != nil {
		return 0, err
	}
	// Repeated words and JSON keys need counting only once per call. Bound the
	// memo and discard it on return so user text never enters a shared cache.
	counts := make(map[string]int)
	count := 0
	for match != nil {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		piece := match.String()
		var n int
		if len(piece) <= countDirectLimit {
			var cached bool
			n, cached = counts[piece]
			if !cached {
				// Re-count only whole matches, never arbitrary chunks. For these
				// pinned patterns each match remains one match on its own.
				n, err = c.source.Count(piece)
				if err == nil && len(counts) < countMemoLimit {
					counts[piece] = n
				}
			}
		} else {
			n, err = c.countPiece(ctx, piece)
		}
		if err != nil {
			return 0, err
		}
		count += n
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		match, err = c.split.FindNextMatch(match)
		if err != nil {
			return 0, err
		}
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	return count, nil
}

type countPart struct {
	offset int
	rank   uint
}

func (c *countCodec) countPiece(ctx context.Context, piece string) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	ranks, err := c.ranks.load(ctx, c.encoding)
	if err != nil {
		return 0, err
	}
	if _, ok := ranks[piece]; ok {
		return 1, nil
	}
	parts := make([]countPart, len(piece)+1)
	for i := range parts {
		if i%countPollInterval == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		parts[i] = countPart{offset: i, rank: math.MaxUint}
	}
	getRank := func(index, skip int) uint {
		if index+skip+2 < len(parts) {
			if rank, ok := ranks[piece[parts[index].offset:parts[index+skip+2].offset]]; ok {
				return rank
			}
		}
		return math.MaxUint
	}
	for i := 0; i < len(parts)-2; i++ {
		if i%countPollInterval == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		parts[i].rank = getRank(i, 0)
	}
	for len(parts) > 1 {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		minRank, minIndex := uint(math.MaxUint), 0
		for i, part := range parts[:len(parts)-1] {
			if i%countPollInterval == 0 {
				if err := ctx.Err(); err != nil {
					return 0, err
				}
			}
			if part.rank < minRank {
				minRank, minIndex = part.rank, i
			}
		}
		if minRank == math.MaxUint {
			break
		}
		parts[minIndex].rank = getRank(minIndex, 1)
		if minIndex > 0 {
			parts[minIndex-1].rank = getRank(minIndex-1, 1)
		}
		// Remove the selected boundary in bounded copies, preserving upstream order.
		for start := minIndex + 1; start < len(parts)-1; {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
			end := min(start+countPollInterval, len(parts)-1)
			copy(parts[start:end], parts[start+1:end+1])
			start = end
		}
		parts = parts[:len(parts)-1]
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	return len(parts) - 1, nil
}
