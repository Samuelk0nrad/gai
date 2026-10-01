package openai

import (
	"context"
	"fmt"

	"github.com/lace-ai/gai/internal/syncutil"
	tiktoken "github.com/tiktoken-go/tokenizer"
)

const tokenCountBlockRunes = 10000

type tokenCodecCache struct {
	mu    syncutil.ContextMutex
	codec tiktoken.Codec
}

var cl100kTokenCodec, o200kTokenCodec tokenCodecCache

func localTokenCodec(ctx context.Context, encoding tiktoken.Encoding) (tiktoken.Codec, error) {
	var cache *tokenCodecCache
	switch encoding {
	case tiktoken.Cl100kBase:
		cache = &cl100kTokenCodec
	case tiktoken.O200kBase:
		cache = &o200kTokenCodec
	default:
		return nil, tiktoken.ErrEncodingNotSupported
	}
	if err := cache.mu.Lock(ctx); err != nil {
		return nil, err
	}
	defer cache.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if cache.codec == nil {
		codec, err := tiktoken.Get(encoding)
		if err != nil {
			return nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		cache.codec = codec
	}
	return cache.codec, nil
}

// countTokenBlocks leaves tokenization to the upstream library. Cancellation
// waits for the current bounded block to finish; no work continues after return.
// Splitting can change counts at block boundaries, so this is an estimate of
// the full text's token count, not a guaranteed upper or lower bound.
func countTokenBlocks(ctx context.Context, codec tiktoken.Codec, text string) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	count := 0
	for text != "" {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		end := tokenCountBlockEnd(text)
		n, err := codec.Count(text[:end])
		if canceled := ctx.Err(); canceled != nil {
			return 0, canceled
		}
		if err != nil {
			return 0, fmt.Errorf("count OpenAI tokens: %w", err)
		}
		count += n
		text = text[end:]
	}
	return count, nil
}

func tokenCountBlockEnd(text string) int {
	if len(text) <= tokenCountBlockRunes {
		return len(text)
	}
	characters := 0
	for offset := range text {
		if characters == tokenCountBlockRunes {
			return offset
		}
		characters++
	}
	return len(text)
}
