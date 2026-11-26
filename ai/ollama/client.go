package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// Ollama documents POST /api/chat in docs/openapi.yaml. The v0.40.18
// specification does not describe tool-call IDs, tool-result correlation, or
// terminal metadata on streamed records, so the small wire structs live in
// this package instead of being generated from an incomplete contract.
func (p *Provider) doChat(ctx context.Context, payload *chatRequest) (*http.Response, error) {
	base, err := p.baseOrigin()
	if err != nil {
		return nil, err
	}
	client, err := p.requestClient()
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode Ollama chat request: %w", err)
	}

	endpoint := base.JoinPath("api/chat")
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create Ollama chat request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/x-ndjson, application/json")
	if p.bearerToken != "" {
		request.Header.Set("Authorization", "Bearer "+p.bearerToken)
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	return response, nil
}
