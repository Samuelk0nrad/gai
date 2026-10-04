// Package exa provides prebuilt tools backed by the Exa API.
package exa

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
)

const (
	defaultEndpoint    = "https://api.exa.ai/search"
	defaultSearchType  = "auto"
	defaultNumResults  = 10
	maxResponseBytes   = 4 << 20
	maxErrorBodyBytes  = 4 << 10
	apiKeyEnvironment  = "EXA_API_KEY"
	defaultHTTPTimeout = 60 * time.Second
)

var (
	ErrAPIKeyMissing    = errors.New("exa API key is missing")
	ErrInvalidOption    = errors.New("invalid Exa search option")
	ErrResponseTooLarge = errors.New("exa response is too large")
	ErrInvalidResponse  = errors.New("exa returned invalid JSON")
	ErrAPIRequest       = errors.New("exa search API request failed")
)

// APIError is returned when Exa responds with a non-success HTTP status.
type APIError struct {
	StatusCode int
	RequestID  string
	Message    string
}

func (e *APIError) Error() string {
	if e == nil {
		return ErrAPIRequest.Error()
	}
	detail := fmt.Sprintf("%s: status %d", ErrAPIRequest, e.StatusCode)
	if e.RequestID != "" {
		detail += ", request " + e.RequestID
	}
	if e.Message != "" {
		detail += ": " + e.Message
	}
	return detail
}

func (e *APIError) Unwrap() error {
	return ErrAPIRequest
}

// SearchTool searches the web with Exa and returns the raw JSON response to the
// agent. Its default request uses type=auto and query-relevant highlights.
type SearchTool struct {
	apiKey      string
	endpoint    string
	client      *http.Client
	searchType  string
	numResults  int
	debug       gai.ObservationSink
	description string
	params      ai.ToolParameters
}

var _ loop.Tool = (*SearchTool)(nil)

// Option configures a SearchTool.
type Option func(*SearchTool) error

// WithSearchType sets the Exa search type.
func WithSearchType(searchType string) Option {
	return func(tool *SearchTool) error {
		switch searchType {
		case "auto", "fast", "instant", "deep-lite", "deep", "deep-reasoning":
			tool.searchType = searchType
			return nil
		default:
			return fmt.Errorf("%w: unsupported search type %q", ErrInvalidOption, searchType)
		}
	}
}

// WithNumResults sets the number of search results requested from Exa.
func WithNumResults(numResults int) Option {
	return func(tool *SearchTool) error {
		if numResults < 1 || numResults > 100 {
			return fmt.Errorf("%w: num results must be between 1 and 100", ErrInvalidOption)
		}
		tool.numResults = numResults
		return nil
	}
}

// WithHTTPClient replaces the HTTP client. It is useful for setting transport,
// proxy, and timeout policy at the application boundary.
func WithHTTPClient(client *http.Client) Option {
	return func(tool *SearchTool) error {
		if client == nil {
			return fmt.Errorf("%w: HTTP client is nil", ErrInvalidOption)
		}
		tool.client = client
		return nil
	}
}

// WithObservationSink emits lifecycle and failure events for the search operation.
// Query text is emitted only when the sink opts into sensitive data.
func WithObservationSink(debug gai.ObservationSink) Option {
	return func(tool *SearchTool) error {
		tool.debug = debug
		return nil
	}
}

// WithEndpoint replaces the Exa endpoint. Most applications should use the
// default; this option also supports compatible gateways and local tests.
func WithEndpoint(endpoint string) Option {
	return func(tool *SearchTool) error {
		endpoint = strings.TrimSpace(endpoint)
		if endpoint == "" {
			return fmt.Errorf("%w: endpoint is empty", ErrInvalidOption)
		}
		tool.endpoint = endpoint
		return nil
	}
}

// WithPromptDescription overrides the prompt-facing tool description.
func WithPromptDescription(description string) Option {
	return func(tool *SearchTool) error {
		description = strings.TrimSpace(description)
		if description == "" {
			return fmt.Errorf("%w: prompt description is empty", ErrInvalidOption)
		}
		tool.description = description
		return nil
	}
}

// WithPromptParams overrides the prompt-facing schema for tool arguments.
func WithPromptParams(params ai.ToolParameters) Option {
	return func(tool *SearchTool) error {
		if _, err := params.JSONSchema(); err != nil {
			return fmt.Errorf("%w: prompt params invalid: %w", ErrInvalidOption, err)
		}
		tool.params = params
		return nil
	}
}

// NewSearchTool constructs an Exa web search tool.
func NewSearchTool(apiKey string, options ...Option) (*SearchTool, error) {
	apiKey = strings.TrimSpace(apiKey)
	if apiKey == "" {
		return nil, ErrAPIKeyMissing
	}

	tool := &SearchTool{
		apiKey:      apiKey,
		endpoint:    defaultEndpoint,
		client:      &http.Client{Timeout: defaultHTTPTimeout},
		searchType:  defaultSearchType,
		numResults:  defaultNumResults,
		description: "Searches the web for current or factual information and returns relevant pages with excerpts.",
		params: ai.ToolParameters{
			Strict: true,
			Properties: []ai.ToolParameter{
				{
					Name:        "query",
					Type:        ai.ToolParameterString,
					Description: "A specific natural-language web search query",
					Required:    true,
				},
			},
		},
	}
	for _, option := range options {
		if option == nil {
			return nil, fmt.Errorf("%w: option is nil", ErrInvalidOption)
		}
		if err := option(tool); err != nil {
			return nil, err
		}
	}
	return tool, nil
}

// NewSearchToolFromEnv constructs a tool using EXA_API_KEY.
func NewSearchToolFromEnv(options ...Option) (*SearchTool, error) {
	return NewSearchTool(os.Getenv(apiKeyEnvironment), options...)
}

func (t *SearchTool) Name() string {
	return "web_search"
}

func (t *SearchTool) Description() string {
	return t.description
}

func (t *SearchTool) Params() ai.ToolParameters {
	return t.params
}

type searchArgs struct {
	Query string `json:"query"`
}

type searchRequest struct {
	Query      string         `json:"query"`
	Type       string         `json:"type"`
	NumResults int            `json:"numResults"`
	Contents   searchContents `json:"contents"`
}

type searchContents struct {
	Highlights bool `json:"highlights"`
}

// Function searches Exa and returns JSON; it honors ctx and preserves direct panic behavior.
func (t *SearchTool) Function(ctx context.Context, call ai.ToolCall) (text string, err error) {
	ctx, observer := newSearchObserver(ctx, t.debug, t.searchType, t.numResults)
	defer func() {
		if value := recover(); value != nil {
			observer.Finish(errObservedExaSearch)
			panic(value)
		}
		observer.Finish(err)
	}()

	var args searchArgs
	if err := loop.DecodeToolArgs(call, &args); err != nil {
		return observer.Failure(ctx, "decode_arguments", err)
	}
	args.Query = strings.TrimSpace(args.Query)
	if args.Query == "" {
		return observer.Failure(ctx, "validate_query", fmt.Errorf("%w: query is empty", loop.ErrToolReqValidation))
	}
	observer.Started(ctx, args.Query)

	payload, err := json.Marshal(searchRequest{
		Query:      args.Query,
		Type:       t.searchType,
		NumResults: t.numResults,
		Contents:   searchContents{Highlights: true},
	})
	if err != nil {
		return observer.Failure(ctx, "marshal_request", fmt.Errorf("marshal Exa search request: %w", err))
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.endpoint, bytes.NewReader(payload))
	if err != nil {
		return observer.Failure(ctx, "create_request", fmt.Errorf("create Exa search request: %w", err))
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", t.apiKey)

	res, err := t.client.Do(req)
	if err != nil {
		return observer.Failure(ctx, "execute_request", fmt.Errorf("%w: %w", ErrAPIRequest, err))
	}
	defer res.Body.Close()
	observer.ResponseReceived(res.StatusCode)

	body, err := io.ReadAll(io.LimitReader(res.Body, maxResponseBytes+1))
	if err != nil {
		return observer.Failure(ctx, "read_response", fmt.Errorf("read Exa search response: %w", err))
	}
	if len(body) > maxResponseBytes {
		return observer.Failure(ctx, "limit_response", ErrResponseTooLarge)
	}
	if res.StatusCode < http.StatusOK || res.StatusCode >= http.StatusMultipleChoices {
		apiErr := decodeAPIError(res.StatusCode, res.Header.Get("x-request-id"), body)
		observer.RequestID(apiErr.RequestID)
		return observer.Failure(ctx, "api_response", apiErr)
	}
	if !json.Valid(body) {
		return observer.Failure(ctx, "decode_response", ErrInvalidResponse)
	}
	var metadata *struct {
		RequestID string            `json:"requestId"`
		Results   []json.RawMessage `json:"results"`
	}
	if err := json.Unmarshal(body, &metadata); err != nil {
		return observer.Failure(ctx, "decode_response", fmt.Errorf("%w: %w", ErrInvalidResponse, err))
	}
	if metadata == nil {
		return observer.Failure(ctx, "decode_response", ErrInvalidResponse)
	}
	observer.Succeeded(ctx, metadata.RequestID, len(metadata.Results), len(body))

	return string(body), nil
}

func decodeAPIError(statusCode int, requestID string, body []byte) *APIError {
	var payload struct {
		Error     string `json:"error"`
		RequestID string `json:"requestId"`
	}
	_ = json.Unmarshal(body, &payload)
	if requestID == "" {
		requestID = payload.RequestID
	}
	message := strings.TrimSpace(payload.Error)
	if message == "" {
		message = strings.TrimSpace(string(body))
	}
	if message == "" {
		message = http.StatusText(statusCode)
	}
	if len(message) > maxErrorBodyBytes {
		message = message[:maxErrorBodyBytes]
	}
	return &APIError{StatusCode: statusCode, RequestID: requestID, Message: message}
}
