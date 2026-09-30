package ai

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ProviderErrorKind is a provider-neutral classification used by retry policy.
type ProviderErrorKind string

const (
	ProviderErrorUnknown        ProviderErrorKind = "unknown"
	ProviderErrorRateLimited    ProviderErrorKind = "rate_limited"
	ProviderErrorTransient      ProviderErrorKind = "transient"
	ProviderErrorAuthentication ProviderErrorKind = "authentication"
	ProviderErrorInvalidRequest ProviderErrorKind = "invalid_request"
	ProviderErrorUnsupported    ProviderErrorKind = "unsupported"
)

// ProviderError carries safe, provider-neutral failure metadata. Provider
// adapters should wrap vendor errors in this type before returning them.
type ProviderError struct {
	Kind       ProviderErrorKind
	StatusCode int
	Code       string
	RequestID  string
	RetryAfter time.Duration
	Err        error
}

func (e *ProviderError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.Err != nil {
		return fmt.Sprintf("provider %s error: %v", e.Kind, e.Err)
	}
	return fmt.Sprintf("provider %s error", e.Kind)
}

func (e *ProviderError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// ClassifyProviderError adds provider-neutral retry metadata to an HTTP API
// error. It preserves err in the returned error chain.
func ClassifyProviderError(err error, statusCode int, code, requestID string, header http.Header) error {
	if err == nil {
		return nil
	}
	var provider *ProviderError
	if errors.As(err, &provider) {
		return err
	}
	return &ProviderError{
		Kind:       providerErrorKind(statusCode, code),
		StatusCode: statusCode,
		Code:       code,
		RequestID:  requestID,
		RetryAfter: retryAfter(header.Get("Retry-After")),
		Err:        err,
	}
}

func providerErrorKind(statusCode int, code string) ProviderErrorKind {
	code = strings.ToLower(code)
	switch {
	case statusCode == http.StatusTooManyRequests || strings.Contains(code, "rate_limit") || strings.Contains(code, "rate-limit"):
		return ProviderErrorRateLimited
	case statusCode == http.StatusNotImplemented || strings.Contains(code, "unsupported"):
		return ProviderErrorUnsupported
	case statusCode == http.StatusRequestTimeout || statusCode == http.StatusConflict || statusCode == http.StatusTooEarly || statusCode >= http.StatusInternalServerError || strings.Contains(code, "server_error") || strings.Contains(code, "overload") || strings.Contains(code, "temporar"):
		return ProviderErrorTransient
	case statusCode == http.StatusUnauthorized || statusCode == http.StatusForbidden || strings.Contains(code, "auth") || strings.Contains(code, "api_key"):
		return ProviderErrorAuthentication
	case statusCode >= http.StatusBadRequest && statusCode < http.StatusInternalServerError:
		return ProviderErrorInvalidRequest
	default:
		return ProviderErrorUnknown
	}
}

func retryAfter(value string) time.Duration {
	value = strings.TrimSpace(value)
	if seconds, err := strconv.ParseInt(value, 10, 64); err == nil {
		if seconds > 0 {
			return time.Duration(seconds) * time.Second
		}
		return 0
	}
	if retryAt, err := http.ParseTime(value); err == nil {
		if delay := time.Until(retryAt); delay > 0 {
			return delay
		}
	}
	return 0
}

var (
	// ErrModelNotFound indicates that a requested model is unavailable.
	ErrModelNotFound = errors.New("model not found")
	// ErrProviderDown indicates that the provider cannot currently serve requests.
	ErrProviderDown = errors.New("provider unavailable")
	// ErrProviderNotFound indicates that a requested provider is not registered.
	ErrProviderNotFound = errors.New("provider not found")
	// ErrProviderInvalid indicates that provider configuration is invalid.
	ErrProviderInvalid = errors.New("provider is invalid")
	// ErrProviderAlreadyExists indicates that a provider name is already registered.
	ErrProviderAlreadyExists = errors.New("provider already exists")
	// ErrNilProvider indicates that a provider argument is nil.
	ErrNilProvider = errors.New("provider is nil")
	// ErrNilModelRepository indicates an operation on a nil repository.
	ErrNilModelRepository = errors.New("model repository is nil")
	// ErrInvalidToolCall indicates a malformed tool call.
	ErrInvalidToolCall = errors.New("invalid tool call")
	// ErrInvalidToolDefinition indicates a malformed tool definition.
	ErrInvalidToolDefinition = errors.New("invalid tool definition")
	// ErrInvalidResponseFormat indicates a malformed structured response request.
	ErrInvalidResponseFormat = errors.New("invalid response format")
	// ErrUnsupportedCapability indicates that a provider cannot satisfy a
	// requested feature or optional operation, such as model discovery.
	ErrUnsupportedCapability = errors.New("unsupported provider capability")
	// ErrTokenizerUnsupported indicates that a tokenizer does not support an operation.
	ErrTokenizerUnsupported = errors.New("tokenizer operation unsupported")
)
