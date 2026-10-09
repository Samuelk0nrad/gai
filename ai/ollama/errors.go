package ollama

import (
	"encoding/json"
	"fmt"
)

// HTTPError reports a failed Ollama HTTP operation without including response
// content in Error. ResponseBody returns a defensive copy for diagnostics.
type HTTPError struct {
	StatusCode int
	body       []byte
}

func newHTTPError(statusCode int, body []byte) *HTTPError {
	return &HTTPError{StatusCode: statusCode, body: append([]byte(nil), body...)}
}

func (e *HTTPError) Error() string {
	if e == nil {
		return "Ollama request failed"
	}
	return fmt.Sprintf("Ollama chat request failed (status %d)", e.StatusCode)
}

func (e *HTTPError) ResponseBody() []byte {
	if e == nil {
		return nil
	}
	return append([]byte(nil), e.body...)
}

func ollamaErrorCode(body []byte) string {
	var payload struct {
		Error string `json:"error"`
		Code  string `json:"code"`
	}
	if json.Unmarshal(body, &payload) != nil {
		return ""
	}
	if payload.Code != "" {
		return payload.Code
	}
	return payload.Error
}
