package ollama

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sync"

	ollamaapi "github.com/ollama/ollama/api"
)

type responseCapture struct {
	mu        sync.Mutex
	status    int
	header    http.Header
	errorBody []byte
	records   [][]byte
	partial   []byte
}

func (c *responseCapture) setResponse(response *http.Response) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.status = response.StatusCode
	c.header = response.Header.Clone()
}

func (c *responseCapture) setErrorBody(body []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.errorBody = append([]byte(nil), body...)
}

func (c *responseCapture) observe(data []byte, eof bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.partial = append(c.partial, data...)
	for {
		index := bytes.IndexByte(c.partial, '\n')
		if index < 0 {
			break
		}
		record := c.partial[:index]
		if len(record) > 0 && record[len(record)-1] == '\r' {
			record = record[:len(record)-1]
		}
		c.records = append(c.records, append([]byte(nil), record...))
		c.partial = append(c.partial[:0], c.partial[index+1:]...)
	}
	if eof && len(c.partial) > 0 {
		c.records = append(c.records, append([]byte(nil), c.partial...))
		c.partial = nil
	}
}

func (c *responseCapture) popRecord() []byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.records) == 0 {
		return nil
	}
	record := c.records[0]
	copy(c.records, c.records[1:])
	c.records = c.records[:len(c.records)-1]
	return append([]byte(nil), record...)
}

func (c *responseCapture) unconsumedRecord() []byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.records) > 0 {
		return append([]byte(nil), c.records[0]...)
	}
	return append([]byte(nil), c.partial...)
}

func (c *responseCapture) metadata() (int, http.Header, []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.status, c.header.Clone(), append([]byte(nil), c.errorBody...)
}

type recordingBody struct {
	io.ReadCloser
	capture *responseCapture
}

func (b *recordingBody) Read(data []byte) (int, error) {
	n, err := b.ReadCloser.Read(data)
	b.capture.observe(data[:n], err == io.EOF)
	return n, err
}

type sdkTransport struct {
	base        http.RoundTripper
	bearerToken string
	schemas     []json.RawMessage
	capture     *responseCapture
}

func (t *sdkTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	request = request.Clone(request.Context())
	request.Header = request.Header.Clone()
	if err := overlayToolSchemas(request, t.schemas); err != nil {
		return nil, fmt.Errorf("prepare Ollama SDK request: %w", err)
	}
	if t.bearerToken != "" {
		request.Header.Set("Authorization", "Bearer "+t.bearerToken)
	}
	response, err := t.base.RoundTrip(request)
	if err != nil {
		return nil, err
	}
	t.capture.setResponse(response)
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		limited := io.LimitReader(response.Body, maxErrorBodySize+1)
		body, readErr := io.ReadAll(limited)
		_ = response.Body.Close()
		if readErr != nil {
			body = nil
		}
		if len(body) > maxErrorBodySize {
			body = body[:maxErrorBodySize]
		}
		t.capture.setErrorBody(body)
		response.Body = io.NopCloser(bytes.NewReader(body))
		response.ContentLength = int64(len(body))
		return response, nil
	}
	response.Body = &recordingBody{ReadCloser: response.Body, capture: t.capture}
	return response, nil
}

func overlayToolSchemas(request *http.Request, schemas []json.RawMessage) error {
	if len(schemas) == 0 || request.Body == nil {
		return nil
	}
	body, err := io.ReadAll(request.Body)
	if err != nil {
		return err
	}
	_ = request.Body.Close()

	var root map[string]json.RawMessage
	if err := json.Unmarshal(body, &root); err != nil {
		return err
	}
	var tools []json.RawMessage
	if err := json.Unmarshal(root["tools"], &tools); err != nil {
		return err
	}
	if len(tools) != len(schemas) {
		return fmt.Errorf("tool schema count changed: got %d, want %d", len(tools), len(schemas))
	}
	for index := range tools {
		var tool map[string]json.RawMessage
		if err := json.Unmarshal(tools[index], &tool); err != nil {
			return err
		}
		var function map[string]json.RawMessage
		if err := json.Unmarshal(tool["function"], &function); err != nil {
			return err
		}
		function["parameters"] = append(json.RawMessage(nil), schemas[index]...)
		encodedFunction, err := json.Marshal(function)
		if err != nil {
			return err
		}
		tool["function"] = encodedFunction
		encodedTool, err := json.Marshal(tool)
		if err != nil {
			return err
		}
		tools[index] = encodedTool
	}
	encodedTools, err := json.Marshal(tools)
	if err != nil {
		return err
	}
	root["tools"] = encodedTools
	body, err = json.Marshal(root)
	if err != nil {
		return err
	}
	setRequestBody(request, body)
	return nil
}

func setRequestBody(request *http.Request, body []byte) {
	request.Body = io.NopCloser(bytes.NewReader(body))
	request.ContentLength = int64(len(body))
	request.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(body)), nil
	}
}

func (p *Provider) sdkClient(schemas []json.RawMessage) (*ollamaapi.Client, *responseCapture, error) {
	base, err := p.baseOrigin()
	if err != nil {
		return nil, nil, err
	}
	client, err := p.requestClient()
	if err != nil {
		return nil, nil, err
	}
	transport := client.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	capture := &responseCapture{}
	client.Transport = &sdkTransport{
		base:        transport,
		bearerToken: p.bearerToken,
		schemas:     cloneSchemas(schemas),
		capture:     capture,
	}
	return ollamaapi.NewClient(cloneURL(base), client), capture, nil
}

func cloneSchemas(schemas []json.RawMessage) []json.RawMessage {
	result := make([]json.RawMessage, len(schemas))
	for index := range schemas {
		result[index] = append(json.RawMessage(nil), schemas[index]...)
	}
	return result
}

func cloneURL(value *url.URL) *url.URL {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}
