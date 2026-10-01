package mistral_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai/mistral"
)

func TestNativeHTTPPreservesPayloadAndResponse(t *testing.T) {
	const body = `{"model":"native-model","messages":[{"role":"user","content":[{"type":"image_url","image_url":"https://example.org/image.png"}]}],"tools":[{"type":"web_search"}]}`
	const response = `{"native_field":{"value":42}}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		if string(data) != body || r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer test-key" || r.Header.Get("X-Native") != "value" {
			t.Errorf("native request changed: %s %s", r.URL.Path, data)
		}
		w.Header().Set("X-Native-Response", "preserved")
		w.WriteHeader(http.StatusTeapot)
		_, _ = io.WriteString(w, response)
	}))
	defer server.Close()
	client, err := mistral.New("test-key", nil, mistral.WithBaseURL(server.URL), mistral.WithHTTPClient(server.Client())).NativeClient()
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("X-Native", "value")
	req.Header.Set("Content-Type", "application/json")
	res, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	data, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != http.StatusTeapot || string(data) != response || res.Header.Get("X-Native-Response") != "preserved" {
		t.Fatalf("response altered: %d %s", res.StatusCode, data)
	}
	if req.Header.Get("Authorization") != "" || req.URL.String() != "v1/chat/completions" {
		t.Fatal("caller request mutated")
	}
}

func TestNativeHTTPRestrictsAuthenticatedOrigin(t *testing.T) {
	var foreignRequests atomic.Int32
	foreign := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { foreignRequests.Add(1) }))
	defer foreign.Close()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, foreign.URL, http.StatusTemporaryRedirect)
	}))
	defer server.Close()
	client, err := mistral.New("test-key", nil, mistral.WithBaseURL(server.URL)).NativeClient()
	if err != nil {
		t.Fatal(err)
	}
	for _, target := range []string{foreign.URL, "//" + strings.TrimPrefix(foreign.URL, "http://"), strings.Replace(server.URL, "http://", "http://user:password@", 1), server.URL} {
		req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, target, nil)
		if err != nil {
			t.Fatal(err)
		}
		res, err := client.Do(req)
		if res != nil {
			res.Body.Close()
		}
		if err == nil {
			t.Errorf("accepted target/redirect %s", target)
		}
	}
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, server.URL, nil)
	req.Host = "foreign.example"
	if _, err := client.Do(req); err == nil {
		t.Error("accepted Host override")
	}
	if foreignRequests.Load() != 0 {
		t.Fatalf("foreign server received %d requests", foreignRequests.Load())
	}
}

func TestNativeHTTPRechecksRedirectAfterCallback(t *testing.T) {
	var foreignRequests atomic.Int32
	foreign := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { foreignRequests.Add(1) }))
	defer foreign.Close()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, "/same-origin", http.StatusFound) }))
	defer server.Close()
	transport := server.Client()
	transport.CheckRedirect = func(req *http.Request, _ []*http.Request) error { req.URL, _ = url.Parse(foreign.URL); return nil }
	client, err := mistral.New("test-key", nil, mistral.WithBaseURL(server.URL), mistral.WithHTTPClient(transport)).NativeClient()
	if err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, server.URL, nil)
	res, err := client.Do(req)
	if res != nil {
		res.Body.Close()
	}
	if err == nil {
		t.Fatal("redirect callback changed origin")
	}
	if foreignRequests.Load() != 0 {
		t.Fatal("credential-bearing request reached changed origin")
	}
}

func TestNativeHTTPHonorsCancellationAndStreamLifetime(t *testing.T) {
	requestSeen := make(chan struct{}, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		w.(http.Flusher).Flush()
		requestSeen <- struct{}{}
		<-r.Context().Done()
	}))
	defer server.Close()
	transport := server.Client()
	transport.Timeout = time.Nanosecond
	client, err := mistral.New("test-key", nil, mistral.WithBaseURL(server.URL), mistral.WithHTTPClient(transport)).NativeClient()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, server.URL, nil)
	res, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	<-requestSeen
	cancel()
	_, err = io.ReadAll(res.Body)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("body error = %v", err)
	}
	if transport.Timeout != time.Nanosecond {
		t.Fatal("mutated caller client")
	}
}
