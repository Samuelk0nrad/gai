package mistral_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
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

// nativeRoundTripper exercises redirects through http.Client without dialing
// privileged default ports or contacting the provider.
type nativeRoundTripper func(*http.Request) (*http.Response, error)

func (f nativeRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

func TestNativeHTTPDefaultPortEquivalence(t *testing.T) {
	for _, tc := range []struct{ name, base, target string }{
		{"https explicit", "https://api.mistral.ai", "https://API.MISTRAL.AI:443"},
		{"https implicit", "https://api.mistral.ai:443", "https://api.mistral.ai"},
		{"http explicit", "http://api.mistral.ai", "http://api.mistral.ai:80"},
		{"http implicit", "http://api.mistral.ai:80", "http://api.mistral.ai"},
		{"ipv6 explicit", "https://[::1]", "https://[::1]:443"},
	} {
		for _, phase := range []string{"direct", "redirect", "host override", "callback host"} {
			t.Run(tc.name+"/"+phase, func(t *testing.T) {
				target, err := url.Parse(tc.target)
				if err != nil {
					t.Fatal(err)
				}
				calls := 0
				transport := &http.Client{Transport: nativeRoundTripper(func(req *http.Request) (*http.Response, error) {
					calls++
					if req.Header.Get("Authorization") != "Bearer test-key" {
						t.Error("authentication lost")
					}
					header := make(http.Header)
					code := http.StatusOK
					if (phase == "redirect" || phase == "callback host") && calls == 1 {
						code = http.StatusTemporaryRedirect
						header.Set("Location", tc.target+"/final")
					}
					return &http.Response{StatusCode: code, Header: header, Body: io.NopCloser(strings.NewReader("ok")), Request: req}, nil
				})}
				if phase == "callback host" {
					transport.CheckRedirect = func(req *http.Request, _ []*http.Request) error { req.Host = target.Host; return nil }
				}
				client, err := mistral.New("test-key", nil, mistral.WithBaseURL(tc.base), mistral.WithHTTPClient(transport)).NativeClient()
				if err != nil {
					t.Fatal(err)
				}
				requestURL := tc.base + "/start"
				if phase == "direct" {
					requestURL = tc.target + "/start"
				}
				req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, requestURL, nil)
				if err != nil {
					t.Fatal(err)
				}
				if phase == "host override" {
					req.Host = target.Host
				}
				res, err := client.Do(req)
				if err != nil {
					t.Fatal(err)
				}
				res.Body.Close()
				want := 1
				if phase == "redirect" || phase == "callback host" {
					want = 2
				}
				if calls != want || res.StatusCode != http.StatusOK {
					t.Fatalf("calls=%d status=%d", calls, res.StatusCode)
				}
			})
		}
	}
}

func TestNativeHTTPDefaultPortNormalizationKeepsOriginBoundary(t *testing.T) {
	for _, target := range []string{
		"https://foreign.example:443/final",
		"https://api.mistral.ai:444/final",
		"http://api.mistral.ai:443/final",
		"https://user@api.mistral.ai:443/final",
	} {
		for _, redirect := range []bool{false, true} {
			t.Run(target+"/redirect="+strconv.FormatBool(redirect), func(t *testing.T) {
				calls := 0
				transport := &http.Client{Transport: nativeRoundTripper(func(req *http.Request) (*http.Response, error) {
					calls++
					if calls > 1 || !redirect {
						t.Error("rejected destination reached transport")
					}
					return &http.Response{StatusCode: http.StatusTemporaryRedirect, Header: http.Header{"Location": []string{target}}, Body: io.NopCloser(strings.NewReader("")), Request: req}, nil
				})}
				client, err := mistral.New("test-key", nil, mistral.WithHTTPClient(transport)).NativeClient()
				if err != nil {
					t.Fatal(err)
				}
				requestURL := target
				if redirect {
					requestURL = "https://api.mistral.ai/start"
				}
				req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, requestURL, nil)
				if err != nil {
					t.Fatal(err)
				}
				res, err := client.Do(req)
				if res != nil {
					res.Body.Close()
				}
				if err == nil {
					t.Fatal("accepted different origin")
				}
				want := 0
				if redirect {
					want = 1
				}
				if calls != want {
					t.Fatalf("transport calls=%d, want %d", calls, want)
				}
			})
		}
	}
	for _, host := range []string{"api.mistral.ai:444", "foreign.example:443", "user@api.mistral.ai:443", "api.mistral.ai:443/path", "api.mistral.ai:443?query", "api.mistral.ai:443#fragment", "api.mistral.ai:invalid"} {
		for _, callback := range []bool{false, true} {
			t.Run("Host="+host+"/callback="+strconv.FormatBool(callback), func(t *testing.T) {
				calls := 0
				transport := &http.Client{Transport: nativeRoundTripper(func(req *http.Request) (*http.Response, error) {
					calls++
					if !callback || calls > 1 {
						t.Error("invalid Host reached transport")
					}
					return &http.Response{StatusCode: http.StatusTemporaryRedirect, Header: http.Header{"Location": []string{"https://api.mistral.ai:443/final"}}, Body: io.NopCloser(strings.NewReader("")), Request: req}, nil
				})}
				if callback {
					transport.CheckRedirect = func(req *http.Request, _ []*http.Request) error { req.Host = host; return nil }
				}
				client, err := mistral.New("test-key", nil, mistral.WithHTTPClient(transport)).NativeClient()
				if err != nil {
					t.Fatal(err)
				}
				req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://api.mistral.ai/start", nil)
				if !callback {
					req.Host = host
				}
				res, err := client.Do(req)
				if res != nil {
					res.Body.Close()
				}
				if err == nil {
					t.Fatal("accepted invalid Host")
				}
			})
		}
	}
}

func TestNativeHTTPRedirectOwnsAuthenticationAfterValidation(t *testing.T) {
	denied := errors.New("redirect denied")
	for _, mode := range []string{"clear", "nil", "replace", "reject", "stop"} {
		t.Run(mode, func(t *testing.T) {
			calls := 0
			var redirect *http.Request
			transport := &http.Client{
				Transport: nativeRoundTripper(func(req *http.Request) (*http.Response, error) {
					calls++
					if req.Header.Get("Authorization") != "Bearer test-key" {
						t.Error("provider authentication lost or replaced")
					}
					header := make(http.Header)
					status := http.StatusOK
					if calls == 1 {
						status = http.StatusFound
						header.Set("Location", "https://API.MISTRAL.AI:443/final")
					}
					return &http.Response{StatusCode: status, Header: header, Body: io.NopCloser(strings.NewReader("")), Request: req}, nil
				}),
				CheckRedirect: func(req *http.Request, _ []*http.Request) error {
					redirect = req
					switch mode {
					case "clear":
						req.Header.Del("Authorization")
					case "nil":
						req.Header = nil
					case "replace":
						req.Header.Set("Authorization", "Bearer other-key")
					case "reject":
						req.Header = nil
						return denied
					case "stop":
						req.Header = nil
						return http.ErrUseLastResponse
					}
					return nil
				},
			}
			client, err := mistral.New("test-key", nil, mistral.WithHTTPClient(transport)).NativeClient()
			if err != nil {
				t.Fatal(err)
			}
			req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://api.mistral.ai/start", nil)
			if err != nil {
				t.Fatal(err)
			}
			res, err := client.Do(req)
			if res != nil {
				defer res.Body.Close()
			}
			if redirect == nil {
				t.Fatal("redirect callback was not invoked")
			}
			if mode == "reject" || mode == "stop" {
				if calls != 1 || redirect.Header != nil {
					t.Fatal("rejected redirect was sent or reauthenticated")
				}
				if mode == "reject" && !errors.Is(err, denied) {
					t.Fatalf("callback error lost: %v", err)
				}
				if mode == "stop" && (err != nil || res == nil || res.StatusCode != http.StatusFound) {
					t.Fatalf("ErrUseLastResponse not preserved: response=%v err=%v", res, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if calls != 2 || res.StatusCode != http.StatusOK {
				t.Fatalf("calls=%d status=%d", calls, res.StatusCode)
			}
		})
	}
}
