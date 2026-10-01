package gemini

import "net/http"

// Option configures a provider at construction.
type Option func(*Provider)

// WithBaseURL sets the endpoint used by discovery, portable generation, and
// native access. Use a trusted endpoint; requests carry the provider API key.
func WithBaseURL(baseURL string) Option { return func(p *Provider) { p.baseURL = baseURL } }

// WithHTTPClient uses a shallow copy of client. Its transport and callbacks
// remain caller-owned and must be safe for concurrent use. Nil keeps the default.
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		if client != nil {
			copy := *client
			p.httpClient = &copy
		}
	}
}
