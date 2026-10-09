# Ollama provider

Package `github.com/lace-ai/gai/ai/ollama` uses Ollama's official Go SDK to
stream the native `POST /api/chat` NDJSON protocol. It does not route through
Ollama's OpenAI-compatible endpoint. Named lookup accepts every non-empty model
name and does not require discovery or a hard-coded catalog.

## Prerequisites

Run Ollama and pull a model that supports tool calling. For example:

```sh
ollama serve
ollama pull qwen3:0.6b
```

Tool support is model-dependent. Verify the exact model tag against Ollama's model
page; installing a model does not imply that it can call tools reliably.

## Minimal tool-using agent

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"

    "github.com/lace-ai/gai/agent"
    "github.com/lace-ai/gai/ai"
    "github.com/lace-ai/gai/ai/ollama"
    gaictx "github.com/lace-ai/gai/context"
    "github.com/lace-ai/gai/loop"
)

func main() {
    ctx := context.Background()
    numCtx := 4096
    provider := ollama.New(nil)
    model, err := provider.TypedModel(
        "qwen3:0.6b",
        // Arbitrary model names default to unknown tool support. Opt in only
        // after choosing a model tag that supports native tools.
        ollama.WithToolSupport(ai.FeatureSupportSupported),
        ollama.WithOptions(ollama.Options{NumCtx: &numCtx}),
    )
    if err != nil {
        panic(err)
    }

    weather, err := loop.NewTool(
        "get_weather",
        "Get the current weather for a city",
        ai.ToolParameters{Properties: []ai.ToolParameter{{
            Name: "city", Type: ai.ToolParameterString, Required: true,
        }}},
        func(_ context.Context, call ai.ToolCall) (string, error) {
            var args struct{ City string `json:"city"` }
            if err := loop.DecodeToolArgs(call, &args); err != nil {
                return "", err
            }
            result, err := json.Marshal(struct {
                City    string `json:"city"`
                Celsius int    `json:"celsius"`
            }{City: args.City, Celsius: 21})
            return string(result), err
        },
    )
    if err != nil {
        panic(err)
    }

    assistant := agent.New(agent.Definition{
        Model: model,
        Tools: []loop.Tool{weather},
        Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
            return gaictx.New(gaictx.Definition{SystemInstructions: []gaictx.Part{
                gaictx.NewTextPart("Use get_weather for weather questions."),
            }}), nil
        },
    })
    workflow, err := assistant.NewRun(ctx, agent.RunInput{
        Prompt: gaictx.PromptInput{User: ai.TextParts("Weather in Tokyo?")},
    })
    if err != nil {
        panic(err)
    }
    result, err := workflow.Run(ctx)
    if err != nil {
        panic(err)
    }
    fmt.Println(result.Text)
}
```

The loop receives the native call, executes `get_weather`, replays the assistant
call and correlated tool result in the next Ollama request, and streams the final
answer.

## Configuration

```go
client := &http.Client{Transport: transport}
provider := ollama.New(nil,
    ollama.WithBaseURL("http://localhost:11434"), // the default
    ollama.WithHTTPClient(client),
    ollama.WithBearerToken(os.Getenv("OLLAMA_TOKEN")), // optional
)

temperature := 0.2
numCtx := 8192
model, err := provider.TypedModel("qwen3:8b",
    ollama.WithOptions(ollama.Options{
        NumCtx: &numCtx,
        Temperature: &temperature,
    }),
)
```

`WithHTTPClient` shallow-copies the client. Its transport and callbacks remain
caller-owned and must be safe for concurrent use. Context deadlines and the
client's own timeout control request lifetime; the provider does not impose a
short whole-stream timeout. Redirects are restricted to the configured origin
before a prompt or bearer token is sent. Bearer authentication is intended for
protected Ollama-compatible endpoints. The official SDK also honors
`OLLAMA_AUTH` and its built-in `ollama.com` signing flow before the HTTP
transport runs; bearer-only clients should leave `OLLAMA_AUTH` unset and use a
non-`ollama.com` endpoint.

`AIRequest.MaxTokens` maps to native `options.num_predict`. `Options.NumCtx` maps
to `options.num_ctx`; it controls Ollama's context window and does not truncate
GAI history or replace GAI request-budget configuration. Models and providers
are immutable after construction and can be shared across concurrent runs.

## Capabilities and boundaries

| Capability | Adapter behavior |
|---|---|
| Canonical system/user/assistant messages | Native mapping |
| Native function tools and correlated results | Supported for opted-in, tool-capable models |
| Missing response call IDs | A unique canonical ID is generated once and replayed |
| Streaming text | Emitted incrementally from NDJSON records |
| Finish reason and usage | `done_reason`, `prompt_eval_count`, and `eval_count` are reported on terminal completion |
| Automatic tool selection | Supported |
| Required, named, or disabled tool choice | Rejected explicitly |
| Structured output | Rejected explicitly in the initial adapter scope |
| Portable reasoning controls | Rejected explicitly; requests send `think:false` |
| Media | Rejected explicitly |
| Model discovery | Not implemented; arbitrary named lookup remains available |

Unknown model names deliberately report native tool support as `Unknown`, not
`Supported`. Without `WithToolSupport(ai.FeatureSupportSupported)`, an agent uses
GAI's text tool protocol. This avoids advertising model-dependent behavior that
the provider cannot infer from a name alone. Portable reasoning is outside the
initial scope, so the adapter requests `think:false`; a model or server that
cannot disable thinking is not compatible with this adapter version and
thinking output is rejected rather than silently exposed as answer text.

The stream ends successfully only after a `done:true` record. Malformed records,
in-band errors, invalid tool arguments, and EOF before terminal completion are
errors. HTTP failures are returned as classified `ai.ProviderError` values.
Cancellation propagates through the HTTP request, and every returned token
channel is closed.

## Live compatibility smoke test

Offline fixture tests validate protocol mapping but do not prove model behavior.
With Ollama running and the documented model installed, run the opt-in live test:

```sh
OLLAMA_SMOKE_MODEL=qwen3:0.6b \
OLLAMA_BASE_URL=http://localhost:11434 \
go test -v ./ai/ollama -run '^TestLiveToolCalling$' -count=1
```

The test requires a real native tool call, executes the tool through GAI's agent
loop, validates the replay transcript, streams a final answer, and checks terminal
usage. It was validated against Ollama 0.40.1 with `qwen3:0.6b`; model output
quality and latency remain model- and hardware-dependent.
