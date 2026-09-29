# Order support agent

This example shows the shortest end-to-end GAI workflow that is still representative of a real application:

- an OpenAI-backed agent;
- a typed `lookup_order` tool;
- provider-native tool calling;
- ordered streaming through `Workflow.RunEvents`, followed by `Workflow.Wait` for the final result;
- deterministic local data, so only one API key is required.

## Run it

From the repository root:

```bash
export OPENAI_API_KEY="..."
go run ./examples/order-support
```

The default question looks up the included mock order `LACE-1042`. You can pass another prompt as command-line arguments:

```bash
go run ./examples/order-support "Has order LACE-1042 left the warehouse?"
```

The example defaults to `gpt-4.1-mini`. Override the model without changing code:

```bash
OPENAI_MODEL="your-chat-completions-model" go run ./examples/order-support
```

A typical response looks like this:

```text
User: Where is order LACE-1042, and when should it arrive?
Assistant: Order LACE-1042 is in transit and has left the Vienna logistics center. Austrian Post currently estimates delivery on August 3, 2026.
```

Model wording can vary. The shipping facts come from the local tool result in `main.go`, not from the model.

## What to inspect

`lookupOrderTool` demonstrates the complete `loop.Tool` contract:

1. `Params` defines a typed JSON Schema for the model.
2. `Function` decodes arguments with `loop.DecodeToolArgs`.
3. The tool returns structured JSON with `loop.NewToolSuccess`.
4. The loop adds the tool result to the conversation and asks the model for the final response.

The tests call the tool directly and do not require network access:

```bash
go test ./examples/order-support
```
