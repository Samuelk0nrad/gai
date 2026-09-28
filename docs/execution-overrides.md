# Execution overrides

An agent's concrete `Definition` provides defaults. `RunInput.Execution` is an
optional patch for one workflow. A nil patch and an empty patch both inherit
all defaults; the definition is never changed.

| Setting | Inherit | Replace or clear |
| --- | --- | --- |
| Model | nil interface | Supply a model; a model is required after resolution |
| Limits.MaxTokens | nil pointer | Positive limit per generation; pointer to zero selects adapter/provider defaults |
| Limits.MaxLoopIterations | nil pointer | Positive iteration limit; pointer to zero selects the loop default |
| Tools | nil slice | Replace the entire list; non-nil empty slice disables tools |
| ToolChoice, ResponseFormat, Reasoning | nil pointer | Replace the entire object, including its zero fields |
| Tokenizer | Optional with Set false | Set true selects Value; nil clears the custom tokenizer and selects the effective model's tokenizer |
| RetryPolicy | Optional with Set false | Set true selects Value; nil removes both retries and policy timeouts |
| ToolResponseProcessor | Optional with Set false | Set true selects Value; nil disables processing |

`Optional[T]{Set: false}` ignores `Value`, even when it is nonzero. A pointer to
an empty tool-choice, response-format, or reasoning object resets that entire
setting to its zero value. There is no recursive object merge. Only limits
inherit independently, field by field.

`MaxTokens: 0` does not promise unlimited output: each provider/adapter chooses
its default. This is an output limit per generation, not a total workflow
budget. A retry policy with `MaxRetries: 0` can still enforce attempt and total
timeouts; removing the policy is a separate operation.

## Resolution and prompt builders

`NewRun` overlays the requested values, validates the effective configuration,
and resolves model capabilities, tool transport, and tokenizer before calling
the prompt factory. Invalid defaults can be replaced by valid overrides.
Negative effective limits, invalid retry policies/response formats/tool
selection, and typed-nil dependency interfaces fail construction. A required
tool choice with no matching tools also fails before the prompt factory runs.
Provider-specific capability validation still happens when the model handles
its request.

The prompt callback keeps the same signature. Its input contains a copy of the
requested patch, without filling in inherited values. It must return a fresh,
run-owned builder. Models with native tools receive native tool definitions;
text-tool models receive the tool protocol in the prompt. Changes to tools,
tool choice, or model reconcile an existing named `tool_definitions` source.
When that requires replacing or removing an existing source, builders that
cannot manage it fail construction. The resolved tool choice takes precedence
over `tooldefinitions.WithToolChoice` in automatic-source options.

A custom definition tokenizer is inherited even when the model changes. To
select the new model's tokenizer instead, explicitly clear it:

```go
input := agent.RunInput{
    Execution: &agent.ExecutionOverrides{
        Model: otherModel,
        Tokenizer: agent.Optional[ai.Tokenizer]{Set: true},
    },
}
```

Compatible builders receive the resolved tokenizer through
`context.TokenizerSetter`, including nil when the model has none. An explicit
model override, custom tokenizer selection, or custom tokenizer clear requires
that interface; otherwise `NewRun` returns `agent.ErrTokenizerNotConfigurable`.
An ordinary inherited model with no custom tokenizer continues to support
opaque third-party builders without that interface. This does not change
token-count estimators or the cache contracts of arbitrary context sources.

## Ownership and middleware

`agent.New` snapshots the definition's mutable configuration. `NewRun` snapshots
the requested patch, passes the prompt factory a separate copy, and retains
separate resolved configuration. Mutating a caller's slices, schema bytes,
limit pointers, or retry-policy structs after those calls does not change the
agent/workflow. Workflow-result input snapshots are independent as well.

Models, tool implementations, tokenizers, processors, middleware implementations,
and callback closures remain shared dependencies. Callers must make them safe
for concurrent use; snapshotting does not clone their internal state.

Middleware agents use their own defaults. The default input mapper does not
forward the primary agent's execution overrides. Use `MapInput` to deliberately
select settings for a nested agent.

## Migrating callers

This is a breaking pre-v1 API change:

- Replace `Execution: agent.ExecutionConfig{...}` with
  `Execution: &agent.ExecutionOverrides{...}`.
- Move `RunInput.MaxTokens` to
  `Execution.Limits.MaxTokens` using an integer pointer.
- Move `RunInput.ResponseFormat` to `Execution.ResponseFormat` using a pointer.
- Put reusable tool choice and response format defaults on `Definition`.
- Use `Optional` only for the three nullable override dependencies above.

Previously, a nonpositive run token limit inherited the definition limit.
To preserve that behavior, leave the new pointer nil. Use a pointer to zero
only when intentionally selecting provider defaults; negative values now fail
construction. The summary helper's own `summary.Request.MaxTokens` contract
continues to inherit for nonpositive values.
