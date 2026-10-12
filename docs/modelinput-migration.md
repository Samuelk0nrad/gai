# Migrating to modelinput

`modelinput` replaces GAI's `context` package. It assembles model-facing
instructions, sources, input, history, tool definitions, rendering, and budgets.
The new name can be imported alongside Go's standard library `context` without
an alias.

## Change imports when convenient

| Former path | Preferred path |
| --- | --- |
| `github.com/lace-ai/gai/context` | `github.com/lace-ai/gai/modelinput` |
| `github.com/lace-ai/gai/context/history` | `github.com/lace-ai/gai/modelinput/history` |
| `github.com/lace-ai/gai/context/tooldefinitions` | `github.com/lace-ai/gai/modelinput/tooldefinitions` |

For the smallest consumer change, keep your existing local import name:

```go
import gaictx "github.com/lace-ai/gai/modelinput"
```

Or use the package's new name directly:

```go
import (
    "context"
    "github.com/lace-ai/gai/ai"
    "github.com/lace-ai/gai/modelinput"
)

func prompt(ctx context.Context) (*modelinput.Builder, error) {
    builder := modelinput.New(modelinput.Definition{
        SystemInstructions: []modelinput.Part{
            modelinput.NewTextPart("Follow the application policy."),
        },
        PromptInput: modelinput.PromptInput{User: ai.TextParts("Hello")},
    })
    _, err := builder.BuildContext(ctx)
    return builder, err
}
```

Public names, struct fields, method signatures, defaults, history JSON schemas,
and observation names are unchanged. This move does not introduce `Source`,
`Input`, or `Config` replacements or change history's summarizer contract.

## Existing imports continue to work

The old paths remain deprecated compatibility packages. Exported types are
aliases to the canonical implementation, so old and new types are identical:
their values, interfaces, slices, callbacks, and builder pointers can be used
together. Constructors and helpers forward to the same implementation; sentinel
errors retain their identity for comparisons and `errors.Is`. History conflict
errors remain compatible with `errors.As` through either import path.

Keep the compatibility packages until consumers have migrated. Removing them
requires a future breaking release; this change sets no removal date. Consumers
such as ETIVERA, Still, and `lace-ai/agent-backend` can update imports in their
own release window without coordinating an immediate upgrade. GAI's internal
consumers migrate in separate, ordered PRs after the compatibility layer lands.
Machine-readable `Deprecated:` package comments identify the old paths to Go
tooling. These markers land after the internal migrations, so each intermediate
PR passes the existing deprecation lint checks. Compatibility tests intentionally
exercise the old paths and suppress only those deprecation warnings.

Type identity belongs to the new declaring package. Reflection's `PkgPath`,
formatted type names, and implicit `encoding/gob` type names can therefore change
to `modelinput`. Applications relying on those strings should use explicit
stable names or adapt them as part of the upgrade. JSON records are unaffected.

Go has no variable aliases. The compatibility sentinel variables hold the same
error values, but assigning a different value to one package's variable does
not reassign the other package's variable. Treat exported sentinels as stable
error values rather than mutable configuration.
