// Package loop executes iterative model conversations with function tools.
//
// It is GAI's deliberate low-level execution API: Loop exposes mutable run
// configuration and completed Iterations for applications that need direct
// orchestration control. Higher-level applications should use agent for
// reusable definitions, per-run configuration, workflow lifecycle, middleware,
// and aggregated copied results.
//
// A Loop builds a canonical request, streams model tokens, executes Tools, adds
// identified results to subsequent requests, and stops when the model returns a final
// response. Each run exposes one ordered Event stream containing tokens,
// attempt starts, retries, completed iterations, and terminal results.
// Iteration retains attempt execution diagnostics and canonical message snapshots.
// ToolExecution.MaxConcurrent bounds active tool pipelines; zero preserves
// unlimited concurrency. WithToolOptions adds per-name Serial execution and
// handler timeouts. Serial includes result processing within one run; shared
// ToolGuard registrations also serialize across runs. Timeouts start at admission,
// excluding queue and approval waits. Handlers must honor cancellation: the loop
// joins admitted work before returning. Handler panics terminate the run without
// exposing panic payloads. Model retries never retry a tool invocation.
//
// ToolPolicy runs before scheduling using copied plain call metadata. Nil policy
// allows registered tools; NewToolPolicy supplies name/effect rules with explicit
// precedence. Workflow middleware is not this authorization boundary. Policies
// can allow, deny, or require approval. Result processors run after invocation and
// before any outer result publication; ChainToolResultProcessors composes them.
// RedactToolResult sanitizes errors as well as text. Put LimitToolResultBytes last.
// ToolExecution records authorization, invocation, and output disposition
// independently: rejected output never undoes a successful external side effect.
//
// Only accepted iterations enter Loop.Messages; retries and discards stay in events.
// Message views clone the retained ai.Message values instead of reconstructing
// different native and rendered transcripts from execution records.
package loop
