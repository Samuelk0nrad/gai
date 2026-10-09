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
// response. Tools registered with ToolOptions.Terminal may instead complete the
// loop after a whole terminal-only batch succeeds, without another model request.
// By default, mixed terminal and ordinary batches are rejected before execution;
// terminal registrations can allow such batches to execute and continue. Terminal
// calls retain normal policy, approval, scheduling, processing, canonical result,
// accounting, and event behavior. Each run exposes one ordered Event stream
// containing tokens, attempt starts, retries, completed iterations, and terminal
// results.
// Iteration retains attempt execution diagnostics and canonical message snapshots.
// ToolExecution.MaxConcurrent bounds active tool pipelines; zero preserves
// unlimited concurrency. WithToolOptions adds per-name Serial execution and
// handler timeouts. Serial includes result processing within one run; shared
// ToolGuard registrations also serialize invoked pipelines across runs. Synthetic
// refusals bypass shared guards but retain bounded processing and local serial order. Timeouts start at admission,
// excluding queue and approval waits. Handlers must honor cancellation: the loop
// joins admitted work before returning. A handler's returned result is authoritative;
// only returned deadline errors are classified as tool timeouts. Calls rejected
// before invocation emit a tool error and finished metadata without a start event.
// Handler panics terminate the run without
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
// ToolApprovalResolver resolves policy-required approvals in process. Every
// policy and approval for a batch completes before any handler starts. Approval
// request IDs must be echoed exactly and are never reused; they are correlation
// tokens, not durable invocation identities. A missing resolver refuses the call
// with ErrToolApprovalRequired and does not suspend the run. Applications own UI,
// transport, and persistence; durable workflow pause/resume is a separate concern.
// Resolver failures retain a failed approval decision and emit a resolved event
// before the terminal failure. A resolver-local timeout remains a failure unless
// the run context is canceled. Run cancellation attempts nonblocking delivery;
// a full buffer can omit resolved and terminal snapshots, so consumers reconcile
// pending approvals when the stream closes.
//
// Only accepted iterations enter Loop.Messages; retries and discards stay in events.
// Message views clone the retained ai.Message values instead of reconstructing
// different native and rendered transcripts from execution records.
package loop
