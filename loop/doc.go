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
// Only accepted iterations enter Loop.Messages; retries and discards stay in events.
// Message views clone the retained ai.Message values instead of reconstructing
// different native and rendered transcripts from execution records.
package loop
