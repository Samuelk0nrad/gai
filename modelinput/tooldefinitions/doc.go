// Package tooldefinitions exposes loop tool schemas as prompt context.
//
// Source converts configured tools into a context part containing their names,
// descriptions, and argument definitions. Agents can include this source in a
// prompt Builder so the model sees the same tool contract used by the execution
// loop.
package tooldefinitions
