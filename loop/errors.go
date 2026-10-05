package loop

import (
	"errors"
	"fmt"
)

var (
	// ErrToolPolicy identifies an authorization implementation failure.
	ErrToolPolicy = errors.New("tool policy failed")
	// ErrToolDenied identifies a policy refusal before invocation.
	ErrToolDenied = errors.New("tool execution denied")
	// ErrToolApprovalRequired identifies a call awaiting application approval.
	ErrToolApprovalRequired = errors.New("tool approval required")
	// ErrToolApproval identifies resolver failures and mismatched responses.
	ErrToolApproval = errors.New("tool approval failed")
	// ErrToolResultRejected identifies withheld output after processing.
	ErrToolResultRejected = errors.New("tool output rejected")
	// ErrToolOutputLimit is a safe replacement for oversized output.
	ErrToolOutputLimit = fmt.Errorf("%w: output exceeds byte limit", ErrToolResultRejected)

	// ErrToolExecutionConfig identifies invalid scheduling or registration options.
	ErrToolExecutionConfig = errors.New("invalid tool execution configuration")
	// ErrToolPanic is a terminal handler/processor panic; panic values are not published.
	ErrToolPanic = errors.New("tool execution panicked")
	// ErrNilLoop indicates an operation on a nil Loop.
	ErrNilLoop = errors.New("loop is nil")
	// ErrModelNotConfigured indicates that a Loop has no model.
	ErrModelNotConfigured = errors.New("model is not configured")
	// ErrPromptNotConfigured indicates that a Loop has no prompt builder.
	ErrPromptNotConfigured = errors.New("prompt builder is not configured")
	// ErrRequiredToolNotConfigured indicates that a required named tool is absent from Loop.Tools.
	ErrRequiredToolNotConfigured = errors.New("required tool is not configured")
	// ErrToolReqValidation indicates that a tool request failed validation.
	ErrToolReqValidation = errors.New("invalid tool call")
	// ErrToolCallMalformed indicates malformed tool-call arguments.
	ErrToolCallMalformed = errors.New("tool call payload is malformed")
	// ErrToolNotFound indicates that no configured tool matches a call.
	ErrToolNotFound = errors.New("tool not found")
	// ErrMaxIterations indicates that the loop reached its iteration limit.
	ErrMaxIterations = errors.New("max loop iterations exceeded")
	// ErrPromptPathEmpty indicates that no prompt file path was provided.
	ErrPromptPathEmpty = errors.New("prompt path is empty")
	// ErrPromptFileType indicates an unsupported prompt file extension.
	ErrPromptFileType = errors.New("prompt file must be .md or .txt")
	// ErrPromptMissing indicates that a prompt file does not exist.
	ErrPromptMissing = errors.New("prompt file is missing")
	// ErrArgsDecodeTarget indicates a nil target passed to DecodeToolArgs.
	ErrArgsDecodeTarget = errors.New("tool args decode target is nil")
	// ErrToolResultProcess indicates that tool-response processing failed.
	ErrToolResultProcess = errors.New("tool response process error")
	// ErrBuildPrompt indicates that prompt construction failed.
	ErrBuildPrompt = errors.New("build prompt error")
	// ErrMaxRetries indicates that model generation exhausted its retry limit.
	ErrMaxRetries = errors.New("max retries exceeded")
)
