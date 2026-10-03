package anthropic

import (
	"context"
	"errors"
	"fmt"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/lace-ai/gai/ai"
)

var _ ai.InputTokenCounter = (*Model)(nil)

// CountInputTokens explicitly calls the Messages count_tokens endpoint using
// the same portable input mapping as generation. Anthropic returns an estimate
// that may differ from actual usage. Native parameter hooks are unsupported:
// independently rerunning a mutable hook cannot preserve request equivalence.
func (m *Model) CountInputTokens(ctx context.Context, request ai.AIRequest) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if len(m.messageHooks) > 0 {
		return 0, &ai.InputTokenCountUnsupportedError{Model: m.name, Reason: "native message parameter hooks cannot be counted equivalently"}
	}
	request = request.Copy()
	if err := ai.ValidateModelRequest(m, request); err != nil {
		return 0, countMappingError(m.name, err)
	}
	params, err := m.messageParams(request)
	if err != nil {
		return 0, countMappingError(m.name, err)
	}
	countParams := sdk.MessageCountTokensParams{
		Model: params.Model, Messages: params.Messages,
		System:     sdk.MessageCountTokensParamsSystemUnion{OfTextBlockArray: params.System},
		ToolChoice: params.ToolChoice, Thinking: params.Thinking, OutputConfig: params.OutputConfig,
		CacheControl: params.CacheControl, UserProfileID: params.UserProfileID, WorkspaceID: params.WorkspaceID,
	}
	for _, tool := range params.Tools {
		countParams.Tools = append(countParams.Tools, sdk.MessageCountTokensToolUnionParam(tool))
	}
	client := m.client.sdkClient()
	count, err := client.Messages.CountTokens(ctx, countParams)
	if err != nil {
		return 0, classifyProviderError(err)
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if count == nil || !count.JSON.InputTokens.Valid() {
		return 0, fmt.Errorf("anthropic count_tokens response is missing input_tokens")
	}
	if count.InputTokens < 0 || uint64(count.InputTokens) > uint64(^uint(0)>>1) {
		return 0, fmt.Errorf("invalid anthropic input token count: %d", count.InputTokens)
	}
	return int(count.InputTokens), nil
}

func countMappingError(model string, err error) error {
	if errors.Is(err, ai.ErrUnsupportedCapability) {
		return &ai.InputTokenCountUnsupportedError{Model: model, Reason: err.Error()}
	}
	return err
}
