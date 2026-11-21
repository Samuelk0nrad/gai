package gemini

import (
	"encoding/json"
	"fmt"

	"github.com/Samuelk0nrad/gai/ai"
	"google.golang.org/genai"
)

const (
	extensionNamespace = "google"
	// hostedToolResultExtension keeps a server-side tool result part, such as
	// toolResponse or codeExecutionResult, as JSON for replay.
	hostedToolResultExtension = "hosted_tool_result"
	// groundingMetadataExtension keeps the candidate's grounding metadata as
	// the API returned it, including the search entry point to display.
	groundingMetadataExtension = "grounding_metadata"
)

// isHostedPart reports whether part is a server-side tool call or result,
// which the API expects back in later turns.
func isHostedPart(part *genai.Part) bool {
	return part.ToolCall != nil || part.ToolResponse != nil || part.ExecutableCode != nil || part.CodeExecutionResult != nil
}

func hostedToolKind(toolType genai.ToolType) string {
	switch toolType {
	case genai.ToolTypeGoogleSearchWeb, genai.ToolTypeGoogleSearchImage:
		return ai.HostedToolWebSearch
	case genai.ToolTypeURLContext:
		return ai.HostedToolURLContext
	case genai.ToolTypeGoogleMaps:
		return ai.HostedToolMaps
	case genai.ToolTypeFileSearch:
		return ai.HostedToolFileSearch
	default:
		return ai.HostedToolOther
	}
}

// hostedPart keeps a server-side tool part. Calls become hosted tool calls;
// results are kept for replay only.
func hostedPart(part *genai.Part) (ai.ContentPart, error) {
	if hasUnsupportedGeminiPartPayload(&genai.Part{
		MediaResolution: part.MediaResolution, FileData: part.FileData, FunctionResponse: part.FunctionResponse,
		InlineData: part.InlineData, VideoMetadata: part.VideoMetadata, PartMetadata: part.PartMetadata,
		AudioTranscription: part.AudioTranscription, MediaProcessing: part.MediaProcessing,
	}) || part.Text != "" || part.FunctionCall != nil {
		return ai.ContentPart{}, fmt.Errorf("%w: Gemini server-side tool part with other content", ai.ErrUnsupportedCapability)
	}
	raw, err := json.Marshal(part)
	if err != nil {
		return ai.ContentPart{}, fmt.Errorf("encode Gemini server-side tool part: %w", err)
	}
	switch {
	case part.ToolCall != nil:
		call := ai.HostedToolCall{ID: part.ToolCall.ID, Kind: hostedToolKind(part.ToolCall.ToolType), Name: string(part.ToolCall.ToolType), Raw: raw}
		call.Query, _ = part.ToolCall.Args["query"].(string)
		call.URL, _ = part.ToolCall.Args["url"].(string)
		return ai.HostedToolCallPart(extensionNamespace, call), nil
	case part.ExecutableCode != nil:
		return ai.HostedToolCallPart(extensionNamespace, ai.HostedToolCall{ID: part.ExecutableCode.ID, Kind: ai.HostedToolCodeExecution, Name: "code_execution", Raw: raw}), nil
	default:
		return ai.ContentPart{Kind: ai.ContentExtension, Extensions: []ai.Extension{{Namespace: extensionNamespace, Type: hostedToolResultExtension, Data: raw}}}, nil
	}
}

// hostedPartInput returns the server-side tool part an extension part keeps
// for replay, or nil when it keeps none.
func hostedPartInput(extensions []ai.Extension) (*genai.Part, error) {
	for _, ext := range extensions {
		if ext.Namespace != extensionNamespace {
			continue
		}
		var raw json.RawMessage
		if ext.Type == hostedToolResultExtension {
			raw = ext.Data
		} else if call, ok := ai.DecodeHostedToolCall(ext); ok {
			raw = call.Raw
		}
		if len(raw) == 0 {
			continue
		}
		var part genai.Part
		if err := json.Unmarshal(raw, &part); err != nil {
			return nil, fmt.Errorf("decode Gemini server-side tool part: %w", err)
		}
		return &part, nil
	}
	return nil, nil
}

// groundingPart keeps the candidate's grounding metadata: its sources as
// citations, the raw metadata, and, unless the response already returned its
// searches as server-side tool calls, one hosted tool call per search query.
// The part is not replayed.
func groundingPart(metadata *genai.GroundingMetadata, searchCallsReturned bool) (ai.ContentPart, bool) {
	if metadata == nil {
		return ai.ContentPart{}, false
	}
	raw, err := json.Marshal(metadata)
	if err != nil || string(raw) == "{}" {
		return ai.ContentPart{}, false
	}
	var extensions []ai.Extension
	if !searchCallsReturned {
		for _, query := range metadata.WebSearchQueries {
			extensions = append(extensions, ai.HostedToolCallPart(extensionNamespace, ai.HostedToolCall{Kind: ai.HostedToolWebSearch, Name: "google_search", Status: "completed", Query: query}).Extensions...)
		}
		for _, query := range metadata.ImageSearchQueries {
			extensions = append(extensions, ai.HostedToolCallPart(extensionNamespace, ai.HostedToolCall{Kind: ai.HostedToolWebSearch, Name: "google_image_search", Status: "completed", Query: query}).Extensions...)
		}
	}
	extensions = append(extensions, ai.CitationsExtension(extensionNamespace, groundingCitations(metadata))...)
	extensions = append(extensions, ai.Extension{Namespace: extensionNamespace, Type: groundingMetadataExtension, Data: raw})
	return ai.ContentPart{Kind: ai.ContentExtension, Extensions: extensions}, true
}

// groundingCitations returns one citation per supported segment and source.
// Sources no segment refers to are listed without text.
func groundingCitations(metadata *genai.GroundingMetadata) []ai.Citation {
	source := func(index int32) (ai.Citation, bool) {
		if index < 0 || int(index) >= len(metadata.GroundingChunks) || metadata.GroundingChunks[index] == nil {
			return ai.Citation{}, false
		}
		chunk := metadata.GroundingChunks[index]
		raw, _ := json.Marshal(chunk)
		citation := ai.Citation{Raw: raw}
		switch {
		case chunk.Web != nil:
			citation.Type, citation.URL, citation.Title = "web", chunk.Web.URI, chunk.Web.Title
		case chunk.RetrievedContext != nil:
			citation.Type, citation.URL, citation.Title, citation.Source = "retrieved_context", chunk.RetrievedContext.URI, chunk.RetrievedContext.Title, chunk.RetrievedContext.Text
		case chunk.Maps != nil:
			citation.Type, citation.URL, citation.Title = "maps", chunk.Maps.URI, chunk.Maps.Title
		default:
			citation.Type = "other"
		}
		return citation, true
	}
	var citations []ai.Citation
	used := map[int32]bool{}
	for _, support := range metadata.GroundingSupports {
		if support == nil {
			continue
		}
		text := ""
		if support.Segment != nil {
			text = support.Segment.Text
		}
		for _, index := range support.GroundingChunkIndices {
			if citation, ok := source(index); ok {
				citation.Text = text
				citations = append(citations, citation)
				used[index] = true
			}
		}
	}
	for i := range metadata.GroundingChunks {
		if !used[int32(i)] {
			if citation, ok := source(int32(i)); ok {
				citations = append(citations, citation)
			}
		}
	}
	return citations
}

func isSearchCall(part *genai.Part) bool {
	return part.ToolCall != nil && hostedToolKind(part.ToolCall.ToolType) == ai.HostedToolWebSearch
}
