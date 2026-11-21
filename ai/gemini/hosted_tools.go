package gemini

import (
	"encoding/json"
	"fmt"
	"slices"

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
		switch ext.Type {
		case hostedToolResultExtension:
			raw = ext.Data
		case ai.ExtensionHostedToolCall:
			call, ok := ai.DecodeHostedToolCall(ext)
			if !ok {
				return nil, fmt.Errorf("decode Gemini hosted tool call: invalid data")
			}
			// Calls known only from grounding metadata have no part.
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
		// Queries are counted once, as a merged stream counts them.
		for _, query := range appendNew(nil, metadata.WebSearchQueries) {
			extensions = append(extensions, ai.HostedToolCallPart(extensionNamespace, ai.HostedToolCall{Kind: ai.HostedToolWebSearch, Name: "google_search", Status: "completed", Query: query}).Extensions...)
		}
		for _, query := range appendNew(nil, metadata.ImageSearchQueries) {
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

// mergeGrounding adds the grounding metadata of one stream chunk to the
// metadata gathered so far. A chunk may repeat earlier metadata or carry only
// new data, so queries, sources and supports are merged without duplicates:
// queries by text, web and maps sources by URI, other sources by their whole
// content (several retrieved passages can share one document URI), and
// supports by segment and sources. Support source indices are renumbered into
// the merged sources. Other fields take the latest value.
func mergeGrounding(dst, src *genai.GroundingMetadata) *genai.GroundingMetadata {
	if src == nil {
		return dst
	}
	if dst == nil {
		dst = &genai.GroundingMetadata{}
	}
	dst.WebSearchQueries = appendNew(dst.WebSearchQueries, src.WebSearchQueries)
	dst.ImageSearchQueries = appendNew(dst.ImageSearchQueries, src.ImageSearchQueries)
	dst.RetrievalQueries = appendNew(dst.RetrievalQueries, src.RetrievalQueries)
	chunkIndex := map[string]int32{}
	for i, chunk := range dst.GroundingChunks {
		chunkIndex[groundingChunkKey(chunk)] = int32(i)
	}
	remap := make([]int32, len(src.GroundingChunks))
	for i, chunk := range src.GroundingChunks {
		key := groundingChunkKey(chunk)
		index, ok := chunkIndex[key]
		if !ok {
			index = int32(len(dst.GroundingChunks))
			dst.GroundingChunks = append(dst.GroundingChunks, chunk)
			chunkIndex[key] = index
		}
		remap[i] = index
	}
	supports := map[string]bool{}
	for _, support := range dst.GroundingSupports {
		supports[groundingSupportKey(support)] = true
	}
	for _, support := range src.GroundingSupports {
		if support == nil {
			continue
		}
		merged := *support
		merged.GroundingChunkIndices = nil
		for _, index := range support.GroundingChunkIndices {
			if index >= 0 && int(index) < len(remap) {
				merged.GroundingChunkIndices = append(merged.GroundingChunkIndices, remap[index])
			}
		}
		if key := groundingSupportKey(&merged); !supports[key] {
			supports[key] = true
			dst.GroundingSupports = append(dst.GroundingSupports, &merged)
		}
	}
	if src.SearchEntryPoint != nil {
		dst.SearchEntryPoint = src.SearchEntryPoint
	}
	if src.RetrievalMetadata != nil {
		dst.RetrievalMetadata = src.RetrievalMetadata
	}
	if src.GoogleMapsWidgetContextToken != "" {
		dst.GoogleMapsWidgetContextToken = src.GoogleMapsWidgetContextToken
	}
	return dst
}

func appendNew(dst, src []string) []string {
	for _, value := range src {
		if !slices.Contains(dst, value) {
			dst = append(dst, value)
		}
	}
	return dst
}

func groundingChunkKey(chunk *genai.GroundingChunk) string {
	switch {
	case chunk == nil:
		return ""
	case chunk.Web != nil && chunk.Web.URI != "":
		return "web:" + chunk.Web.URI
	case chunk.Maps != nil && chunk.Maps.URI != "":
		return "maps:" + chunk.Maps.URI
	}
	raw, _ := json.Marshal(chunk)
	return "json:" + string(raw)
}

func groundingSupportKey(support *genai.GroundingSupport) string {
	key := struct {
		Segment *genai.Segment `json:"s"`
		Chunks  []int32        `json:"c"`
	}{support.Segment, support.GroundingChunkIndices}
	raw, _ := json.Marshal(key)
	return string(raw)
}
