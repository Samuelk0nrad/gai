package exa

import (
	"context"
	"errors"

	"github.com/lace-ai/gai"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

const exaTracerName = "github.com/lace-ai/gai/loop/tools/exa"

var errObservedExaSearch = errors.New("exa search failed")

type searchObserver struct {
	debug      gai.ObservationSink
	span       trace.Span
	searchType string
	numResults int
	statusCode int
	requestID  string
}

func newSearchObserver(ctx context.Context, debug gai.ObservationSink, searchType string, numResults int) (context.Context, *searchObserver) {
	ctx, span := gai.StartOperationSpan(ctx, exaTracerName, "tool.exa", "tool.operation", "search",
		attribute.String("tool.name", "web_search"),
		attribute.String("exa.search_type", searchType),
		attribute.Int("exa.num_results", numResults),
	)
	return ctx, &searchObserver{
		debug:      debug,
		span:       span,
		searchType: searchType,
		numResults: numResults,
	}
}

func (o *searchObserver) Finish(err error) {
	if o == nil || o.span == nil {
		return
	}
	attrs := make([]attribute.KeyValue, 0, 2)
	if o.statusCode != 0 {
		attrs = append(attrs, attribute.Int("http.response.status_code", o.statusCode))
	}
	if o.requestID != "" {
		attrs = append(attrs, attribute.String("exa.request_id", o.requestID))
	}
	if len(attrs) > 0 {
		o.span.SetAttributes(attrs...)
	}
	if err != nil {
		gai.EndSpan(o.span, errObservedExaSearch)
		return
	}
	gai.EndSpan(o.span, nil)
}

func (o *searchObserver) Started(ctx context.Context, query string) {
	fields := o.baseFields()
	gai.AddObservationContent(ctx, o.debug, fields, "query", gai.ContentKindToolInput, query)
	o.emit(ctx, "exa_search_started", fields, nil)
}

func (o *searchObserver) ResponseReceived(statusCode int) {
	if o == nil {
		return
	}
	o.statusCode = statusCode
	if o.span != nil {
		o.span.SetAttributes(attribute.Int("http.response.status_code", statusCode))
	}
}

func (o *searchObserver) RequestID(requestID string) {
	if o == nil {
		return
	}
	o.requestID = requestID
}

func (o *searchObserver) Succeeded(ctx context.Context, requestID string, resultCount int, responseBytes int) {
	if o == nil {
		return
	}
	o.requestID = requestID
	if o.span != nil {
		attrs := []attribute.KeyValue{
			attribute.Int("exa.result_count", resultCount),
			attribute.Int("exa.response_bytes", responseBytes),
		}
		if requestID != "" {
			attrs = append(attrs, attribute.String("exa.request_id", requestID))
		}
		o.span.SetAttributes(attrs...)
	}
	fields := o.baseFields()
	if requestID != "" {
		fields["request_id"] = requestID
	}
	fields["status_code"] = o.statusCode
	fields["result_count"] = resultCount
	fields["response_bytes"] = responseBytes
	o.emit(ctx, "exa_search_finished", fields, nil)
}

// Failure records sanitized diagnostics while returning the original error to execution.
func (o *searchObserver) Failure(ctx context.Context, stage string, err error) (string, error) {
	fields := o.baseFields()
	fields["stage"] = stage
	if o != nil && o.statusCode != 0 {
		fields["status_code"] = o.statusCode
	}
	if o != nil && o.requestID != "" {
		fields["request_id"] = o.requestID
	}
	if err != nil {
		gai.AddObservationContent(ctx, o.debug, fields, "error_output", gai.ContentKindToolOutput, err.Error())
	}
	observedErr := error(errObservedExaSearch)
	o.emit(ctx, "exa_search_failed", fields, observedErr)
	return "", err
}

func (o *searchObserver) baseFields() map[string]any {
	fields := map[string]any{}
	if o == nil {
		return fields
	}
	fields["search_type"] = o.searchType
	fields["num_results"] = o.numResults
	return fields
}

func (o *searchObserver) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if o == nil || o.debug == nil {
		return
	}
	gai.EmitObservation(ctx, o.debug, gai.Observation{
		Name:   name,
		Source: "loop/tools/exa:SearchTool.Function",
		Fields: fields,
		Err:    err,
	})
}
