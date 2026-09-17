package openai

import (
	"context"
	"errors"
	"fmt"
	"io"

	"charm.land/fantasy"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

var _ fantasy.Compactor = responsesLanguageModel{}

// Compact requests one checkpoint through the Responses compaction trigger.
func (o responsesLanguageModel) Compact(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	// A forced tool call would prevent the model from producing a checkpoint.
	call.ToolChoice = nil
	params, warnings, err := o.prepareParams(call)
	if err != nil {
		return nil, err
	}
	params.Input.OfInputItemList = append(params.Input.OfInputItemList, responses.ResponseInputItemUnionParam{
		OfCompactionTrigger: &responses.ResponseInputItemCompactionTriggerParam{},
	})
	params.ParallelToolCalls = param.NewOpt(true)
	capture := responseCapture{}
	stream := o.client.Responses.NewStreaming(ctx, *params, capture.requestOptions(o.headerFunc, append(callUARequestOptions(call), callHeadersRequestOptions(call)...))...)
	defer func() { _ = stream.Close() }()

	var checkpoint responses.ResponseOutputItemUnion
	var completed responses.Response
	var count int
	sawCompleted := false
streamLoop:
	for stream.Next() {
		event := stream.Current()
		switch event.Type {
		case "response.output_item.done":
			if event.Item.Type == "compaction" {
				checkpoint = event.Item
				count++
			}
		case "response.completed":
			completed = event.Response
			sawCompleted = true
			// Terminal output is authoritative when the provider includes it.
			if len(completed.Output) > 0 {
				count = 0
				for _, item := range completed.Output {
					if item.Type == "compaction" {
						checkpoint = item
						count++
					}
				}
			}
			break streamLoop
		case "response.failed":
			return nil, responsesFailedStreamError(event.Response.Error.Message, string(event.Response.Error.Code))
		case "response.incomplete":
			return nil, responsesStreamFailureError("compaction incomplete", event.Response.IncompleteDetails.Reason, "incomplete")
		case "error":
			return nil, responsesErrorStreamError(event.Message, event.Code)
		}
	}
	if !sawCompleted {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := stream.Err(); err != nil && !errors.Is(err, io.EOF) {
			return nil, toProviderErr(err)
		}
		return nil, fantasy.NewIncompleteStreamError()
	}
	if count != 1 {
		return nil, fmt.Errorf("compaction expected one checkpoint, got %d", count)
	}
	if checkpoint.EncryptedContent == "" {
		return nil, errors.New("compaction checkpoint has no encrypted content")
	}
	metadata := responsesProviderMetadata(completed.ID)
	o.applyHeaders(capture.header(), &metadata)
	return &fantasy.Response{
		Content:          fantasy.ResponseContent{fantasy.CompactionContent{ProviderMetadata: responsesCompactionMetadata(checkpoint)}},
		Usage:            responsesUsage(completed),
		FinishReason:     fantasy.FinishReasonStop,
		Warnings:         warnings,
		ProviderMetadata: metadata,
	}, nil
}
