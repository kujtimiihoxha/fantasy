package fantasy

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestToolResultPartsJSONRoundTrip(t *testing.T) {
	t.Parallel()

	part := ToolResultPart{
		ToolCallID: "call-1",
		Output: ToolResultOutputContentParts{Parts: []ToolResultOutputPart{
			NewToolResultTextPart("first"),
			NewToolResultMediaPart([]byte{1, 2, 3}, "image/png"),
			NewToolResultMediaPart([]byte{4, 5}, "image/jpeg"),
		}},
	}
	data, err := json.Marshal(part)
	require.NoError(t, err)

	var decoded ToolResultPart
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, part.Output, decoded.Output)

	output, err := UnmarshalToolResultOutputContent(mustMarshal(t, part.Output))
	require.NoError(t, err)
	require.Equal(t, part.Output, output)
}

func TestToolResultOutputPartKinds(t *testing.T) {
	t.Parallel()

	require.False(t, NewToolResultTextPart("text").IsMedia())
	media := NewToolResultMediaPart([]byte{9}, "image/gif")
	require.True(t, media.IsMedia())
	require.Equal(t, base64.StdEncoding.EncodeToString([]byte{9}), media.Data)
}

func TestAgentPartsToolResponse(t *testing.T) {
	t.Parallel()

	parts := []ToolResultOutputPart{
		NewToolResultMediaPart([]byte{1}, "image/png"),
		NewToolResultMediaPart([]byte{2}, "image/png"),
	}
	tool := &mockTool{
		name:        "view",
		description: "Shows images",
		executeFunc: func(context.Context, ToolCall) (ToolResponse, error) {
			return NewPartsResponse(parts...), nil
		},
	}
	model := &mockLanguageModel{
		generateFunc: func(_ context.Context, call Call) (*Response, error) {
			if len(call.Prompt) == 1 {
				return &Response{
					Content:      []Content{ToolCallContent{ToolCallID: "view-1", ToolName: "view", Input: `{}`}},
					FinishReason: FinishReasonToolCalls,
				}, nil
			}
			return &Response{Content: []Content{TextContent{Text: "done"}}, FinishReason: FinishReasonStop}, nil
		},
	}
	agent := NewAgent(model, WithTools(tool), WithStopConditions(StepCountIs(3)))

	result, err := agent.Generate(context.Background(), AgentCall{Prompt: "Show the images"})

	require.NoError(t, err)
	results := result.Steps[0].Content.ToolResults()
	require.Len(t, results, 1)
	require.Equal(t, ToolResultOutputContentParts{Parts: parts}, results[0].Result)
}

func mustMarshal(t *testing.T, value any) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	require.NoError(t, err)
	return data
}
