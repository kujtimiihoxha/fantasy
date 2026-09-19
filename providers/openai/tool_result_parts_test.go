package openai

import (
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func partsPrompt(parts ...fantasy.ToolResultOutputPart) fantasy.Prompt {
	return fantasy.Prompt{
		{
			Role: fantasy.MessageRoleAssistant,
			Content: []fantasy.MessagePart{
				fantasy.ToolCallPart{ToolCallID: "parts-1", ToolName: "view", Input: "{}"},
			},
		},
		{
			Role: fantasy.MessageRoleTool,
			Content: []fantasy.MessagePart{
				fantasy.ToolResultPart{ToolCallID: "parts-1", Output: fantasy.ToolResultOutputContentParts{Parts: parts}},
			},
		},
	}
}

// The Responses API accepts a content list as function call output, so the
// images stay inside the tool result without a synthetic user message.
func TestToResponsesPrompt_PartsToolResult(t *testing.T) {
	t.Parallel()

	first := fantasy.NewToolResultMediaPart([]byte{1, 2}, "image/png")
	second := fantasy.NewToolResultMediaPart([]byte{3}, "image/jpeg")
	input, warnings := toResponsesPrompt(partsPrompt(fantasy.NewToolResultTextPart("exit_code: 0"), first, second), "system", false, false)

	require.Empty(t, warnings)
	require.Len(t, input, 2)
	output := input[1].OfFunctionCallOutput
	require.NotNil(t, output)
	require.Equal(t, "parts-1", output.CallID.Value)
	items := output.Output.OfResponseFunctionCallOutputItemArray
	require.Len(t, items, 3)
	require.Equal(t, "exit_code: 0", items[0].OfInputText.Text)
	require.Equal(t, "data:image/png;base64,"+first.Data, items[1].OfInputImage.ImageURL.Value)
	require.Equal(t, "data:image/jpeg;base64,"+second.Data, items[2].OfInputImage.ImageURL.Value)
}

func TestToResponsesPrompt_PartsToolResultDropsOtherMedia(t *testing.T) {
	t.Parallel()

	input, warnings := toResponsesPrompt(partsPrompt(
		fantasy.NewToolResultMediaPart([]byte{1}, "video/mp4"),
		fantasy.NewToolResultMediaPart([]byte{2}, "image/png"),
	), "system", false, false)

	require.Len(t, warnings, 1)
	require.Contains(t, warnings[0].Message, "video/mp4")
	items := input[1].OfFunctionCallOutput.Output.OfResponseFunctionCallOutputItemArray
	require.Len(t, items, 1)
	require.NotNil(t, items[0].OfInputImage)
}

// Chat Completions tool messages carry only text, so the images follow the
// run of tool messages in one user message.
func TestDefaultToPrompt_PartsToolResult(t *testing.T) {
	t.Parallel()

	first := fantasy.NewToolResultMediaPart([]byte{1}, "image/png")
	second := fantasy.NewToolResultMediaPart([]byte{2}, "image/png")
	messages, warnings := DefaultToPrompt(partsPrompt(fantasy.NewToolResultTextPart("shown"), first, second), "openai", "gpt-4o")

	require.Empty(t, warnings)
	require.Len(t, messages, 3)
	require.Equal(t, "shown", messages[1].OfTool.Content.OfString.Value)
	parts := messages[2].OfUser.Content.OfArrayOfContentParts
	require.Len(t, parts, 2)
	require.Equal(t, "data:image/png;base64,"+first.Data, parts[0].OfImageURL.ImageURL.URL)
	require.Equal(t, "data:image/png;base64,"+second.Data, parts[1].OfImageURL.ImageURL.URL)
}

func TestDefaultToPrompt_PartsToolResultImagesOnly(t *testing.T) {
	t.Parallel()

	messages, warnings := DefaultToPrompt(partsPrompt(fantasy.NewToolResultMediaPart([]byte{1}, "image/png")), "openai", "gpt-4o")

	require.Empty(t, warnings)
	require.Len(t, messages, 3)
	require.Contains(t, messages[1].OfTool.Content.OfString.Value, "see the following user message")
	require.Len(t, messages[2].OfUser.Content.OfArrayOfContentParts, 1)
}
