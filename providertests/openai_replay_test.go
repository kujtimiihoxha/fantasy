package providertests

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"charm.land/x/vcr"
	"github.com/stretchr/testify/require"
)

func TestOpenAIResponsesFullReplay(t *testing.T) {
	if _, err := os.Stat("testdata/" + t.Name() + ".yaml"); os.IsNotExist(err) && os.Getenv("FANTASY_OPENAI_API_KEY") == "" {
		t.Skip("set FANTASY_OPENAI_API_KEY to record the replay test")
	}
	r := vcr.NewRecorder(t)
	model, err := openAIReasoningBuilder("gpt-5.6-luna")(t, r)
	require.NoError(t, err)
	tool := fantasy.NewAgentTool("lookup_number", "Return the stored number", func(context.Context, struct{}, fantasy.ToolCall) (fantasy.ToolResponse, error) {
		return fantasy.NewTextResponse("The stored number is 41."), nil
	})
	opts := openai.NewResponsesProviderOptions(&openai.ResponsesProviderOptions{
		FullReplay:       true,
		Store:            new(false),
		Include:          []openai.IncludeType{openai.IncludeReasoningEncryptedContent},
		ReasoningEffort:  openai.ReasoningEffortOption(openai.ReasoningEffortHigh),
		ReasoningSummary: new("auto"),
	})
	agent := fantasy.NewAgent(model, fantasy.WithTools(tool), fantasy.WithStopConditions(fantasy.StepCountIs(2)))
	const prompt = "Call lookup_number once. Add one to the stored number. Reply with only the result."
	result, err := agent.Stream(t.Context(), fantasy.AgentStreamCall{
		Prompt:          prompt,
		ProviderOptions: opts,
		MaxOutputTokens: new(int64(1024)),
		MaxRetries:      new(0),
	})
	require.NoError(t, err)
	require.Len(t, result.Steps, 2)
	require.Contains(t, result.Response.Content.Text(), "42")
	require.Equal(t, fantasy.FinishReasonStop, result.Response.FinishReason)

	history := []fantasy.Message{fantasy.NewUserMessage(prompt)}
	var reasoning, messages, calls int
	for _, step := range result.Steps {
		history = append(history, step.Messages...)
		for _, content := range step.Content {
			switch part := content.(type) {
			case fantasy.ReasoningContent:
				metadata, ok := part.ProviderMetadata[openai.Name].(*openai.ResponsesReasoningMetadata)
				require.True(t, ok)
				require.NotEmpty(t, metadata.ItemID)
				require.NotNil(t, metadata.EncryptedContent)
				require.NotEmpty(t, *metadata.EncryptedContent)
				reasoning++
			case fantasy.TextContent:
				metadata, ok := part.ProviderMetadata[openai.Name].(*openai.ResponsesMessageMetadata)
				require.True(t, ok)
				require.NotEmpty(t, metadata.ItemID)
				require.Contains(t, []string{"commentary", "final_answer"}, metadata.Phase)
				messages++
			case fantasy.ToolCallContent:
				metadata, ok := part.ProviderMetadata[openai.Name].(*openai.ResponsesToolCallMetadata)
				require.True(t, ok)
				require.NotEmpty(t, metadata.ItemID)
				require.NotEqual(t, part.ToolCallID, metadata.ItemID)
				calls++
			}
		}
	}
	require.Positive(t, reasoning)
	require.Positive(t, messages)
	require.Equal(t, 1, calls)
	data, err := json.Marshal(history)
	require.NoError(t, err)
	var restored []fantasy.Message
	require.NoError(t, json.Unmarshal(data, &restored))
	result, err = fantasy.NewAgent(model).Stream(t.Context(), fantasy.AgentStreamCall{
		Messages:        restored,
		Prompt:          "Add one more to your previous result. Reply with only the result.",
		ProviderOptions: opts,
		MaxOutputTokens: new(int64(1024)),
		MaxRetries:      new(0),
	})
	require.NoError(t, err)
	require.Contains(t, result.Response.Content.Text(), "43")
	require.Equal(t, fantasy.FinishReasonStop, result.Response.FinishReason)
}
