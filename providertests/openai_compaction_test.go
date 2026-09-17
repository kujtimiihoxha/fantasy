package providertests

import (
	"encoding/json"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"charm.land/x/vcr"
	"github.com/stretchr/testify/require"
)

func TestOpenAIResponsesCompaction(t *testing.T) {
	if _, err := os.Stat("testdata/" + t.Name() + ".yaml"); os.IsNotExist(err) && os.Getenv("FANTASY_OPENAI_API_KEY") == "" {
		t.Skip("set FANTASY_OPENAI_API_KEY to record the compaction test")
	}
	recorder := vcr.NewRecorder(t)
	model, err := openAIReasoningBuilder("gpt-5.6-luna")(t, recorder)
	require.NoError(t, err)
	opts := openai.NewResponsesProviderOptions(&openai.ResponsesProviderOptions{
		FullReplay:      true,
		Store:           new(false),
		ReasoningEffort: openai.ReasoningEffortOption(openai.ReasoningEffortLow),
		PromptCacheKey:  new("fantasy-compaction-test"),
	})
	compactor, ok := model.(fantasy.Compactor)
	require.True(t, ok)
	response, err := compactor.Compact(t.Context(), fantasy.Call{
		Prompt: fantasy.Prompt{
			fantasy.NewUserMessage("Remember this fact for the next turn: the stored number is 41."),
			{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "The stored number is 41."}}},
		},
		ProviderOptions: opts,
		// The compaction trigger requires at least 20,000 tokens when a limit is set.
		MaxOutputTokens: new(int64(20000)),
	})
	require.NoError(t, err)
	require.Len(t, response.Content, 1)
	require.Positive(t, response.Usage.TotalTokens)
	data, err := json.Marshal(response)
	require.NoError(t, err)
	var restored fantasy.Response
	require.NoError(t, json.Unmarshal(data, &restored))
	checkpoint := restored.Content[0].(fantasy.CompactionContent)
	metadata := checkpoint.ProviderMetadata[openai.Name].(*openai.ResponsesCompactionMetadata)
	require.NotEmpty(t, metadata.EncryptedContent)
	require.NotEmpty(t, metadata.ItemID)
	history := []fantasy.Message{{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
		fantasy.CompactionPart{ProviderOptions: fantasy.ProviderOptions(checkpoint.ProviderMetadata)},
	}}}
	data, err = json.Marshal(history)
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(data, &history))
	result, err := fantasy.NewAgent(model).Stream(t.Context(), fantasy.AgentStreamCall{
		Messages:        history,
		Prompt:          "Add one to the stored number. Reply with only the result.",
		ProviderOptions: opts,
		MaxOutputTokens: new(int64(1024)),
		MaxRetries:      new(0),
	})
	require.NoError(t, err)
	require.Equal(t, fantasy.FinishReasonStop, result.Response.FinishReason)
	require.Contains(t, result.Response.Content.Text(), "42")
}
