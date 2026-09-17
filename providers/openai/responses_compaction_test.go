package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

const replayCompaction = `{"type":"compaction","id":"cmp_1","encrypted_content":"opaque-checkpoint"}`

func TestResponsesCompactionReplay(t *testing.T) {
	t.Parallel()
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", streaming), func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if !streaming {
					w.Header().Set("Content-Type", "application/json")
					fmt.Fprintf(w, `{"id":"resp_1","status":"completed","output":[%s]}`, replayCompaction)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintf(w, "data: {\"type\":\"response.output_item.done\",\"item\":%s}\n\n", replayCompaction)
				fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\"}}\n\n")
			}))
			defer server.Close()
			agent := fantasy.NewAgent(newResponsesProvider(t, server.URL))
			var result *fantasy.AgentResult
			var err error
			if streaming {
				result, err = agent.Stream(t.Context(), fantasy.AgentStreamCall{Prompt: "Continue"})
			} else {
				result, err = agent.Generate(t.Context(), fantasy.AgentCall{Prompt: "Continue"})
			}
			require.NoError(t, err)
			require.Len(t, result.Response.Content, 1)
			require.Equal(t, fantasy.ContentTypeCompaction, result.Response.Content[0].GetType())
			encoded, err := json.Marshal(result.Response)
			require.NoError(t, err)
			var restored fantasy.Response
			require.NoError(t, json.Unmarshal(encoded, &restored))
			require.Equal(t, result.Response.Content, restored.Content)
			require.NotContains(t, string(encoded), `"text"`)
			encoded, err = json.Marshal(result.Steps[0].Messages)
			require.NoError(t, err)
			var history []fantasy.Message
			require.NoError(t, json.Unmarshal(encoded, &history))
			for _, fullReplay := range []bool{false, true} {
				params, warnings, err := testResponsesLM().prepareParams(testCall(history, &ResponsesProviderOptions{FullReplay: fullReplay}))
				require.NoError(t, err)
				require.Empty(t, warnings)
				require.Len(t, params.Input.OfInputItemList, 1)
				encoded, err := json.Marshal(params.Input.OfInputItemList[0])
				require.NoError(t, err)
				require.JSONEq(t, replayCompaction, string(encoded))
			}
		})
	}
}

func TestResponsesCompactionMissingMetadata(t *testing.T) {
	t.Parallel()
	for _, metadata := range []fantasy.ProviderOptions{nil, {Name: &ResponsesCompactionMetadata{ItemID: "cmp_1"}}} {
		input, warnings := toResponsesPrompt(fantasy.Prompt{{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{fantasy.CompactionPart{ProviderOptions: metadata}}}}, "developer", false, true)
		require.Empty(t, input)
		require.NotEmpty(t, warnings)
		require.Contains(t, warnings[0].Message, "encrypted content")
	}
}
