package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

const replayReasoning = `{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Checking."},{"type":"summary_text","text":"Ready."}],"content":[{"type":"reasoning_text","text":"Reasoning content."}],"encrypted_content":"opaque-reasoning"}`

const replayMessage = `{"type":"message","id":"msg_1","role":"assistant","phase":"commentary","content":[{"type":"output_text","text":"Checking."}]}`

const replayToolCall = `{"type":"function_call","id":"fc_1","call_id":"call_1","name":"check","arguments":"{}"}`

// These events cover native Codex history replay.
func TestResponsesFullReplayCapture(t *testing.T) {
	t.Parallel()
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", streaming), func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if !streaming {
					w.Header().Set("Content-Type", "application/json")
					fmt.Fprintf(w, `{"id":"resp_1","status":"completed","output":[%s,%s,%s]}`, replayReasoning, replayMessage, replayToolCall)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				for _, event := range []string{
					`{"type":"response.output_item.added","item":{"type":"reasoning","id":"rs_1"}}`,
					`{"type":"response.reasoning_summary_part.added","item_id":"rs_1","summary_index":0,"part":{"type":"summary_text","text":""}}`,
					`{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","summary_index":0,"delta":"Partial."}`,
					`{"type":"response.output_item.done","item":` + replayReasoning + `}`,
					`{"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}`,
					`{"type":"response.output_text.delta","item_id":"msg_1","delta":"Checking."}`,
					`{"type":"response.output_item.done","item":` + replayMessage + `}`,
					`{"type":"response.output_item.added","output_index":2,"item":` + replayToolCall + `}`,
					`{"type":"response.output_item.done","output_index":2,"item":` + replayToolCall + `}`,
					`{"type":"response.completed","response":{"id":"resp_1","status":"completed"}}`,
				} {
					fmt.Fprintf(w, "data: %s\n\n", event)
				}
			}))
			defer server.Close()
			provider, err := New(WithAPIKey("test-key"), WithBaseURL(server.URL), WithUseResponsesAPI())
			require.NoError(t, err)
			model, err := provider.LanguageModel(t.Context(), "gpt-5.4")
			require.NoError(t, err)
			tool := fantasy.NewAgentTool("check", "Check work", func(context.Context, struct{}, fantasy.ToolCall) (fantasy.ToolResponse, error) {
				return fantasy.NewTextResponse("checked"), nil
			})
			agent := fantasy.NewAgent(model, fantasy.WithTools(tool), fantasy.WithStopConditions(fantasy.StepCountIs(1)))
			opts := NewResponsesProviderOptions(&ResponsesProviderOptions{FullReplay: true})
			var result *fantasy.AgentResult
			if streaming {
				result, err = agent.Stream(t.Context(), fantasy.AgentStreamCall{Prompt: "Check", ProviderOptions: opts})
			} else {
				result, err = agent.Generate(t.Context(), fantasy.AgentCall{Prompt: "Check", ProviderOptions: opts})
			}
			require.NoError(t, err)
			require.Len(t, result.Steps, 1)
			require.Equal(t, fantasy.FinishReasonToolCalls, result.Steps[0].FinishReason)

			// Persist the messages before replay to exercise the metadata registry.
			data, err := json.Marshal(result.Steps[0].Messages)
			require.NoError(t, err)
			var history []fantasy.Message
			require.NoError(t, json.Unmarshal(data, &history))
			params, warnings, err := testResponsesLM().prepareParams(testCall(history, &ResponsesProviderOptions{FullReplay: true}))
			require.NoError(t, err)
			require.Empty(t, warnings)
			require.False(t, params.Store.Value)
			items := params.Input.OfInputItemList
			require.Len(t, items, 4)
			for i, expected := range []string{replayReasoning, replayMessage, replayToolCall} {
				data, err := json.Marshal(items[i])
				require.NoError(t, err)
				require.JSONEq(t, expected, string(data))
			}

			params, warnings, err = testResponsesLM().prepareParams(testCall(history, nil))
			require.NoError(t, err)
			require.Empty(t, warnings)
			items = params.Input.OfInputItemList
			require.Len(t, items, 3)
			require.NotNil(t, items[0].OfMessage)
			require.False(t, items[1].OfFunctionCall.ID.Valid())
		})
	}
}

func TestResponsesFullReplayPrompt(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		parts   []fantasy.MessagePart
		want    string
		warning bool
	}{
		{
			name: "reasoning without visible text",
			parts: []fantasy.MessagePart{fantasy.ReasoningPart{ProviderOptions: fantasy.ProviderOptions{
				Name: &ResponsesReasoningMetadata{ItemID: "rs_1", EncryptedContent: new("opaque")},
			}}},
			want: `[{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque"}]`,
		},
		{
			name:  "text without metadata",
			parts: []fantasy.MessagePart{fantasy.TextPart{Text: "Hello"}},
			want:  `[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello"}]}]`,
		},
		{
			name: "multiple text parts in one item",
			parts: []fantasy.MessagePart{
				fantasy.TextPart{Text: "First", ProviderOptions: fantasy.ProviderOptions{Name: &ResponsesMessageMetadata{ItemID: "msg_1", Phase: "final_answer"}}},
				fantasy.TextPart{Text: "Last", ProviderOptions: fantasy.ProviderOptions{Name: &ResponsesMessageMetadata{ItemID: "msg_1", Phase: "final_answer"}}},
			},
			want: `[{"type":"message","role":"assistant","id":"msg_1","phase":"final_answer","content":[{"type":"output_text","text":"First"},{"type":"output_text","text":"Last"}]}]`,
		},
		{
			name: "reasoning from another provider",
			parts: []fantasy.MessagePart{fantasy.ReasoningPart{ProviderOptions: fantasy.ProviderOptions{
				"other": &ResponsesReasoningMetadata{ItemID: "rs_1"},
			}}},
			want: "null", warning: true,
		},
		{
			name: "reasoning without an item ID",
			parts: []fantasy.MessagePart{fantasy.ReasoningPart{ProviderOptions: fantasy.ProviderOptions{
				Name: &ResponsesReasoningMetadata{},
			}}},
			want: "null", warning: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			input, warnings := toResponsesPrompt(fantasy.Prompt{{Role: fantasy.MessageRoleAssistant, Content: tt.parts}}, "system", false, true)
			data, err := json.Marshal(input)
			require.NoError(t, err)
			require.JSONEq(t, tt.want, string(data))
			if tt.warning {
				require.NotEmpty(t, warnings)
			} else {
				require.Empty(t, warnings)
			}
		})
	}
}
