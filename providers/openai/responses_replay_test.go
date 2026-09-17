package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"charm.land/fantasy"
	"github.com/openai/openai-go/v3/responses"
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
					fmt.Fprintf(w, `{"id":"resp_1","status":"completed","output":[%s,%s,%s]}`, replayReasoning, replayToolCall, replayMessage)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				for _, event := range []string{
					`{"type":"response.output_item.added","item":{"type":"reasoning","id":"rs_1"}}`,
					`{"type":"response.reasoning_summary_part.added","item_id":"rs_1","summary_index":0,"part":{"type":"summary_text","text":""}}`,
					`{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","summary_index":0,"delta":"Partial."}`,
					`{"type":"response.output_item.done","item":` + replayReasoning + `}`,
					`{"type":"response.output_item.added","output_index":1,"item":` + replayToolCall + `}`,
					`{"type":"response.output_item.done","output_index":1,"item":` + replayToolCall + `}`,
					`{"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}`,
					`{"type":"response.output_text.delta","item_id":"msg_1","delta":"Checking."}`,
					`{"type":"response.output_item.done","item":` + replayMessage + `}`,
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
			for i, expected := range []string{replayReasoning, replayToolCall, replayMessage} {
				data, err := json.Marshal(items[i])
				require.NoError(t, err)
				require.JSONEq(t, expected, string(data))
			}

			params, warnings, err = testResponsesLM().prepareParams(testCall(history, nil))
			require.NoError(t, err)
			require.Empty(t, warnings)
			items = params.Input.OfInputItemList
			require.Len(t, items, 3)
			require.False(t, items[0].OfFunctionCall.ID.Valid())
			require.NotNil(t, items[1].OfMessage)
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

func TestResponsesFullReplayMessageContent(t *testing.T) {
	t.Parallel()
	for _, content := range []string{
		`[{"type":"refusal","refusal":"Cannot do that."}]`,
		`[{"type":"output_text","text":"First."},{"type":"refusal","refusal":"Cannot do that."},{"type":"output_text","text":"Last."}]`,
		`[{"type":"output_text","text":""},{"type":"refusal","refusal":""}]`,
	} {
		for _, mode := range []string{"generate", "stream", "stream without deltas"} {
			t.Run(mode+"/"+content, func(t *testing.T) {
				t.Parallel()
				message := `{"type":"message","id":"msg_1","role":"assistant","phase":"final_answer","content":` + content + `}`
				var item responses.ResponseOutputItemUnion
				require.NoError(t, json.Unmarshal([]byte(message), &item))
				var wantText strings.Builder
				for _, part := range item.Content {
					wantText.WriteString(part.Text + part.Refusal)
				}
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					if mode == "generate" {
						w.Header().Set("Content-Type", "application/json")
						fmt.Fprintf(w, `{"id":"resp_1","status":"completed","output":[%s]}`, message)
						return
					}
					w.Header().Set("Content-Type", "text/event-stream")
					fmt.Fprint(w, "data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"msg_1\",\"type\":\"message\"}}\n\n")
					if mode == "stream" {
						for i, part := range item.Content {
							text := part.Text + part.Refusal
							for _, delta := range []string{text[:len(text)/2], text[len(text)/2:]} {
								encoded, err := json.Marshal(delta)
								if err != nil {
									t.Error(err)
									return
								}
								fmt.Fprintf(w, "data: {\"type\":\"response.%s.delta\",\"item_id\":\"msg_1\",\"content_index\":%d,\"delta\":%s}\n\n", part.Type, i, encoded)
							}
						}
					}
					fmt.Fprintf(w, "data: {\"type\":\"response.output_item.done\",\"item\":%s}\n\n", message)
					fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\"}}\n\n")
				}))
				defer server.Close()
				provider, err := New(WithAPIKey("test-key"), WithBaseURL(server.URL), WithUseResponsesAPI())
				require.NoError(t, err)
				model, err := provider.LanguageModel(t.Context(), "gpt-5.6-luna")
				require.NoError(t, err)
				opts := &ResponsesProviderOptions{FullReplay: true}
				agent := fantasy.NewAgent(model, fantasy.WithProviderOptions(NewResponsesProviderOptions(opts)))
				var result *fantasy.AgentResult
				if mode == "generate" {
					result, err = agent.Generate(t.Context(), fantasy.AgentCall{Prompt: "Check"})
				} else {
					var streamed strings.Builder
					result, err = agent.Stream(t.Context(), fantasy.AgentStreamCall{
						Prompt:      "Check",
						OnTextDelta: func(_, delta string) error { streamed.WriteString(delta); return nil },
					})
					require.Equal(t, wantText.String(), streamed.String())
				}
				require.NoError(t, err)
				require.Len(t, result.Steps, 1)
				require.Len(t, result.Steps[0].Content, len(item.Content))
				data, err := json.Marshal(result.Steps[0].Messages)
				require.NoError(t, err)
				var history []fantasy.Message
				require.NoError(t, json.Unmarshal(data, &history))
				for _, followup := range []bool{false, true} {
					if followup {
						history = append(history, fantasy.NewUserMessage("Continue"))
					}
					params, warnings, err := testResponsesLM().prepareParams(testCall(history, opts))
					require.NoError(t, err)
					require.Empty(t, warnings)
					data, err := json.Marshal(params.Input.OfInputItemList[0])
					require.NoError(t, err)
					require.JSONEq(t, message, string(data), "the replay prefix must preserve all message content")
				}
				part := history[0].Content[0].(fantasy.TextPart)
				part.Text = "Edited."
				history[0].Content[0] = part
				params, _, err := testResponsesLM().prepareParams(testCall(history, opts))
				require.NoError(t, err)
				replayed := params.Input.OfInputItemList[0].OfOutputMessage.Content[0]
				if item.Content[0].Type == "refusal" {
					require.Equal(t, "Edited.", replayed.OfRefusal.Refusal)
				} else {
					require.Equal(t, "Edited.", replayed.OfOutputText.Text)
				}
			})
		}
	}
}

func TestResponsesFullReplayTruncatedToolCalls(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[%s,%s]}`, replayToolCall, replayMessage)
	}))
	defer server.Close()
	model := newResponsesProvider(t, server.URL)
	result, err := model.Generate(t.Context(), testCall(testPrompt, &ResponsesProviderOptions{FullReplay: true}))
	require.NoError(t, err)
	require.Equal(t, fantasy.FinishReasonLength, result.FinishReason)
	require.NotEmpty(t, result.Warnings)
	require.Len(t, result.Content, 1)
	require.Equal(t, "Checking.", result.Content.Text())
}
