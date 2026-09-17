package codex

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	checkpoint = `{"type":"compaction","id":"cmp_1","encrypted_content":"opaque-checkpoint"}`
	reasoning  = `{"type":"reasoning","id":"rs_1","encrypted_content":"opaque-reasoning","summary":[]}`
	message    = `{"type":"message","id":"msg_1","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"{\"value\":42}"}]}`
	completed  = `{"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":10,"output_tokens":5,"input_tokens_details":{"cached_tokens":3}}}}`
)

func credentials(context.Context) (Credentials, error) {
	return Credentials{AccessToken: "test-token", AccountID: "test-account"}, nil
}

func newModel(t *testing.T, serverURL string, change func(*Config)) fantasy.LanguageModel {
	t.Helper()
	config := Config{Credentials: credentials, SessionID: "session-1", BaseURL: serverURL}
	if change != nil {
		change(&config)
	}
	provider, err := New(config)
	require.NoError(t, err)
	require.Equal(t, Name, provider.Name())
	model, err := provider.LanguageModel(t.Context(), "gpt-5.6-luna")
	require.NoError(t, err)
	require.Equal(t, Name, model.Provider())
	return model
}

func emit(w http.ResponseWriter, events ...string) {
	w.Header().Set("Content-Type", "text/event-stream")
	for _, event := range events {
		fmt.Fprintf(w, "data: %s\n\n", event)
	}
}

func emitAnswer(w http.ResponseWriter) {
	emit(w,
		`{"type":"response.output_item.added","item":{"type":"reasoning","id":"rs_1"}}`,
		`{"type":"response.output_item.done","item":`+reasoning+`}`,
		`{"type":"response.output_text.delta","item_id":"msg_1","delta":"{\"value\":42}"}`,
		`{"type":"response.output_item.done","item":`+message+`}`, completed)
}

func TestProviderCalls(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "must-not-send")
	t.Setenv("OPENAI_ORG_ID", "must-not-send")
	t.Setenv("OPENAI_PROJECT_ID", "must-not-send")
	for _, method := range []string{"generate", "stream", "generate-object", "stream-object", "compact"} {
		t.Run(method, func(t *testing.T) {
			requests := make(chan map[string]json.RawMessage, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/responses", r.URL.Path)
				assert.Equal(t, "Bearer test-token", r.Header.Get("Authorization"))
				assert.Equal(t, "test-account", r.Header.Get("Chatgpt-Account-Id"))
				assert.Equal(t, "session-1", r.Header.Get("Session-Id"))
				assert.Equal(t, "fantasy", r.Header.Get("Originator"))
				assert.Empty(t, r.Header.Get("Openai-Organization"))
				assert.Empty(t, r.Header.Get("Openai-Project"))
				var body map[string]json.RawMessage
				assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
				requests <- body
				if method == "compact" {
					emit(w, `{"type":"response.output_item.done","item":`+checkpoint+`}`, completed)
				} else {
					emitAnswer(w)
				}
			}))
			defer server.Close()
			model := newModel(t, server.URL, nil)
			opts := openai.NewResponsesProviderOptions(&openai.ResponsesProviderOptions{Store: new(true), FullReplay: false, ReasoningEffort: openai.ReasoningEffortOption(openai.ReasoningEffortLow)})
			before, err := json.Marshal(opts)
			require.NoError(t, err)
			call := fantasy.Call{Prompt: fantasy.Prompt{fantasy.NewUserMessage("Continue")}, ProviderOptions: opts, MaxOutputTokens: new(int64(1024))}
			objectCall := fantasy.ObjectCall{Prompt: call.Prompt, ProviderOptions: opts, MaxOutputTokens: call.MaxOutputTokens, Schema: fantasy.Schema{Type: "object", Properties: map[string]*fantasy.Schema{"value": {Type: "integer"}}, Required: []string{"value"}}}
			switch method {
			case "generate":
				result, err := model.Generate(t.Context(), call)
				require.NoError(t, err)
				require.Len(t, result.Content, 2)
				require.Equal(t, fantasy.Usage{InputTokens: 7, OutputTokens: 5, TotalTokens: 15, CacheReadTokens: 3}, result.Usage)
				meta := result.Content[1].(fantasy.TextContent).ProviderMetadata[openai.Name].(*openai.ResponsesMessageMetadata)
				require.Equal(t, "msg_1", meta.ItemID)
				require.Equal(t, "final_answer", meta.Phase)
			case "stream":
				agent := fantasy.NewAgent(model)
				result, err := agent.Stream(t.Context(), fantasy.AgentStreamCall{Prompt: "Continue", ProviderOptions: opts, MaxOutputTokens: call.MaxOutputTokens})
				require.NoError(t, err)
				require.Equal(t, `{"value":42}`, result.Response.Content.Text())
				require.Len(t, result.Steps[0].Messages[0].Content, 2)
				history, err := json.Marshal(result.Steps[0].Messages)
				require.NoError(t, err)
				var restored []fantasy.Message
				require.NoError(t, json.Unmarshal(history, &restored))
				require.Equal(t, result.Steps[0].Messages, restored)
			case "generate-object":
				result, err := model.GenerateObject(t.Context(), objectCall)
				require.NoError(t, err)
				require.JSONEq(t, `{"value":42}`, result.RawText)
			case "stream-object":
				stream, err := model.StreamObject(t.Context(), objectCall)
				require.NoError(t, err)
				finished := false
				for part := range stream {
					require.NoError(t, part.Error)
					if part.Type == fantasy.ObjectStreamPartTypeFinish {
						finished = true
					}
				}
				require.True(t, finished)
			case "compact":
				result, err := model.(fantasy.Compactor).Compact(t.Context(), call)
				require.NoError(t, err)
				require.Equal(t, fantasy.ContentTypeCompaction, result.Content[0].GetType())
			}
			after, err := json.Marshal(opts)
			require.NoError(t, err)
			require.Equal(t, string(before), string(after))
			body := <-requests
			require.JSONEq(t, `true`, string(body["stream"]))
			require.JSONEq(t, `false`, string(body["store"]))
			require.JSONEq(t, `""`, string(body["instructions"]))
			require.JSONEq(t, `"session-1"`, string(body["prompt_cache_key"]))
			require.Contains(t, string(body["include"]), "reasoning.encrypted_content")
			require.NotContains(t, body, "max_output_tokens")
		})
	}
}

func TestRefresh(t *testing.T) {
	t.Parallel()
	for _, rejected := range []bool{false, true} {
		t.Run(fmt.Sprint(rejected), func(t *testing.T) {
			var requests, refreshes atomic.Int32
			bodies := make(chan string, 2)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				data, _ := io.ReadAll(r.Body)
				bodies <- string(data)
				if requests.Add(1) == 1 || rejected {
					w.WriteHeader(http.StatusUnauthorized)
					fmt.Fprint(w, `{"detail":"expired"}`)
					return
				}
				assert.Equal(t, "Bearer new-token", r.Header.Get("Authorization"))
				emitAnswer(w)
			}))
			defer server.Close()
			model := newModel(t, server.URL, func(c *Config) {
				c.Refresh = func(ctx context.Context, token string) (Credentials, error) {
					require.NoError(t, ctx.Err())
					assert.Equal(t, "test-token", token)
					refreshes.Add(1)
					return Credentials{AccessToken: "new-token", AccountID: "test-account"}, nil
				}
			})
			result, err := model.Generate(t.Context(), fantasy.Call{Prompt: fantasy.Prompt{fantasy.NewUserMessage("Continue")}})
			if rejected {
				require.ErrorContains(t, err, "expired")
				require.Nil(t, result)
			} else {
				require.NoError(t, err)
			}
			require.EqualValues(t, 1, refreshes.Load())
			require.EqualValues(t, 2, requests.Load())
			require.Equal(t, <-bodies, <-bodies)
		})
	}
}

func TestStreamCloses(t *testing.T) {
	t.Parallel()
	for _, early := range []bool{false, true} {
		t.Run(fmt.Sprint(early), func(t *testing.T) {
			closed := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				emitAnswer(w)
				w.(http.Flusher).Flush()
				<-r.Context().Done()
				close(closed)
			}))
			defer server.Close()
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			stream, err := newModel(t, server.URL, nil).Stream(ctx, fantasy.Call{Prompt: fantasy.Prompt{fantasy.NewUserMessage("Continue")}})
			require.NoError(t, err)
			for part := range stream {
				require.NoError(t, part.Error)
				if early {
					break
				}
			}
			select {
			case <-closed:
			case <-ctx.Done():
				t.Fatal("stream did not close")
			}
			require.NoError(t, ctx.Err())
		})
	}
}

func TestValidationAndRouting(t *testing.T) {
	t.Parallel()
	for _, endpoint := range []string{"http://example.com", "https://user:pass@example.com", "https://example.com?key=value", "https://example.com#fragment"} {
		_, err := New(Config{Credentials: credentials, SessionID: "session", BaseURL: endpoint})
		require.Error(t, err)
	}
	for _, value := range []string{"", " leading", "new\nline"} {
		_, err := New(Config{Credentials: credentials, SessionID: value})
		require.Error(t, err)
	}
	payload := base64.RawURLEncoding.EncodeToString([]byte(`{"https://api.openai.com/auth":{"chatgpt_compute_residency":"eu"}}`))
	require.Equal(t, "eu", residency("header."+payload+".signature"))
	var leaked atomic.Bool
	target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { leaked.Store(true) }))
	defer target.Close()
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer redirect.Close()
	_, err := newModel(t, redirect.URL, nil).Generate(t.Context(), fantasy.Call{})
	require.Error(t, err)
	require.False(t, leaked.Load())
}

func TestGenerateTruncatedStream(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		emit(w, `{"type":"response.output_item.done","item":`+message+`}`)
	}))
	defer server.Close()
	result, err := newModel(t, server.URL, nil).Generate(t.Context(), fantasy.Call{})
	require.Nil(t, result)
	require.ErrorIs(t, err, io.ErrUnexpectedEOF)
}

func TestReplayRequest(t *testing.T) {
	t.Parallel()
	requests := make(chan map[string]json.RawMessage, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		requests <- body
		emitAnswer(w)
	}))
	defer server.Close()
	model := newModel(t, server.URL, nil)
	history := fantasy.Prompt{{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
		fantasy.CompactionPart{ProviderOptions: fantasy.ProviderOptions{openai.Name: &openai.ResponsesCompactionMetadata{ItemID: "cmp_1", EncryptedContent: "opaque-checkpoint"}}},
		fantasy.ReasoningPart{ProviderOptions: fantasy.ProviderOptions{openai.Name: &openai.ResponsesReasoningMetadata{ItemID: "rs_1", EncryptedContent: new("opaque-reasoning")}}},
		fantasy.TextPart{Text: `{"value":42}`, ProviderOptions: fantasy.ProviderOptions{openai.Name: &openai.ResponsesMessageMetadata{ItemID: "msg_1", Phase: "final_answer"}}},
	}}, fantasy.NewUserMessage("Continue")}
	_, err := model.Generate(t.Context(), fantasy.Call{Prompt: history})
	require.NoError(t, err)
	body := <-requests
	var input []json.RawMessage
	require.NoError(t, json.Unmarshal(body["input"], &input))
	require.Len(t, input, 4)
	for i, expected := range []string{checkpoint, reasoning, message} {
		require.JSONEq(t, expected, string(input[i]))
	}
	require.False(t, strings.Contains(string(body["input"]), "compaction_trigger"))
}
