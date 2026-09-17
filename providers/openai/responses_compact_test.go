package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestResponsesCompact(t *testing.T) {
	t.Parallel()
	checkpointEvent := `{"type":"response.output_item.done","item":` + replayCompaction + `}`
	completed := func(output string) string {
		return `{"type":"response.completed","response":{"id":"resp_compact","status":"completed","output":` + output + `,"usage":{"input_tokens":10,"output_tokens":5,"input_tokens_details":{"cached_tokens":4},"output_tokens_details":{"reasoning_tokens":2}}}}`
	}
	for _, tc := range []struct {
		name      string
		events    []string
		wantError string
	}{
		{name: "item events", events: []string{checkpointEvent, completed(`[]`)}},
		{name: "terminal only", events: []string{completed(`[` + replayCompaction + `]`)}},
		{name: "both", events: []string{checkpointEvent, completed(`[` + replayCompaction + `]`)}},
		{name: "terminal overrides", events: []string{`{"type":"response.output_item.done","item":{"type":"compaction","id":"partial"}}`, completed(`[` + replayCompaction + `]`)}},
		{name: "missing", events: []string{completed(`[]`)}, wantError: "got 0"},
		{name: "empty", events: []string{completed(`[{"type":"compaction","id":"cmp_1"}]`)}, wantError: "no encrypted content"},
		{name: "multiple", events: []string{completed(`[` + replayCompaction + `,` + replayCompaction + `]`)}, wantError: "got 2"},
		{name: "terminal replaces", events: []string{checkpointEvent, completed(`[` + replayMessage + `]`)}, wantError: "got 0"},
		{name: "failed", events: []string{checkpointEvent, `{"type":"response.failed","response":{"error":{"code":"server_error","message":"failed checkpoint"}}}`}, wantError: "failed checkpoint"},
		{name: "incomplete", events: []string{checkpointEvent, `{"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}}`}, wantError: "max_output_tokens"},
		{name: "error", events: []string{`{"type":"error","code":"server_error","message":"stream failure"}`}, wantError: "stream failure"},
		{name: "truncated", events: []string{checkpointEvent}, wantError: "stream"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			requests := make(chan map[string]any, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request map[string]any
				require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
				require.Equal(t, "compact-test", r.Header.Get("X-Test"))
				require.Equal(t, "fantasy-compact-test", r.UserAgent())
				requests <- request
				w.Header().Set("Content-Type", "text/event-stream")
				for _, event := range tc.events {
					fmt.Fprintf(w, "data: %s\n\n", event)
				}
			}))
			defer server.Close()
			call := testCall(fantasy.Prompt{fantasy.NewUserMessage("The stored number is 41.")}, &ResponsesProviderOptions{FullReplay: true, Store: new(false), Instructions: new("Be concise."), PromptCacheKey: new("test-session"), ParallelToolCalls: new(false)})
			call.MaxOutputTokens = new(int64(20000))
			call.ToolChoice = new(fantasy.ToolChoiceRequired)
			call.Tools = []fantasy.Tool{fantasy.FunctionTool{Name: "lookup", Description: "Look up a value", InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}}}
			call.Headers = map[string]string{"X-Test": "compact-test"}
			call.UserAgent = "fantasy-compact-test"
			before, err := json.Marshal(call)
			require.NoError(t, err)
			model := newResponsesProvider(t, server.URL)
			result, err := model.(fantasy.Compactor).Compact(t.Context(), call)
			after, marshalErr := json.Marshal(call)
			require.NoError(t, marshalErr)
			require.Equal(t, string(before), string(after), "compaction must not change the caller's history or options")
			request := <-requests
			require.Equal(t, false, request["store"])
			require.Equal(t, true, request["stream"])
			require.Equal(t, true, request["parallel_tool_calls"])
			require.Equal(t, float64(20000), request["max_output_tokens"])
			require.Equal(t, "Be concise.", request["instructions"])
			require.Equal(t, "test-session", request["prompt_cache_key"])
			require.NotContains(t, request, "tool_choice")
			require.Len(t, request["tools"], 1)
			input := request["input"].([]any)
			require.Len(t, input, 2)
			require.Equal(t, map[string]any{"type": "compaction_trigger"}, input[1])
			if tc.wantError != "" {
				require.ErrorContains(t, err, tc.wantError)
				require.Nil(t, result)
				if tc.name == "truncated" {
					require.ErrorIs(t, err, io.ErrUnexpectedEOF)
				}
				return
			}
			require.NoError(t, err)
			require.Len(t, result.Content, 1)
			require.Equal(t, fantasy.CompactionContent{ProviderMetadata: fantasy.ProviderMetadata{Name: &ResponsesCompactionMetadata{ItemID: "cmp_1", EncryptedContent: "opaque-checkpoint"}}}, result.Content[0])
			require.Equal(t, fantasy.Usage{InputTokens: 6, OutputTokens: 5, TotalTokens: 15, ReasoningTokens: 2, CacheReadTokens: 4}, result.Usage)
			require.Equal(t, fantasy.FinishReasonStop, result.FinishReason)
			require.Equal(t, "resp_compact", result.ProviderMetadata[Name].(*ResponsesProviderMetadata).ResponseID)
		})
	}
}

func TestResponsesCompactCancellation(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(t.Context())
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"type\":\"response.output_item.done\",\"item\":%s}\n\n", replayCompaction)
		w.(http.Flusher).Flush()
		cancel()
	}))
	defer server.Close()
	defer cancel()
	model := newResponsesProvider(t, server.URL)
	result, err := model.(fantasy.Compactor).Compact(ctx, testCall(testPrompt, nil))
	require.ErrorIs(t, err, context.Canceled)
	require.Nil(t, result)
}

func TestResponsesCompactHTTPError(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"type":"invalid_request_error","message":"compaction limit too small","code":"invalid_value"}}`)
	}))
	defer server.Close()
	result, err := newResponsesProvider(t, server.URL).(fantasy.Compactor).Compact(t.Context(), testCall(testPrompt, nil))
	require.Nil(t, result)
	var providerError *fantasy.ProviderError
	require.ErrorAs(t, err, &providerError)
	require.Equal(t, http.StatusBadRequest, providerError.StatusCode)
	require.Contains(t, providerError.Message, "compaction limit too small")
}
