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

func TestResponsesObjectRetainsAllTextAndFailureUsage(t *testing.T) {
	t.Parallel()
	for name, test := range map[string]struct {
		parts  []map[string]string
		want   string
		failed bool
	}{
		"split text":          {parts: []map[string]string{{"type": "output_text", "text": `{"value":`}, {"type": "output_text", "text": `42}`}}, want: `{"value":42}`},
		"refusal after text":  {parts: []map[string]string{{"type": "output_text", "text": `{"value":42}`}, {"type": "refusal", "refusal": "No."}}, failed: true},
		"refusal before text": {parts: []map[string]string{{"type": "refusal", "refusal": "No."}, {"type": "output_text", "text": `{"value":42}`}}, failed: true},
		"invalid object":      {parts: []map[string]string{{"type": "output_text", "text": `{"value":"wrong"}`}}, failed: true},
		"no text":             {failed: true},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request struct {
					Text struct{ Format struct{ Strict bool } }
				}
				require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
				require.True(t, request.Text.Format.Strict)
				w.Header().Set("Content-Type", "application/json")
				content, err := json.Marshal(test.parts)
				require.NoError(t, err)
				fmt.Fprintf(w, `{"id":"response","status":"completed","output":[{"type":"message","id":"message","role":"assistant","content":%s}],"usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14,"input_tokens_details":{"cached_tokens":3},"output_tokens_details":{"reasoning_tokens":2}}}`, content)
			}))
			defer server.Close()
			provider, err := New(WithAPIKey("test"), WithBaseURL(server.URL), WithUseResponsesAPI())
			require.NoError(t, err)
			model, err := provider.LanguageModel(t.Context(), "gpt-5.6-luna")
			require.NoError(t, err)
			result, err := model.GenerateObject(t.Context(), fantasy.ObjectCall{ProviderOptions: NewResponsesProviderOptions(&ResponsesProviderOptions{StrictJSONSchema: new(true)}), Schema: fantasy.Schema{
				Type: "object", Properties: map[string]*fantasy.Schema{"value": {Type: "integer"}}, Required: []string{"value"},
			}})
			wantUsage := fantasy.Usage{InputTokens: 7, OutputTokens: 4, TotalTokens: 14, CacheReadTokens: 3, ReasoningTokens: 2}
			if test.failed {
				require.Nil(t, result)
				var objectErr *fantasy.NoObjectGeneratedError
				require.ErrorAs(t, err, &objectErr)
				require.Equal(t, wantUsage, objectErr.Usage)
			} else {
				require.NoError(t, err)
				require.Equal(t, test.want, result.RawText)
				require.Equal(t, wantUsage, result.Usage)
			}
		})
	}
}

func TestResponsesObjectStreamRejectsMixedRefusal(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Text struct{ Format struct{ Strict bool } }
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		require.True(t, request.Text.Format.Strict)
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range []string{
			`{"type":"response.output_text.delta","delta":"{\"value\":42}"}`,
			`{"type":"response.completed","response":{"id":"response","status":"completed","output":[{"type":"message","content":[{"type":"refusal","refusal":"No."}]}],"usage":{"input_tokens":10,"output_tokens":4}}}`,
		} {
			fmt.Fprintf(w, "data: %s\n\n", event)
		}
	}))
	defer server.Close()
	provider, err := New(WithAPIKey("test"), WithBaseURL(server.URL), WithUseResponsesAPI())
	require.NoError(t, err)
	model, err := provider.LanguageModel(t.Context(), "gpt-5.6-luna")
	require.NoError(t, err)
	stream, err := model.StreamObject(t.Context(), fantasy.ObjectCall{ProviderOptions: NewResponsesProviderOptions(&ResponsesProviderOptions{StrictJSONSchema: new(true)}), Schema: fantasy.Schema{
		Type: "object", Properties: map[string]*fantasy.Schema{"value": {Type: "integer"}}, Required: []string{"value"},
	}})
	require.NoError(t, err)
	var objectErr *fantasy.NoObjectGeneratedError
	for part := range stream {
		require.NotEqual(t, fantasy.ObjectStreamPartTypeFinish, part.Type)
		if part.Type == fantasy.ObjectStreamPartTypeError {
			require.ErrorAs(t, part.Error, &objectErr)
		}
	}
	require.NotNil(t, objectErr)
	require.Equal(t, int64(14), objectErr.Usage.TotalTokens)
	require.Equal(t, fantasy.FinishReasonContentFilter, objectErr.FinishReason)
}
