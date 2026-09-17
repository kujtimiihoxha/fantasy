package providertests

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/codex"
	"charm.land/fantasy/providers/openai"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

// TestCodexLive is opt-in and never records account credentials or responses.
func TestCodexLive(t *testing.T) {
	path := os.Getenv("FANTASY_CODEX_TOKEN_FILE")
	if path == "" {
		t.Skip("set FANTASY_CODEX_TOKEN_FILE to a JSON file with access_token and account_id")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	provider, err := codex.New(codex.Config{
		SessionID:  uuid.NewString(),
		Originator: "fantasy",
		Credentials: func(context.Context) (codex.Credentials, error) {
			data, err := os.ReadFile(path)
			if err != nil {
				return codex.Credentials{}, err
			}
			var stored struct {
				AccessToken string `json:"access_token"`
				AccountID   string `json:"account_id"`
			}
			if err := json.Unmarshal(data, &stored); err != nil {
				return codex.Credentials{}, err
			}
			return codex.Credentials{AccessToken: stored.AccessToken, AccountID: stored.AccountID}, nil
		},
	})
	require.NoError(t, err)
	model, err := provider.LanguageModel(ctx, "gpt-5.6-luna")
	require.NoError(t, err)
	opts := openai.NewResponsesProviderOptions(&openai.ResponsesProviderOptions{ReasoningEffort: openai.ReasoningEffortOption(openai.ReasoningEffortHigh)})
	tool := fantasy.NewAgentTool("lookup_number", "Return the stored number", func(context.Context, struct{}, fantasy.ToolCall) (fantasy.ToolResponse, error) {
		return fantasy.NewTextResponse("The stored number is 41."), nil
	})
	agent := fantasy.NewAgent(model, fantasy.WithTools(tool), fantasy.WithStopConditions(fantasy.StepCountIs(2)))
	const prompt = "Call lookup_number once. Add one to the stored number. Reply with only the result."
	generated, err := agent.Generate(ctx, fantasy.AgentCall{Prompt: prompt, ProviderOptions: opts, MaxRetries: new(0)})
	require.NoError(t, err)
	require.Len(t, generated.Steps, 2)
	require.Contains(t, generated.Response.Content.Text(), "42")
	history := []fantasy.Message{fantasy.NewUserMessage(prompt)}
	var encrypted, calls, messages int
	for _, step := range generated.Steps {
		history = append(history, step.Messages...)
		for _, content := range step.Content {
			switch part := content.(type) {
			case fantasy.ReasoningContent:
				metadata := part.ProviderMetadata[openai.Name].(*openai.ResponsesReasoningMetadata)
				if metadata.EncryptedContent != nil && *metadata.EncryptedContent != "" {
					encrypted++
				}
			case fantasy.ToolCallContent:
				metadata := part.ProviderMetadata[openai.Name].(*openai.ResponsesToolCallMetadata)
				require.NotEmpty(t, metadata.ItemID)
				calls++
			case fantasy.TextContent:
				metadata := part.ProviderMetadata[openai.Name].(*openai.ResponsesMessageMetadata)
				require.NotEmpty(t, metadata.ItemID)
				require.NotEmpty(t, metadata.Phase)
				messages++
			}
		}
	}
	require.Positive(t, encrypted)
	require.Equal(t, 1, calls)
	require.Positive(t, messages)
	data, err := json.Marshal(history)
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(data, &history))
	result, err := fantasy.NewAgent(model).Stream(ctx, fantasy.AgentStreamCall{Messages: history, Prompt: "Add one more to your previous result. Reply with only the result.", ProviderOptions: opts, MaxRetries: new(0)})
	require.NoError(t, err)
	require.Contains(t, result.Response.Content.Text(), "43")
	t.Log("Generate, tool replay, encrypted reasoning, phase, and Stream passed")

	compacted, err := model.(fantasy.Compactor).Compact(ctx, fantasy.Call{Prompt: history, ProviderOptions: opts})
	require.NoError(t, err)
	require.Len(t, compacted.Content, 1)
	data, err = json.Marshal(compacted)
	require.NoError(t, err)
	var restored fantasy.Response
	require.NoError(t, json.Unmarshal(data, &restored))
	checkpoint := restored.Content[0].(fantasy.CompactionContent)
	result, err = fantasy.NewAgent(model).Stream(ctx, fantasy.AgentStreamCall{
		Messages: []fantasy.Message{{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{fantasy.CompactionPart{ProviderOptions: fantasy.ProviderOptions(checkpoint.ProviderMetadata)}}}},
		Prompt:   "Add one to your previous result. Reply with only the result.", ProviderOptions: opts, MaxRetries: new(0),
	})
	require.NoError(t, err)
	require.Contains(t, result.Response.Content.Text(), "43")
	t.Log("Compact and checkpoint replay passed")

	call := fantasy.ObjectCall{
		Prompt: fantasy.Prompt{fantasy.NewUserMessage("Return the number 42 in the value field.")},
		Schema: fantasy.Schema{Type: "object", Properties: map[string]*fantasy.Schema{"value": {Type: "integer"}}, Required: []string{"value"}},
	}
	object, err := model.GenerateObject(ctx, call)
	require.NoError(t, err)
	require.JSONEq(t, `{"value":42}`, object.RawText)
	stream, err := model.StreamObject(ctx, call)
	require.NoError(t, err)
	var last any
	finished := false
	for part := range stream {
		require.NoError(t, part.Error)
		if part.Type == fantasy.ObjectStreamPartTypeObject {
			last = part.Object
		}
		if part.Type == fantasy.ObjectStreamPartTypeFinish {
			finished = true
		}
	}
	require.True(t, finished)
	data, err = json.Marshal(last)
	require.NoError(t, err)
	require.JSONEq(t, `{"value":42}`, string(data))
	t.Log("GenerateObject and StreamObject passed")
}
