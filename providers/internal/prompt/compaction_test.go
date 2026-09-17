package prompt

import (
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestSkipCompaction(t *testing.T) {
	t.Parallel()
	for _, role := range []fantasy.MessageRole{fantasy.MessageRoleSystem, fantasy.MessageRoleUser, fantasy.MessageRoleAssistant, fantasy.MessageRoleTool} {
		t.Run(string(role), func(t *testing.T) {
			before := fantasy.NewUserMessage("Before")
			after := fantasy.NewUserMessage("After")
			text := fantasy.TextPart{Text: "Keep"}
			input := fantasy.Prompt{before, {Role: role, Content: []fantasy.MessagePart{fantasy.CompactionPart{}, text}}, {Role: role, Content: []fantasy.MessagePart{&fantasy.CompactionPart{}}}, after}
			got, warnings := SkipCompaction(input)
			require.Equal(t, fantasy.Prompt{before, {Role: role, Content: []fantasy.MessagePart{text}}, after}, got)
			require.Len(t, warnings, 2)
			require.Len(t, input, 4)
			require.Len(t, input[1].Content, 2)
			require.Equal(t, fantasy.ContentTypeCompaction, input[1].Content[0].GetType())
		})
	}
	input := fantasy.Prompt{fantasy.NewUserMessage("Keep")}
	got, warnings := SkipCompaction(input)
	require.Equal(t, input, got)
	require.Empty(t, warnings)
	got, warnings = SkipCompaction(fantasy.Prompt{{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{fantasy.CompactionPart{}}}})
	require.Empty(t, got)
	require.Len(t, warnings, 1)
}
