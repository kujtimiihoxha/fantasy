// Package prompt handles content that providers cannot replay.
package prompt

import (
	"slices"

	"charm.land/fantasy"
)

// SkipCompaction removes checkpoints that the provider cannot read.
// It keeps the caller's history unchanged and drops checkpoint-only messages.
func SkipCompaction(input fantasy.Prompt) (fantasy.Prompt, []fantasy.CallWarning) {
	var warnings []fantasy.CallWarning
	var output fantasy.Prompt
	for i, message := range input {
		if !slices.ContainsFunc(message.Content, func(part fantasy.MessagePart) bool { return part.GetType() == fantasy.ContentTypeCompaction }) {
			if output != nil {
				output = append(output, message)
			}
			continue
		}
		if output == nil {
			output = append(make(fantasy.Prompt, 0, len(input)), input[:i]...)
		}
		parts := make([]fantasy.MessagePart, 0, len(message.Content))
		for _, part := range message.Content {
			if part.GetType() == fantasy.ContentTypeCompaction {
				warnings = append(warnings, fantasy.CallWarning{Type: fantasy.CallWarningTypeOther, Message: "skipping compaction checkpoint: this provider cannot replay it"})
			} else {
				parts = append(parts, part)
			}
		}
		if len(parts) > 0 {
			message.Content = parts
			output = append(output, message)
		}
	}
	if output == nil {
		return input, warnings
	}
	return output, warnings
}
