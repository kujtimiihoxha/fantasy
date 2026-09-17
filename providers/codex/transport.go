package codex

import (
	"bytes"
	"cmp"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"unicode"

	"charm.land/fantasy"
	"github.com/openai/openai-go/v3/packages/ssestream"
)

type transport struct {
	config   Config
	endpoint *url.URL
	next     http.RoundTripper
}

func (t *transport) RoundTrip(request *http.Request) (*http.Response, error) {
	if request.Body == nil {
		return nil, errors.New("codex request body is required")
	}
	defer func() { _ = request.Body.Close() }()
	if request.Method != http.MethodPost || request.URL.Scheme != t.endpoint.Scheme || !strings.EqualFold(request.URL.Host, t.endpoint.Host) || request.URL.EscapedPath() != t.endpoint.EscapedPath() || request.URL.RawQuery != "" {
		return nil, errors.New("refusing to send codex credentials to an unexpected endpoint")
	}
	var body map[string]json.RawMessage
	if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
		return nil, err
	}
	if body == nil {
		return nil, errors.New("codex request must be a JSON object")
	}
	streaming := string(body["stream"]) == "true"
	body["stream"] = json.RawMessage("true")
	// Codex rejects output token limits, including limits set by the agent defaults.
	delete(body, "max_output_tokens")
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	credentials, err := t.config.Credentials(request.Context())
	if err != nil {
		return nil, err
	}
	response, err := t.send(request, data, credentials)
	if err != nil {
		return nil, err
	}
	if response.StatusCode == http.StatusUnauthorized && t.config.Refresh != nil {
		_ = response.Body.Close()
		if err := request.Context().Err(); err != nil {
			return nil, err
		}
		refreshed, err := t.config.Refresh(request.Context(), credentials.AccessToken)
		if err != nil {
			return nil, err
		}
		if refreshed.AccountID != credentials.AccountID {
			return nil, errors.New("codex refresh changed the account identity")
		}
		response, err = t.send(request, data, refreshed)
		if err != nil {
			return nil, err
		}
	}
	if response.StatusCode >= 400 {
		return normalizeError(response)
	}
	if !streaming && response.StatusCode >= 200 && response.StatusCode < 300 {
		return collectResponse(response)
	}
	return response, nil
}

func (t *transport) send(request *http.Request, data []byte, credentials Credentials) (*http.Response, error) {
	if !validHeaderValue(credentials.AccessToken) || !validHeaderValue(credentials.AccountID) {
		return nil, errors.New("codex access token and account ID are required")
	}
	clone := request.Clone(request.Context())
	clone.Body = io.NopCloser(bytes.NewReader(data))
	clone.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(data)), nil }
	clone.ContentLength = int64(len(data))
	clone.Header.Set("Content-Length", strconv.Itoa(len(data)))
	clone.Header.Set("Accept", "text/event-stream")
	clone.Header.Set("Authorization", "Bearer "+credentials.AccessToken)
	clone.Header.Set("Chatgpt-Account-Id", credentials.AccountID)
	clone.Header.Set("Session-Id", t.config.SessionID)
	clone.Header.Set("Originator", t.config.Originator)
	clone.Header.Del("Openai-Organization")
	clone.Header.Del("Openai-Project")
	clone.Header.Del("X-Openai-Internal-Codex-Residency")
	if value := residency(credentials.AccessToken); value != "" {
		if !validHeaderValue(value) {
			return nil, errors.New("invalid codex residency")
		}
		clone.Header.Set("X-Openai-Internal-Codex-Residency", value)
	}
	return t.next.RoundTrip(clone)
}

// collectResponse adapts Codex's streaming-only endpoint for Generate and GenerateObject.
// The existing Responses provider still decodes all output items and metadata.
func collectResponse(response *http.Response) (*http.Response, error) {
	decoder := ssestream.NewDecoder(response)
	defer func() { _ = decoder.Close() }()
	var output []json.RawMessage
	for decoder.Next() {
		var event struct {
			Type     string                     `json:"type"`
			Item     json.RawMessage            `json:"item"`
			Response map[string]json.RawMessage `json:"response"`
			Error    json.RawMessage            `json:"error"`
			Message  string                     `json:"message"`
			Code     string                     `json:"code"`
		}
		raw := decoder.Event().Data
		if bytes.Equal(bytes.TrimSpace(raw), []byte("[DONE]")) {
			break
		}
		if err := json.Unmarshal(raw, &event); err != nil {
			return nil, err
		}
		switch event.Type {
		case "response.output_item.done":
			output = append(output, event.Item)
		case "response.completed", "response.incomplete", "response.failed":
			if event.Response == nil {
				return nil, fantasy.NewIncompleteStreamError()
			}
			var terminalOutput []json.RawMessage
			if err := json.Unmarshal(event.Response["output"], &terminalOutput); err != nil && len(event.Response["output"]) > 0 {
				return nil, err
			}
			if len(terminalOutput) == 0 {
				event.Response["output"], _ = json.Marshal(output)
			}
			data, err := json.Marshal(event.Response)
			if err != nil {
				return nil, err
			}
			response.Body = io.NopCloser(bytes.NewReader(data))
			response.ContentLength = int64(len(data))
			response.Header.Set("Content-Type", "application/json")
			response.Header.Set("Content-Length", strconv.Itoa(len(data)))
			return response, nil
		case "error":
			return nil, &fantasy.Error{Title: "codex stream error", Message: cmp.Or(event.Message, event.Code, "response failed")}
		}
		if len(event.Error) > 0 && string(event.Error) != "null" {
			return nil, &ssestream.StreamError{Message: "codex stream error", Event: decoder.Event()}
		}
	}
	if err := decoder.Err(); err != nil {
		return nil, err
	}
	return nil, fantasy.NewIncompleteStreamError()
}

func validHeaderValue(value string) bool {
	return strings.TrimSpace(value) != "" && strings.TrimSpace(value) == value && !strings.ContainsFunc(value, unicode.IsControl)
}

// Token claims supply routing metadata only. The server validates the token.
func residency(token string) string {
	parts := strings.Split(token, ".")
	if len(parts) != 3 || len(parts[1]) > 96<<10 {
		return ""
	}
	data, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return ""
	}
	var claims struct {
		Residency string `json:"chatgpt_compute_residency"`
		Auth      struct {
			Residency string `json:"chatgpt_compute_residency"`
		} `json:"https://api.openai.com/auth"`
	}
	if json.Unmarshal(data, &claims) != nil {
		return ""
	}
	value := cmp.Or(claims.Auth.Residency, claims.Residency)
	if value == "no_constraint" {
		return ""
	}
	return value
}

// Codex can return detail outside the OpenAI error envelope.
func normalizeError(response *http.Response) (*http.Response, error) {
	defer func(body io.ReadCloser) { _ = body.Close() }(response.Body)
	data, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	var body struct {
		Detail string          `json:"detail"`
		Error  json.RawMessage `json:"error"`
	}
	if json.Unmarshal(data, &body) == nil && body.Detail != "" && (len(body.Error) == 0 || string(body.Error) == "null") {
		data, err = json.Marshal(map[string]any{"error": map[string]string{"message": body.Detail}})
		if err != nil {
			return nil, err
		}
	}
	response.Body = io.NopCloser(bytes.NewReader(data))
	response.ContentLength = int64(len(data))
	response.Header.Set("Content-Length", strconv.Itoa(len(data)))
	return response, nil
}
