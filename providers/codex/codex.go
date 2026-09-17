// Package codex connects Fantasy to the ChatGPT Codex Responses endpoint.
package codex

import (
	"cmp"
	"context"
	"errors"
	"maps"
	"net"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"github.com/openai/openai-go/v3/option"
)

const (
	// Name identifies the Codex provider. Replay metadata uses the openai key.
	Name = "codex"
	// DefaultURL is the ChatGPT Codex endpoint.
	DefaultURL = "https://chatgpt.com/backend-api/codex"
)

// Credentials contains a ChatGPT access token and its account identity.
type Credentials struct {
	AccessToken string
	AccountID   string
}

// Config connects a provider to one account and conversation.
type Config struct {
	// Credentials returns current credentials for each request. The caller owns login and storage.
	Credentials func(context.Context) (Credentials, error)
	// Refresh replaces a rejected token. It is called at most once after HTTP 401.
	// Both callbacks must be safe for concurrent calls.
	Refresh    func(context.Context, string) (Credentials, error)
	SessionID  string
	BaseURL    string
	Originator string
	HTTPClient *http.Client
}

type provider struct {
	fantasy.Provider
	sessionID string
}

// New creates a Codex provider without reading credentials from the environment or disk.
func New(config Config) (fantasy.Provider, error) {
	if config.Credentials == nil {
		return nil, errors.New("codex credentials callback is required")
	}
	if !validHeaderValue(config.SessionID) {
		return nil, errors.New("codex session ID is required and must be a valid header value")
	}
	config.Originator = cmp.Or(config.Originator, "fantasy")
	if !validHeaderValue(config.Originator) {
		return nil, errors.New("invalid codex originator")
	}
	base, err := url.Parse(cmp.Or(config.BaseURL, DefaultURL))
	if err != nil {
		return nil, errors.New("invalid codex endpoint")
	}
	loopback := strings.EqualFold(base.Hostname(), "localhost") || net.ParseIP(base.Hostname()).IsLoopback()
	if base.Host == "" || base.User != nil || base.RawQuery != "" || base.Fragment != "" || (base.Scheme != "https" && (base.Scheme != "http" || !loopback)) {
		return nil, errors.New("invalid codex endpoint")
	}
	base.Path = strings.TrimRight(base.Path, "/") + "/"
	base.RawPath = ""
	endpoint := *base
	endpoint.Path += "responses"
	client := *cmp.Or(config.HTTPClient, http.DefaultClient)
	client.Transport = &transport{config: config, endpoint: &endpoint, next: cmp.Or(client.Transport, http.DefaultTransport)}
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	inner, err := openai.New(
		openai.WithName(Name), openai.WithBaseURL(base.String()), openai.WithHTTPClient(&client),
		openai.WithUseResponsesAPI(), openai.WithResponsesAPIFunc(func(string) bool { return true }),
		openai.WithSDKOptions(option.WithAPIKey(""), option.WithOrganization(""), option.WithProject("")),
	)
	if err != nil {
		return nil, err
	}
	return &provider{Provider: inner, sessionID: config.SessionID}, nil
}

func (p *provider) LanguageModel(ctx context.Context, modelID string) (fantasy.LanguageModel, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("codex model ID is required")
	}
	inner, err := p.Provider.LanguageModel(ctx, modelID)
	if err != nil {
		return nil, err
	}
	return &model{LanguageModel: inner, sessionID: p.sessionID}, nil
}

type model struct {
	fantasy.LanguageModel
	sessionID string
}

var _ fantasy.Compactor = (*model)(nil)

func (m *model) options(input fantasy.ProviderOptions) fantasy.ProviderOptions {
	result := maps.Clone(input)
	if result == nil {
		result = make(fantasy.ProviderOptions)
	}
	var opts openai.ResponsesProviderOptions
	if supplied, ok := input[openai.Name].(*openai.ResponsesProviderOptions); ok && supplied != nil {
		opts = *supplied
	}
	opts.FullReplay = true
	opts.Store = new(false)
	if opts.Instructions == nil {
		opts.Instructions = new("")
	}
	if opts.PromptCacheKey == nil {
		opts.PromptCacheKey = new(m.sessionID)
	}
	opts.Include = slices.Clone(opts.Include)
	if !slices.Contains(opts.Include, openai.IncludeReasoningEncryptedContent) {
		opts.Include = append(opts.Include, openai.IncludeReasoningEncryptedContent)
	}
	result[openai.Name] = &opts
	return result
}

func (m *model) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	call.ProviderOptions = m.options(call.ProviderOptions)
	return m.LanguageModel.Generate(ctx, call)
}

func (m *model) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	call.ProviderOptions = m.options(call.ProviderOptions)
	return m.LanguageModel.Stream(ctx, call)
}

func (m *model) Compact(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	call.ProviderOptions = m.options(call.ProviderOptions)
	return m.LanguageModel.(fantasy.Compactor).Compact(ctx, call)
}

func (m *model) GenerateObject(ctx context.Context, call fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	call.ProviderOptions = m.options(call.ProviderOptions)
	return m.LanguageModel.GenerateObject(ctx, call)
}

func (m *model) StreamObject(ctx context.Context, call fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	call.ProviderOptions = m.options(call.ProviderOptions)
	return m.LanguageModel.StreamObject(ctx, call)
}
