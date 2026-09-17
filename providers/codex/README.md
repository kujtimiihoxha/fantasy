# Codex

This provider uses the ChatGPT Codex endpoint. It implements all Fantasy model
methods and `fantasy.Compactor` through the OpenAI Responses provider.

Supply a ChatGPT access token and its account ID. The caller owns login, token
storage, and refresh. Both credential callbacks must be safe for concurrent calls.
The provider does not read the environment or `~/.codex/auth.json`.

```go
provider, err := codex.New(codex.Config{
    SessionID: sessionID,
    Originator: "my-app",
    Credentials: func(ctx context.Context) (codex.Credentials, error) {
        return loadCurrentCredentials(ctx)
    },
    Refresh: func(ctx context.Context, rejectedToken string) (codex.Credentials, error) {
        return refreshRejectedCredentials(ctx, rejectedToken)
    },
})
if err != nil {
    return err
}
model, err := provider.LanguageModel(ctx, "gpt-5.6-luna")
```

`Refresh` is optional. The provider retries HTTP 401 once when `Refresh` is set.
Refresh must keep the same account ID. The caller must store refreshed credentials
before returning them. Use one provider per account and conversation. Keep
`SessionID` stable across turns. It supplies the default prompt cache key.

Use `openai.ResponsesProviderOptions` under the `openai` key for request options.
Response metadata also uses the `openai` key. Store and restore Fantasy messages
with `encoding/json` to preserve reasoning, item IDs, phase, and checkpoints.
The provider always enables full replay and encrypted reasoning. It always sets
`store=false`. It does not support `PreviousResponseID` chaining.

Codex requires streaming requests. `Generate` and `GenerateObject` collect the
stream before the OpenAI provider decodes the result. `MaxOutputTokens` is omitted
because the Codex endpoint rejects it. Use a context deadline to bound a call.
An empty instructions string is sent if no instructions are supplied.

`HTTPClient` can supply a shared transport. The provider copies the client and
blocks redirects. Credentials go only to the configured Responses endpoint.
`BaseURL` accepts HTTPS, or loopback HTTP for tests.

[OpenAI authentication documentation](https://developers.openai.com/codex/auth)
describes ChatGPT login and API key login. This package uses ChatGPT credentials.
Use `providers/openai` for Platform API keys.

The provider test replays its cassette without credentials or network access.
To record it again, remove `providertests/testdata/TestCodexResponses.yaml` and
supply a JSON file with `access_token` and `account_id`. The test uses
`gpt-5.6-luna`. It does not refresh credentials. The recorder removes private
headers, and the test checks that credentials and account IDs are absent.

```sh
FANTASY_CODEX_TOKEN_FILE=/path/to/tokens.json go test ./providertests -run '^TestCodexResponses$' -count=1
```
