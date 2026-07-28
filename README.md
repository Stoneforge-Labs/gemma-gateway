# Gemma Gateway

Run the upstream [Gemini CLI](https://github.com/google-gemini/gemini-cli)
against a local model served by vLLM. The CLI is not forked or patched.

```text
Gemini CLI ── Gemini REST ──▶ gemma-gateway ── OpenAI chat ──▶ vLLM ──▶ Gemma
```

## Quick start

Requirements: Rust, Gemini CLI, and an OpenAI-compatible vLLM server.

```bash
cargo build --release

./target/release/gemma-gateway \
  --listen 127.0.0.1:8899 \
  --upstream http://127.0.0.1:8891/v1

GEMINI_API_KEY=local \
GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:8899 \
gemini
```

Or run the published container:

```bash
docker run --rm -p 127.0.0.1:8899:8899 \
  --add-host host.docker.internal:host-gateway \
  ghcr.io/stoneforge-labs/gemma-gateway:latest
```

Production should pin the immutable commit tag or its resolved digest, not
`latest`.

Without `--model`, the gateway reads vLLM's `/models` endpoint and follows model
changes. Use `--model NAME` only to pin a fixed upstream.

Gemini CLI also needs API-key auth selected in `~/.gemini/settings.json`:

```json
{
  "security": {
    "auth": {
      "selectedType": "gemini-api-key"
    }
  }
}
```

## What it translates

| Gemini API | Local OpenAI/vLLM |
|---|---|
| `contents[].parts[]` and system instructions | chat-completion messages |
| function declarations, calls, and responses | tools and tool messages |
| fragmented streamed function calls | complete tool calls with JSON arguments |
| `thought: true` parts | `reasoning_content` |
| JSON/OpenAPI response schemas | vLLM structured output |
| `toolConfig` modes | `tool_choice` |
| streamed usage metadata | `stream_options.include_usage` |
| inline and file media parts | OpenAI image, audio, and video parts |
| `countTokens` | vLLM `/tokenize`, with a documented estimate fallback |
| Gemini model listing | vLLM `/models` |

The gateway implements `generateContent`, `streamGenerateContent`, `countTokens`,
and model listing on both `/v1` and `/v1beta`. OpenAI-compatible clients can
share the same ingress through `/v1/chat/completions`.

## Gemini CLI reliability behavior

- Gemini CLI's `auto` mode asks stock Gemini router model names first. The
  gateway advertises those aliases but resolves every request to the model
  actually loaded in vLLM.
- A rejected request clears the discovered-model cache and retries once after
  rediscovery, so switching vLLM profiles does not require restarting the
  gateway.
- A streamed turn that produces no visible text or tool call is retried at most
  twice, but only before anything has been sent to the client.
- Upstream 4xx responses remain 4xx; transport and upstream server failures are
  reported as gateway failures.

`/health` proves that the gateway process is alive. Use `/v1/models` to prove
that the upstream model is ready.

## Why this shape

`GOOGLE_GEMINI_BASE_URL` redirects Gemini CLI's inference requests to the local
gateway. The CLI stays upstream-compatible, while the gateway owns the protocol
differences that otherwise break tools, reasoning, structured output, and token
accounting.

This replaces the inference path, not the CLI. MCP tools and other integrations
may still use their own network connections.

## Verification

```bash
cargo test

# With the gateway and vLLM running:
GEMINI_MODEL=gemma-4-26b-a4b python3 verify_live.py
```

The repository has 31 translation tests. `verify_live.py` checks structured
JSON, tool suppression, and streamed token usage against a live model.

## Security

The gateway does not implement authentication. Keep it and vLLM bound to
`127.0.0.1`; do not expose either port to an untrusted network. `GEMINI_API_KEY`
is a placeholder required by the client SDK and is not validated by the
gateway.

## License

MIT

## Author

[Jayson Stone](https://stepdaddenergy.github.io/) ·
[Google Scholar](https://scholar.google.com/citations?user=SrzMV7sAAAAJ&hl=en) ·
[Stoneforge Labs](https://github.com/Stoneforge-Labs)
