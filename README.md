# gemma-gateway

Speaks the **Gemini REST API** at the front and **OpenAI chat-completions** at the
back, so Gemini CLI can drive a local vLLM-served Gemma.

    gemma-gateway --listen 127.0.0.1:8899 --upstream http://127.0.0.1:8891/v1 --model gemma-4-26b-a4b
    GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:8899 gemini

## Why a proxy instead of a fork

Gemini CLI honours `GOOGLE_GEMINI_BASE_URL` and has a matching `AuthType::GATEWAY`,
so **no CLI source is modified.** The fork stays byte-identical to upstream:
`git pull` keeps working, and every feature Google ships — hooks, MCP, subagents,
checkpointing, plan mode, sandboxing — arrives for free instead of being ported
one at a time.

Google never merged local-model support (eight closed community PRs for Ollama,
LM Studio, LocalAI and custom base URLs), but Apache-2.0 means the extension
point is still there to use.

## The three translations that matter

Everything else is mechanical. These are the ones with teeth:

| | Gemini | OpenAI |
|---|---|---|
| structure | `contents[].parts[]` | flat `messages[]` |
| tool call | a `functionCall` **part** | `tool_calls[]` with **JSON-string** arguments |
| reasoning | a part with `thought: true` | `delta.reasoning_content` |

The third one is a trap: a thinking model streams its first tokens into
`reasoning_content`, not `content`. Waiting only on `content` makes a working
model look hung — the same bug bit the benchmark harness on this rig.

## Status

15 translation unit tests, plus verified end-to-end against a live Gemma:
non-streaming, SSE streaming, `countTokens` (real counts via vLLM `/tokenize`),
and model listing.

Not yet exercised: multi-turn tool-call round-trips under a real agent loop, and
images. Both are expected to need iteration.
