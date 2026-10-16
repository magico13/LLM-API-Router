# LLM API Router

OpenAI-compatible reverse proxy with automatic failover for local LLM backends (LM Studio, llama.cpp, etc.).

Try backends in priority order on every request — if the primary is down, fall through to the next one transparently. Supports both streaming (SSE) and non-streaming responses.

## Quick Start

### Docker (recommended)

```bash
# Edit `data/config.json` with your backend URLs and model overrides
docker compose up -d --build
```

The router listens on `http://localhost:9000`.

### Bare Python

```bash
pip install -r requirements.txt
python main.py
```

## Configuration

Edit [`data/config.json`](data/config.json):

```json
{
  "backends": [
    {
      "name": "primary",
      "url": "http://192.168.1.10:1234",
      "model_override": "your-primary-model"
    },
    {
      "name": "fallback",
      "url": "http://192.168.1.20:8080",
      "model_override": "your-fallback-model"
    }
  ],
  "connect_timeout_seconds": 1,
  "read_timeout_seconds": 300,
  "api_key": "your-secret-key"
}
```

| Field | Description |
|---|---|
| `name` | Human-readable label (shown in logs and error messages) |
| `url` | Base URL of the backend (no trailing `/v1/`) |
| `model_override` | Model name to send to this backend (overrides whatever the client requests) |
| `connect_timeout_seconds` | How long to wait for TCP connection before failing over (keep low — 1s is fine) |
| `read_timeout_seconds` | How long to wait after connecting for the response (LLM inference can take minutes) |
| `api_key` | Optional. If set, clients must present this key or the request is rejected with `401`. Omit or set to `null` to disable auth. |

### API Key Auth

When `api_key` is set, every request to `/v1/chat/completions`, `/v1/models`, and `/router/models` must present the key. The `/health` endpoint stays open for liveness probes.

The key is read from the `Authorization: Bearer <key>` header (standard for OpenAI-compatible clients) or, as a fallback, the `?api_key=<key>` query string.

```bash
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{ "model": "any", "messages": [{"role": "user", "content": "Hi"}] }'
```

> **Note:** This is a simple shared-secret gate, not a full auth system. It's fine for keeping casual strangers out of a home network, but if you expose the router to the public internet, put it behind a reverse proxy with TLS (and consider a stronger auth layer).

Backends are tried **in order** — list your preferred server first.

### Environment Variable Overrides (optional)

You can override any config value without touching `config.json` by setting environment variables:

| Env Var | Description | Example |
|---|---|---|
| `BACKEND_0_URL` | URL for backend at index 0 (primary) | `http://192.168.1.10:1234` |
| `BACKEND_0_MODEL_OVERRIDE` | Model override for backend 0 | `your-primary-model` |
| `BACKEND_1_URL` | URL for backend at index 1 (fallback) | `http://192.168.1.20:8080` |
| `BACKEND_1_MODEL_OVERRIDE` | Model override for backend 1 | `your-fallback-model` |
| `CONNECT_TIMEOUT_SECONDS` | TCP connect timeout (keep low for fast failover) | `1` |
| `READ_TIMEOUT_SECONDS` | Read/inference timeout (LLM can take minutes) | `300` |
| `ROUTER_API_KEY` | Override the `api_key` from config (useful for keeping the secret out of the JSON) | `your-secret-key` |

Index matches the order in `config.json`. Unset vars fall back to the JSON values.

```yaml
# docker-compose.yml example with overrides
environment:
  BACKEND_0_URL: "http://new-host:1234"
  BACKEND_1_MODEL_OVERRIDE: "different-model"
```

## Usage

Point any OpenAI-compatible client at `http://<router-host>:9000/v1/chat/completions`.

### Non-streaming (curl)

```bash
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any-model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming (curl)

```bash
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Home Assistant (OpenAI Conversation)

In your `configuration.yaml`, point the OpenAI integration at the router:

```yaml
conversation:
  - platform: openai_conversation
    api_key: "your-secret-key"     # the router's api_key (required when auth is enabled)
    model: "any"                   # overridden by the router per-backend
```

Then set the base URL in your OpenAI config to `http://<router-host>:9000`.

### Discord Bot (discord.py + openai package)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9000/v1",
    api_key="your-secret-key"      # the router's api_key (required when auth is enabled)
)

response = client.chat.completions.create(
    model="any",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Reasoning Control

The router normalizes the several incompatible ways clients express reasoning
(thinking) control into the fields llama.cpp actually honors. This matters
because llama.cpp's OpenAI endpoint **silently ignores**
`thinking: {type: "disabled"}` — it only reacts to `type == "enabled"` — so
without this translation the model reasons anyway.

Any of these client dialects are accepted and translated:

| Client sends | Effect |
|---|---|
| `thinking: {type: "disabled"}` | Reasoning off |
| `thinking: {type: "enabled"}` | Reasoning on |
| `thinking: {type: "enabled", budget_tokens: N}` | Reasoning on, capped at N tokens |
| `reasoning_effort: "none"` | Reasoning off |
| `reasoning_effort: "low" \| "medium" \| "high"` | Reasoning on at that effort |
| `reasoning: {effort: X, max_tokens: N}` | Reasoning on at effort X, capped at N tokens |

Under the hood these become llama.cpp-native fields:

- `chat_template_kwargs.enable_thinking` (a JSON **boolean**) — the on/off
  switch the model's Jinja template checks.
- `reasoning_effort` — passed through for real effort levels.
- `reasoning_budget_tokens` — token budget for the reasoning block.

The client dialect keys (`thinking`, `reasoning`) are stripped before the
request is forwarded, so they are not double-processed by the backend.

```bash
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "thinking": {"type": "disabled"}
  }'
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Proxied with failover (streaming + non-streaming) |
| `GET`  | `/v1/models` | Merged, deduplicated model list from all available backends |
| `GET`  | `/router/models` | Per-backend breakdown: which models each backend serves |
| `GET`  | `/health` | Liveness probe — lists configured backends |

## Response Headers

Every successful response includes an `X-Backend` header showing which backend handled the request (e.g., `X-Backend: primary`).

## Failover Behavior

- On each request, the router tries backends in config order.
- Connection refused, timeout, or non-2xx → try next backend immediately.
- If **all** backends fail → returns HTTP 503 with details of what went wrong.
- No background polling — health is checked on-request only (low overhead).

## Future Ideas

- **Model-aware routing**: Instead of overriding the model per-backend, route requests to whichever backend actually hosts the requested model. Would be an alternate `routing_mode` alongside the current `"override"` mode — useful when clients explicitly want specific models rather than just "the best available."
