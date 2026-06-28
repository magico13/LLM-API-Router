"""LLM API Router — OpenAI-compatible reverse proxy with failover."""

import json
import logging
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llm-router")

# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


class BackendServer(BaseModel):
    name: str
    url: str
    model_override: str


class RouterConfig(BaseModel):
    backends: list[BackendServer]
    connect_timeout_seconds: float = 1.0
    read_timeout_seconds: float = 300.0


# ---------------------------------------------------------------------------
# Config loading (cached after first load)
# ---------------------------------------------------------------------------

_config_cache: RouterConfig | None = None


def load_config() -> RouterConfig:
    """Load config from data/config.json, then apply env var overrides."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config_path = Path(__file__).parent / "data" / "config.json"
    logger.info("Loading config from %s", config_path)
    data = json.loads(config_path.read_text())
    cfg = RouterConfig(**data)

    # Apply env var overrides (BACKEND_<index>_<FIELD>)
    for i, backend in enumerate(cfg.backends):
        prefix = f"BACKEND_{i}"
        if os.getenv(f"{prefix}_URL"):
            backend.url = os.environ[f"{prefix}_URL"]
            logger.info("Overriding %s URL via env: %s", backend.name, backend.url)
        if os.getenv(f"{prefix}_MODEL_OVERRIDE"):
            backend.model_override = os.environ[f"{prefix}_MODEL_OVERRIDE"]
            logger.info(
                "Overriding %s model via env: %s", backend.name, backend.model_override
            )
        if os.getenv(f"{prefix}_NAME"):
            backend.name = os.environ[f"{prefix}_NAME"]

    if os.getenv("CONNECT_TIMEOUT_SECONDS"):
        cfg.connect_timeout_seconds = float(os.environ["CONNECT_TIMEOUT_SECONDS"])
    if os.getenv("READ_TIMEOUT_SECONDS"):
        cfg.read_timeout_seconds = float(os.environ["READ_TIMEOUT_SECONDS"])

    _config_cache = cfg
    logger.info(
        "Loaded %d backend(s): %s",
        len(_config_cache.backends),
        ", ".join(b.name for b in _config_cache.backends),
    )
    return _config_cache


# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------


def apply_overrides(body: dict, backend: BackendServer) -> dict:
    """Return a copy of the request body with per-server overrides applied."""
    patched = {k: v for k, v in body.items() if not k.startswith("_")}
    if backend.model_override:
        patched["model"] = backend.model_override
    return patched


async def proxy_non_streaming(
    client: httpx.AsyncClient, backend: BackendServer, body: dict
) -> tuple[int, dict, bytes]:
    """Forward a non-streaming request to *backend* and return (status, headers, raw_body)."""
    url = str(backend.url.rstrip("/") + "/v1/chat/completions")
    patched = apply_overrides(body, backend)

    cfg = load_config()
    timeout = httpx.Timeout(
        connect=cfg.connect_timeout_seconds,
        read=cfg.read_timeout_seconds,
        write=5.0,
        pool=cfg.connect_timeout_seconds,
    )
    resp = await client.post(url, json=patched, timeout=timeout)
    return resp.status_code, dict(resp.headers), resp.content


async def proxy_streaming(
    client: httpx.AsyncClient, backend: BackendServer, body: dict
):
    """Forward a streaming request and yield SSE chunks from the backend."""
    url = str(backend.url.rstrip("/") + "/v1/chat/completions")
    patched = apply_overrides(body, backend)

    cfg = load_config()
    timeout = httpx.Timeout(
        connect=cfg.connect_timeout_seconds,
        read=cfg.read_timeout_seconds,
        write=5.0,
        pool=cfg.connect_timeout_seconds,
    )
    async with client.stream("POST", url, json=patched, timeout=timeout) as resp:
        if resp.status_code != 200:
            error_body = await resp.aread()
            raise httpx.HTTPStatusError(
                f"{resp.status_code} from {backend.name}",
                request=resp.request,
                response=resp,
            )
        async for chunk in resp.aiter_bytes():
            yield chunk


def backend_timeout(client: httpx.AsyncClient) -> float | None:
    """Return the read timeout value from config (connect is set on client init)."""
    cfg = load_config()
    return cfg.read_timeout_seconds


def connect_timeout(cfg: RouterConfig | None = None) -> float:
    """Return the connect timeout value from config."""
    if cfg is None:
        cfg = load_config()
    return cfg.connect_timeout_seconds


# ---------------------------------------------------------------------------
# Failover routing
# ---------------------------------------------------------------------------


async def route_request(
    client: httpx.AsyncClient, body: dict, streaming: bool
) -> tuple[StreamingResponse | JSONResponse, BackendServer]:
    """Try backends in priority order until one succeeds.

    Returns (response_object, backend_that_succeeded).
    On total failure returns a 503 JSONResponse and None as the backend.
    """
    cfg = load_config()
    errors: list[str] = []

    for backend in cfg.backends:
        logger.info("Trying %s (%s) …", backend.name, backend.url)
        try:
            if streaming:
                # Streaming — _stream_with_failover manages its own client lifecycle
                extra_headers = {}
                auth_header = body.pop("_auth", None)  # injected by endpoint
                if auth_header:
                    extra_headers["Authorization"] = auth_header
                stream_gen = _stream_with_failover(
                    list(cfg.backends), body, start_idx=list(cfg.backends).index(backend),
                    extra_headers=extra_headers or None,
                )
                return (
                    StreamingResponse(
                        stream_gen,
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Backend": backend.name,
                        },
                    ),
                    backend,
                )
            else:
                status, headers, content = await proxy_non_streaming(client, backend, body)

                if status == 200:
                    # Drop headers that FastAPI will recompute (content-length, transfer-encoding)
                    resp_headers = {
                        k: v for k, v in headers.items()
                        if k.lower() not in ("content-length", "transfer-encoding")
                    }
                    resp_headers["X-Backend"] = backend.name
                    return (
                        JSONResponse(
                            content=json.loads(content),
                            status_code=status,
                            headers=resp_headers,
                        ),
                        backend,
                    )
                else:
                    errors.append(f"{backend.name}: HTTP {status}")
                    logger.warning("%s returned HTTP %d", backend.name, status)

        except httpx.TimeoutException:
            errors.append(f"{backend.name}: timeout")
            logger.warning("Timeout connecting to %s", backend.name)
        except httpx.ConnectError as exc:
            errors.append(f"{backend.name}: connection refused ({exc})")
            logger.warning("Cannot connect to %s — %s", backend.name, exc)
        except httpx.HTTPStatusError as exc:
            errors.append(f"{backend.name}: {exc}")
            logger.warning("HTTP error from %s — %s", backend.name, exc)
        except Exception as exc:
            errors.append(f"{backend.name}: {exc!r}")
            logger.exception("Unexpected error with %s", backend.name)

    # All backends failed
    logger.error("All backends unavailable: %s", "; ".join(errors))
    return (
        JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "All LLM backends are unavailable",
                    "details": errors,
                }
            },
        ),
        None,  # type: ignore
    )


async def _stream_with_failover(
    backends: list[BackendServer],
    body: dict,
    start_idx: int,
    extra_headers: dict | None = None,
):
    """Async generator that tries streaming from *start_idx* onward.

    Manages its own httpx client so it stays alive for the full stream lifetime.
    """
    client_kwargs = {"verify": False}
    if extra_headers:
        client_kwargs["headers"] = extra_headers

    async with httpx.AsyncClient(**client_kwargs) as client:
        for i in range(start_idx, len(backends)):
            backend = backends[i]
            logger.info("Streaming from %s (%s) …", backend.name, backend.url)
            try:
                async for chunk in proxy_streaming(client, backend, body):
                    yield chunk
                return  # success — exit
            except httpx.TimeoutException:
                logger.warning("Timeout streaming from %s, trying next …", backend.name)
            except httpx.ConnectError as exc:
                logger.warning("Cannot connect to %s (%s), trying next …", backend.name, exc)
            except httpx.HTTPStatusError as exc:
                logger.warning("HTTP error streaming from %s (%s), trying next …", backend.name, exc)
            except Exception as exc:
                logger.exception("Unexpected stream error with %s, trying next …", backend.name)

        # Exhausted all backends during streaming — send a final error event
        yield 'data: {"error": "All LLM backends are unavailable"}\n\n'


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="LLM API Router", version="1.0.0")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Proxy an OpenAI-compatible chat completion request with failover."""
    body = await request.json()
    streaming = body.get("stream", False)

    auth_header = request.headers.get("authorization")
    extra_headers = {}
    if auth_header:
        extra_headers["Authorization"] = auth_header

    # For streaming, inject auth into body so _stream_with_failover can use it
    # (it manages its own client lifecycle to keep the connection alive)
    if streaming and auth_header:
        body["_auth"] = auth_header

    async with httpx.AsyncClient(headers=extra_headers, verify=False) as client:
        response, backend = await route_request(client, body, streaming)
        return response


@app.get("/v1/models")
async def list_models():
    """Aggregate models from all available backends (merged, deduplicated)."""
    cfg = load_config()

    async with httpx.AsyncClient(verify=False) as client:
        seen_ids: set[str] = set()
        merged_models: list[dict] = []

        for backend in cfg.backends:
            url = str(backend.url.rstrip("/") + "/v1/models")
            try:
                timeout = httpx.Timeout(
                    connect=cfg.connect_timeout_seconds,
                    read=cfg.read_timeout_seconds,
                    write=5.0,
                    pool=cfg.connect_timeout_seconds,
                )
                resp = await client.get(url, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    for model in data.get("data", []):
                        mid = model.get("id")
                        if mid and mid not in seen_ids:
                            seen_ids.add(mid)
                            merged_models.append(model)
            except Exception as exc:
                logger.warning("Failed to fetch models from %s — %s", backend.name, exc)

    return JSONResponse(content={"object": "list", "data": merged_models})


@app.get("/router/models")
async def list_backends_with_models():
    """Router-aware endpoint: shows each backend and which models it serves."""
    cfg = load_config()
    results: list[dict] = []

    async with httpx.AsyncClient(verify=False) as client:
        for backend in cfg.backends:
            url = str(backend.url.rstrip("/") + "/v1/models")
            try:
                timeout = httpx.Timeout(
                    connect=cfg.connect_timeout_seconds,
                    read=cfg.read_timeout_seconds,
                    write=5.0,
                    pool=cfg.connect_timeout_seconds,
                )
                resp = await client.get(url, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    model_ids = [m.get("id", "") for m in data.get("data", [])]
                    results.append(
                        {
                            "name": backend.name,
                            "url": backend.url,
                            "model_override": backend.model_override,
                            "status": "online",
                            "models": model_ids,
                        }
                    )
                else:
                    results.append(
                        {
                            "name": backend.name,
                            "url": backend.url,
                            "model_override": backend.model_override,
                            "status": f"http_{resp.status_code}",
                            "models": [],
                        }
                    )
            except Exception as exc:
                results.append(
                    {
                        "name": backend.name,
                        "url": backend.url,
                        "model_override": backend.model_override,
                        "status": f"unreachable ({exc!r})",
                        "models": [],
                    }
                )

    return JSONResponse(content={"backends": results})


@app.get("/health")
async def health():
    """Simple liveness probe."""
    cfg = load_config()
    return {
        "status": "ok",
        "backends": [
            {"name": b.name, "url": b.url, "model_override": b.model_override}
            for b in cfg.backends
        ],
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
