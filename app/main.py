"""FastAPI app: HTTP boundary only. Business logic lives in openai_adapter /
claude_runner; this module wires routing, auth, request-size limits,
concurrency limiting, and error-to-HTTP mapping.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.auth import require_api_key
from app.claude_runner import (
    ClaudeExecutableNotFoundError,
    ClaudeInvalidOutputError,
    ClaudeRunner,
    ClaudeTimeoutError,
    ClaudeUpstreamError,
)
from app.config import Settings, claude_binary_path, get_settings
from app.errors import (
    InvalidRequestError,
    PayloadTooLargeError,
    ProxyError,
    RateLimitError,
    ServiceNotReadyError,
    UpstreamError,
    UpstreamTimeoutError,
)
from app.logging_config import configure_logging, get_request_id, set_request_id
from app.models import ChatCompletionRequest, ModelInfo, ModelList
from app.openai_adapter import build_chat_response, prepare_invocation

logger = logging.getLogger(__name__)

settings = get_settings()
configure_logging(settings.log_level)

_runner = ClaudeRunner(settings)
_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
_state: dict[str, str | None] = {"claude_version": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    _state["claude_version"] = await _runner.version()
    if _state["claude_version"]:
        logger.info("Claude Code version: %s", _state["claude_version"])
    else:
        logger.warning(
            "Claude Code executable %r not found on PATH; /readyz will report not_ready",
            settings.claude_binary,
        )
    yield
    logger.info("shutting down")


app = FastAPI(title="karakeep-claude-proxy", lifespan=lifespan, redoc_url=None)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex
    set_request_id(request_id)
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = int((time.monotonic() - start) * 1000)
    response.headers["X-Request-Id"] = request_id
    logger.info(
        "method=%s path=%s status=%s duration_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(ProxyError)
async def proxy_error_handler(_: Request, exc: ProxyError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error (request_id=%s)", get_request_id())
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "param": None,
                "code": "internal_error",
            }
        },
    )


async def _read_body_capped(request: Request, cap: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > cap:
                raise PayloadTooLargeError()
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > cap:
            raise PayloadTooLargeError()
        chunks.append(chunk)
    return b"".join(chunks)


def _clean_upstream_message(message: str) -> str:
    message = message.strip() or "Claude Code reported an error"
    return message[:500]


@app.get("/")
async def root() -> dict:
    return {"service": "karakeep-claude-proxy", "status": "ok"}


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(settings: Settings = Depends(get_settings)) -> JSONResponse:
    claude_cli = claude_binary_path(settings) is not None
    oauth_configured = bool(settings.claude_oauth_token)
    proxy_key_configured = bool(settings.proxy_api_key)
    model_map_valid = bool(settings.model_map) and settings.default_model in settings.model_map

    ready = claude_cli and oauth_configured and proxy_key_configured and model_map_valid
    payload = {
        "status": "ready" if ready else "not_ready",
        "claude_cli": claude_cli,
        "claude_version": _state["claude_version"],
        "oauth_token_configured": oauth_configured,
        "proxy_api_key_configured": proxy_key_configured,
        "model_map_valid": model_map_valid,
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models(settings: Settings = Depends(get_settings)) -> ModelList:
    return ModelList(data=[ModelInfo(id=alias) for alias in settings.available_models])


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(request: Request, settings: Settings = Depends(get_settings)):
    body = await _read_body_capped(request, settings.max_request_body_bytes)

    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidRequestError(f"Invalid JSON body: {exc}") from exc

    try:
        chat_request = ChatCompletionRequest.model_validate(payload)
    except ValidationError as exc:
        raise InvalidRequestError(f"Invalid request body: {exc.errors()[0]['msg']}") from exc

    resolved_model, prompt, json_schema, wants_structured = prepare_invocation(settings, chat_request)

    async with _semaphore:
        start = time.monotonic()
        try:
            result = await _runner.run(prompt=prompt, model=resolved_model, json_schema=json_schema)
        except ClaudeTimeoutError as exc:
            logger.warning(
                "requested_model=%s resolved_model=%s structured=%s claude_timeout_sec=%s",
                chat_request.model, resolved_model, wants_structured, settings.claude_timeout_sec,
            )
            raise UpstreamTimeoutError(str(exc)) from exc
        except ClaudeExecutableNotFoundError as exc:
            raise ServiceNotReadyError(str(exc)) from exc
        except ClaudeUpstreamError as exc:
            logger.warning(
                "requested_model=%s resolved_model=%s structured=%s claude_error_status=%s rate_limited=%s",
                chat_request.model, resolved_model, wants_structured, exc.api_error_status, exc.is_rate_limited,
            )
            if exc.is_rate_limited:
                raise RateLimitError(_clean_upstream_message(str(exc))) from exc
            raise UpstreamError(_clean_upstream_message(str(exc))) from exc
        except ClaudeInvalidOutputError as exc:
            logger.warning(
                "requested_model=%s resolved_model=%s structured=%s claude_invalid_output=%s",
                chat_request.model, resolved_model, wants_structured, str(exc),
            )
            raise UpstreamError("Claude Code returned an invalid response") from exc
        duration_ms = int((time.monotonic() - start) * 1000)

    logger.info(
        "requested_model=%s resolved_model=%s structured=%s claude_duration_ms=%s",
        chat_request.model, resolved_model, wants_structured, duration_ms,
    )
    return build_chat_response(
        request_model_alias=chat_request.model, claude_result=result, wants_structured=wants_structured
    )


if settings.enable_test_claude_endpoint:

    @app.post("/internal/test-claude", dependencies=[Depends(require_api_key)])
    async def test_claude(settings: Settings = Depends(get_settings)) -> dict:
        model = settings.resolve_model(settings.default_model)
        async with _semaphore:
            result = await _runner.run(prompt="Reply with exactly: OK", model=model, json_schema=None)
        return {"result": result.text}
