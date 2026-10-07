"""Environment-driven configuration."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from functools import lru_cache


class ConfigError(Exception):
    """Raised when the environment is misconfigured."""


def _parse_model_map(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ConfigError(f"Invalid MODEL_MAP entry (expected alias=model): {pair!r}")
        alias, _, target = pair.partition("=")
        alias, target = alias.strip(), target.strip()
        if not alias or not target:
            raise ConfigError(f"Invalid MODEL_MAP entry (expected alias=model): {pair!r}")
        mapping[alias] = target
    if not mapping:
        raise ConfigError("MODEL_MAP must define at least one alias=model pair")
    return mapping


def _env_bool(env: dict[str, str], name: str, default: bool) -> bool:
    val = env.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(env: dict[str, str], name: str, default: int) -> int:
    val = env.get(name)
    if val is None or not val.strip():
        return default
    try:
        return int(val)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {val!r}") from exc


@dataclass(frozen=True)
class Settings:
    proxy_api_key: str | None
    claude_oauth_token: str | None
    claude_binary: str
    default_model: str
    model_map: dict[str, str]
    claude_timeout_sec: int
    max_concurrent_requests: int
    max_request_body_bytes: int
    max_retries: int
    log_level: str
    enable_test_claude_endpoint: bool
    claude_enable_web_tools: bool
    permission_mode: str = "none"  # fixed: Claude never gets an interactive permission prompt

    def resolve_model(self, requested: str) -> str | None:
        return self.model_map.get(requested)

    @property
    def available_models(self) -> list[str]:
        return list(self.model_map.keys())


def load_settings(env: dict[str, str] | None = None) -> Settings:
    e = env if env is not None else os.environ

    model_map_raw = e.get(
        "MODEL_MAP", "claude-sonnet=sonnet,claude-haiku=haiku,claude-opus=opus"
    )
    model_map = _parse_model_map(model_map_raw)

    default_model = e.get("DEFAULT_MODEL", "claude-sonnet")
    if default_model not in model_map:
        raise ConfigError(
            f"DEFAULT_MODEL={default_model!r} is not present in MODEL_MAP {model_map!r}"
        )

    claude_binary = e.get("CLAUDE_BINARY", "claude")

    return Settings(
        proxy_api_key=e.get("PROXY_API_KEY") or None,
        claude_oauth_token=e.get("CLAUDE_CODE_OAUTH_TOKEN") or None,
        claude_binary=claude_binary,
        default_model=default_model,
        model_map=model_map,
        claude_timeout_sec=_env_int(e, "CLAUDE_TIMEOUT_SEC", 180),
        max_concurrent_requests=_env_int(e, "MAX_CONCURRENT_REQUESTS", 1),
        max_request_body_bytes=_env_int(e, "MAX_REQUEST_BODY_BYTES", 2 * 1024 * 1024),
        max_retries=_env_int(e, "MAX_RETRIES", 0),
        log_level=e.get("LOG_LEVEL", "INFO").upper(),
        enable_test_claude_endpoint=_env_bool(e, "ENABLE_TEST_CLAUDE_ENDPOINT", False),
        claude_enable_web_tools=_env_bool(e, "CLAUDE_ENABLE_WEB_TOOLS", False),
    )


def claude_binary_path(settings: Settings) -> str | None:
    """Resolve the configured Claude binary on PATH, or None if missing."""
    return shutil.which(settings.claude_binary)


@lru_cache
def get_settings() -> Settings:
    return load_settings()
