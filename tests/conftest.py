"""Shared fixtures. Environment must be set before app.main is imported,
since settings/app are built at module import time."""
from __future__ import annotations

import os

os.environ.setdefault("PROXY_API_KEY", "test-proxy-key")
os.environ.setdefault("CLAUDE_CODE_OAUTH_TOKEN", "test-oauth-token")
os.environ.setdefault("DEFAULT_MODEL", "claude-sonnet")
os.environ.setdefault("MODEL_MAP", "claude-sonnet=sonnet,claude-haiku=haiku,claude-opus=opus")
os.environ.setdefault("MAX_CONCURRENT_REQUESTS", "2")
os.environ.setdefault("CLAUDE_TIMEOUT_SEC", "5")
os.environ.setdefault("LOG_LEVEL", "WARNING")

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.claude_runner import ClaudeResult

API_KEY = os.environ["PROXY_API_KEY"]


class FakeClaudeRunner:
    """Stand-in for ClaudeRunner: unit tests never spawn a real `claude` process."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.next_result: ClaudeResult | None = None
        self.next_exception: Exception | None = None

    async def run(self, *, prompt: str, model: str, json_schema: dict | None, timeout_sec: float | None = None):
        self.calls.append({"prompt": prompt, "model": model, "json_schema": json_schema})
        if self.next_exception is not None:
            raise self.next_exception
        if self.next_result is not None:
            return self.next_result
        return ClaudeResult(text="OK", structured=None, session_id="fake-session", duration_ms=1)

    async def version(self) -> str:
        return "fake-version"


@pytest.fixture
def fake_runner(monkeypatch) -> FakeClaudeRunner:
    fake = FakeClaudeRunner()
    monkeypatch.setattr(main_module, "_runner", fake)
    return fake


@pytest.fixture
def client(fake_runner: FakeClaudeRunner):
    # raise_server_exceptions=False: exercise the real production path, where
    # our Exception handler turns a bug into a clean 500 instead of letting
    # the test client re-raise it.
    with TestClient(main_module.app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}
