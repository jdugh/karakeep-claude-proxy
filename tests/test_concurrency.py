"""Verifies MAX_CONCURRENT_REQUESTS actually bounds concurrent `claude -p`
invocations (PROMPT.md section 14), using real concurrent async requests
against the ASGI app (TestClient alone can't issue overlapping requests).
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

import app.main as main_module
from app.claude_runner import ClaudeResult

from tests.conftest import API_KEY


class TrackingRunner:
    def __init__(self, hold_seconds: float = 0.05) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self._hold_seconds = hold_seconds

    async def run(self, *, prompt, model, json_schema, timeout_sec=None):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self._hold_seconds)
        finally:
            self.in_flight -= 1
        return ClaudeResult(text="OK", structured=None, session_id="s", duration_ms=1)

    async def version(self) -> str:
        return "fake-version"


@pytest.fixture
def tracking_runner(monkeypatch):
    fake = TrackingRunner()
    monkeypatch.setattr(main_module, "_runner", fake)
    monkeypatch.setattr(main_module, "_semaphore", asyncio.Semaphore(1))
    return fake


async def test_requests_are_serialized_by_semaphore(tracking_runner):
    transport = httpx.ASGITransport(app=main_module.app)
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {"model": "claude-sonnet", "messages": [{"role": "user", "content": "hi"}]}

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        results = await asyncio.gather(
            client.post("/v1/chat/completions", headers=headers, json=payload),
            client.post("/v1/chat/completions", headers=headers, json=payload),
            client.post("/v1/chat/completions", headers=headers, json=payload),
        )

    assert all(r.status_code == 200 for r in results)
    assert tracking_runner.max_in_flight == 1
