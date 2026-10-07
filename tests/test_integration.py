"""Real calls to Claude Code. Consumes subscription quota, so this never runs
by default -- only with `RUN_CLAUDE_INTEGRATION_TESTS=1 pytest -m integration`.
"""
from __future__ import annotations

import json
import os

import pytest

from app.claude_runner import ClaudeRunner
from app.config import load_settings

pytestmark = pytest.mark.integration

_ENABLED = os.environ.get("RUN_CLAUDE_INTEGRATION_TESTS") == "1"


def _require_enabled():
    if not _ENABLED:
        pytest.skip("set RUN_CLAUDE_INTEGRATION_TESTS=1 and CLAUDE_CODE_OAUTH_TOKEN to run")


async def test_real_text_completion():
    _require_enabled()
    settings = load_settings(dict(os.environ))
    runner = ClaudeRunner(settings)
    result = await runner.run(prompt="Reply with exactly: OK", model="sonnet", json_schema=None)
    assert result.text.strip() == "OK"


async def test_real_structured_output():
    _require_enabled()
    settings = load_settings(dict(os.environ))
    runner = ClaudeRunner(settings)
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
        "additionalProperties": False,
    }
    result = await runner.run(prompt="Return status ok", model="sonnet", json_schema=schema)
    assert result.structured is not None
    json.dumps(result.structured)  # must be serializable
    assert "status" in result.structured
