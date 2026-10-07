"""Startup (lifespan) behavior for the web-tools prompt file: must fail
fast and explicitly when CLAUDE_ENABLE_WEB_TOOLS=true but the file is
missing/empty, rather than silently starting without the instruction."""
from __future__ import annotations

import pytest

import app.main as main_module
from app.claude_runner import WebPromptFileError
from app.config import load_settings


def _settings(**overrides):
    env = {
        "PROXY_API_KEY": "k",
        "CLAUDE_CODE_OAUTH_TOKEN": "t",
        "MODEL_MAP": "claude-sonnet=sonnet",
        "DEFAULT_MODEL": "claude-sonnet",
    }
    env.update(overrides)
    return load_settings(env)


async def test_lifespan_fails_fast_when_web_tools_enabled_with_missing_prompt_file(
    monkeypatch, tmp_path
):
    missing = tmp_path / "missing.txt"
    bad_settings = _settings(CLAUDE_ENABLE_WEB_TOOLS="true", CLAUDE_WEB_PROMPT_FILE=str(missing))
    monkeypatch.setattr(main_module, "settings", bad_settings)

    with pytest.raises(WebPromptFileError):
        async with main_module.lifespan(main_module.app):
            pass


async def test_lifespan_starts_cleanly_when_web_tools_disabled_even_with_bad_path(
    monkeypatch, tmp_path
):
    """Disabled mode must never even look at the path."""
    missing = tmp_path / "missing.txt"
    ok_settings = _settings(CLAUDE_ENABLE_WEB_TOOLS="false", CLAUDE_WEB_PROMPT_FILE=str(missing))
    monkeypatch.setattr(main_module, "settings", ok_settings)

    async with main_module.lifespan(main_module.app):
        pass  # must not raise


async def test_lifespan_starts_cleanly_when_web_tools_enabled_with_valid_prompt_file(
    monkeypatch, tmp_path
):
    path = tmp_path / "web-tools.txt"
    path.write_text("Valid instruction.", encoding="utf-8")
    good_settings = _settings(CLAUDE_ENABLE_WEB_TOOLS="true", CLAUDE_WEB_PROMPT_FILE=str(path))
    monkeypatch.setattr(main_module, "settings", good_settings)

    async with main_module.lifespan(main_module.app):
        pass  # must not raise
