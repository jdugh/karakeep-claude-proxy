from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from app.claude_runner import (
    ClaudeExecutableNotFoundError,
    ClaudeInvalidOutputError,
    ClaudeRunner,
    ClaudeTimeoutError,
    ClaudeUpstreamError,
    WebPromptFileError,
    _read_capped,
    read_web_prompt_file,
)
from app.config import load_settings


def _settings(**overrides):
    env = {
        "PROXY_API_KEY": "k",
        "CLAUDE_CODE_OAUTH_TOKEN": "t",
        "MODEL_MAP": "claude-sonnet=sonnet",
        "DEFAULT_MODEL": "claude-sonnet",
        "CLAUDE_TIMEOUT_SEC": "5",
    }
    env.update(overrides)
    return load_settings(env)


WEB_PROMPT_TEST_CONTENT = "Test search guidance instruction."


@pytest.fixture
def web_prompt_file(tmp_path) -> str:
    """A valid CLAUDE_WEB_PROMPT_FILE for tests that enable web tools."""
    path = tmp_path / "web-tools.txt"
    path.write_text(WEB_PROMPT_TEST_CONTENT, encoding="utf-8")
    return str(path)


class FakeStreamWriter:
    def __init__(self) -> None:
        self.written = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


class FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0, hang_seconds: float = 0):
        self.stdout = _reader(stdout)
        self.stderr = _reader(stderr)
        self.stdin = FakeStreamWriter()
        self.returncode: int | None = returncode
        self._hang_seconds = hang_seconds
        self.killed = False
        self.pid = 1234

    async def wait(self):
        if self._hang_seconds:
            await asyncio.sleep(self._hang_seconds)
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


@pytest.fixture
def runner_with_fake_process(monkeypatch):
    def _install(process: FakeProcess):
        captured_argv: list[str] = []

        async def fake_create_subprocess_exec(*argv, **kwargs):
            captured_argv.extend(argv)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
        monkeypatch.setattr("app.claude_runner.claude_binary_path", lambda settings: "/usr/bin/fake-claude")
        return captured_argv

    return _install


async def test_successful_text_response(runner_with_fake_process):
    envelope = {"is_error": False, "result": "OK", "session_id": "abc"}
    process = FakeProcess(stdout=json.dumps(envelope).encode())
    argv = runner_with_fake_process(process)

    runner = ClaudeRunner(_settings())
    result = await runner.run(prompt="hi", model="sonnet", json_schema=None)

    assert result.text == "OK"
    assert result.structured is None
    assert process.stdin.written == b"hi"
    assert process.stdin.closed
    assert "--tools" in argv and "" in argv  # --tools "" present somewhere
    assert "--json-schema" not in argv


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


async def test_web_tools_disabled_by_default(runner_with_fake_process):
    """CLAUDE_ENABLE_WEB_TOOLS absent or false: unchanged -- no tools at all."""
    envelope = {"is_error": False, "result": "OK"}
    process = FakeProcess(stdout=json.dumps(envelope).encode())
    argv = runner_with_fake_process(process)

    runner = ClaudeRunner(_settings(CLAUDE_ENABLE_WEB_TOOLS="false"))
    await runner.run(prompt="hi", model="sonnet", json_schema=None)

    assert _flag_value(argv, "--tools") == ""
    assert "--allowedTools" not in argv
    assert _flag_value(argv, "--disallowedTools") == "mcp__*"


async def test_web_tools_enabled_exposes_only_websearch_and_webfetch(
    runner_with_fake_process, web_prompt_file
):
    envelope = {"is_error": False, "result": "OK"}
    process = FakeProcess(stdout=json.dumps(envelope).encode())
    argv = runner_with_fake_process(process)

    runner = ClaudeRunner(_settings(CLAUDE_ENABLE_WEB_TOOLS="true", CLAUDE_WEB_PROMPT_FILE=web_prompt_file))
    await runner.run(prompt="hi", model="sonnet", json_schema=None)

    assert _flag_value(argv, "--tools") == "WebSearch,WebFetch"
    assert _flag_value(argv, "--allowedTools") == "WebSearch,WebFetch"
    # no other built-in tool, and never an unattended bypass of permissions
    for forbidden in ("Bash", "Read", "Write", "Edit"):
        assert forbidden not in _flag_value(argv, "--tools").split(",")
    assert "--dangerously-skip-permissions" not in argv
    assert "--allow-dangerously-skip-permissions" not in argv


async def test_web_tools_enabled_still_disallows_mcp_and_prompts(
    runner_with_fake_process, web_prompt_file
):
    envelope = {"is_error": False, "result": "OK"}
    process = FakeProcess(stdout=json.dumps(envelope).encode())
    argv = runner_with_fake_process(process)

    runner = ClaudeRunner(_settings(CLAUDE_ENABLE_WEB_TOOLS="true", CLAUDE_WEB_PROMPT_FILE=web_prompt_file))
    await runner.run(prompt="hi", model="sonnet", json_schema=None)

    assert _flag_value(argv, "--disallowedTools") == "mcp__*"
    assert _flag_value(argv, "--permission-prompts") == "none"
    assert _flag_value(argv, "--setting-sources") == ""
    assert "--disable-slash-commands" in argv
    assert "--no-session-persistence" in argv


async def test_web_tools_enabled_still_compatible_with_json_schema(
    runner_with_fake_process, web_prompt_file
):
    schema = {"type": "object", "properties": {"tags": {"type": "array"}}}
    envelope = {"is_error": False, "result": '{"tags":[]}', "structured_output": {"tags": []}}
    process = FakeProcess(stdout=json.dumps(envelope).encode())
    argv = runner_with_fake_process(process)

    runner = ClaudeRunner(_settings(CLAUDE_ENABLE_WEB_TOOLS="true", CLAUDE_WEB_PROMPT_FILE=web_prompt_file))
    result = await runner.run(prompt="hi", model="sonnet", json_schema=schema)

    assert _flag_value(argv, "--json-schema") == json.dumps(schema)
    assert result.structured == {"tags": []}


async def test_web_tools_flag_does_not_alter_the_prompt_sent_to_claude(
    runner_with_fake_process, web_prompt_file
):
    """Enabling web tools must never rewrite, prefix, or otherwise touch the
    prompt -- only the tool-related argv flags change."""
    envelope = {"is_error": False, "result": "OK"}
    process = FakeProcess(stdout=json.dumps(envelope).encode())
    runner_with_fake_process(process)

    runner = ClaudeRunner(_settings(CLAUDE_ENABLE_WEB_TOOLS="true", CLAUDE_WEB_PROMPT_FILE=web_prompt_file))
    await runner.run(prompt="Résume cet article en français.", model="sonnet", json_schema=None)

    assert process.stdin.written == "Résume cet article en français.".encode("utf-8")


async def test_web_tools_enabled_appends_prompt_file_content_via_argv(
    runner_with_fake_process, web_prompt_file
):
    """When web tools are on, the search-guidance instruction is read from
    CLAUDE_WEB_PROMPT_FILE and sent via --append-system-prompt in argv --
    never through stdin/the prompt."""
    envelope = {"is_error": False, "result": "OK"}
    process = FakeProcess(stdout=json.dumps(envelope).encode())
    argv = runner_with_fake_process(process)

    runner = ClaudeRunner(_settings(CLAUDE_ENABLE_WEB_TOOLS="true", CLAUDE_WEB_PROMPT_FILE=web_prompt_file))
    await runner.run(prompt="hi", model="sonnet", json_schema=None)

    assert _flag_value(argv, "--append-system-prompt") == WEB_PROMPT_TEST_CONTENT
    assert _flag_value(argv, "--tools") == "WebSearch,WebFetch"
    assert _flag_value(argv, "--allowedTools") == "WebSearch,WebFetch"
    # the instruction travels via argv, never mixed into the stdin prompt
    assert process.stdin.written == b"hi"


async def test_web_tools_enabled_rereads_prompt_file_on_every_request(
    runner_with_fake_process, tmp_path
):
    """No caching: a change to the file between two requests must be picked
    up on the very next request, with no proxy restart."""
    path = tmp_path / "web-tools.txt"
    path.write_text("Version A of the instruction.", encoding="utf-8")
    settings = _settings(CLAUDE_ENABLE_WEB_TOOLS="true", CLAUDE_WEB_PROMPT_FILE=str(path))
    runner = ClaudeRunner(settings)
    envelope = {"is_error": False, "result": "OK"}

    process1 = FakeProcess(stdout=json.dumps(envelope).encode())
    argv1 = runner_with_fake_process(process1)
    await runner.run(prompt="hi", model="sonnet", json_schema=None)
    assert _flag_value(argv1, "--append-system-prompt") == "Version A of the instruction."

    path.write_text("Version B of the instruction, edited.", encoding="utf-8")
    process2 = FakeProcess(stdout=json.dumps(envelope).encode())
    argv2 = runner_with_fake_process(process2)
    await runner.run(prompt="hi", model="sonnet", json_schema=None)
    assert _flag_value(argv2, "--append-system-prompt") == "Version B of the instruction, edited."


async def test_web_tools_disabled_never_appends_search_instruction(runner_with_fake_process):
    envelope = {"is_error": False, "result": "OK"}
    process = FakeProcess(stdout=json.dumps(envelope).encode())
    argv = runner_with_fake_process(process)

    runner = ClaudeRunner(_settings(CLAUDE_ENABLE_WEB_TOOLS="false"))
    await runner.run(prompt="hi", model="sonnet", json_schema=None)

    assert "--append-system-prompt" not in argv
    assert _flag_value(argv, "--tools") == ""
    assert "--allowedTools" not in argv


async def test_web_tools_disabled_never_reads_prompt_file_even_if_missing(
    runner_with_fake_process, tmp_path
):
    """Disabled mode must not touch CLAUDE_WEB_PROMPT_FILE at all -- not even
    to check it exists -- so a bad/missing path is harmless when disabled."""
    bad_path = tmp_path / "does-not-exist.txt"
    envelope = {"is_error": False, "result": "OK"}
    process = FakeProcess(stdout=json.dumps(envelope).encode())
    runner_with_fake_process(process)

    runner = ClaudeRunner(
        _settings(CLAUDE_ENABLE_WEB_TOOLS="false", CLAUDE_WEB_PROMPT_FILE=str(bad_path))
    )
    result = await runner.run(prompt="hi", model="sonnet", json_schema=None)
    assert result.text == "OK"
    assert not bad_path.exists()


async def test_web_tools_default_absent_also_skips_search_instruction(runner_with_fake_process):
    """Variable absent entirely (not just 'false') behaves identically."""
    envelope = {"is_error": False, "result": "OK"}
    process = FakeProcess(stdout=json.dumps(envelope).encode())
    argv = runner_with_fake_process(process)

    runner = ClaudeRunner(_settings())  # CLAUDE_ENABLE_WEB_TOOLS not set at all
    await runner.run(prompt="hi", model="sonnet", json_schema=None)

    assert "--append-system-prompt" not in argv
    assert _flag_value(argv, "--tools") == ""


async def test_web_prompt_file_missing_raises_controlled_error(tmp_path):
    missing = tmp_path / "missing.txt"
    with pytest.raises(WebPromptFileError):
        read_web_prompt_file(str(missing))


async def test_web_prompt_file_empty_raises_controlled_error(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(WebPromptFileError):
        read_web_prompt_file(str(empty))


async def test_web_prompt_file_enabled_with_missing_file_surfaces_as_request_error(
    runner_with_fake_process, tmp_path
):
    """If the file disappears after startup, a request in web mode must
    fail with a controlled error, not continue silently without guidance."""
    missing = tmp_path / "missing.txt"
    runner = ClaudeRunner(
        _settings(CLAUDE_ENABLE_WEB_TOOLS="true", CLAUDE_WEB_PROMPT_FILE=str(missing))
    )
    with pytest.raises(WebPromptFileError):
        await runner.run(prompt="hi", model="sonnet", json_schema=None)


async def test_invocation_log_reflects_web_tools_mode_without_leaking_prompt(
    runner_with_fake_process, web_prompt_file, caplog
):
    """The per-invocation INFO log must state the configured mode, and must
    never contain the request's prompt content."""
    import logging

    envelope = {"is_error": False, "result": "OK"}

    for enabled, secret_prompt in (
        ("true", "Confidential bookmark about a merger, keyword ZXQ7secret"),
        ("false", "Another confidential bookmark, keyword PLAINMODEsecret"),
    ):
        process = FakeProcess(stdout=json.dumps(envelope).encode())
        runner_with_fake_process(process)
        runner = ClaudeRunner(
            _settings(CLAUDE_ENABLE_WEB_TOOLS=enabled, CLAUDE_WEB_PROMPT_FILE=web_prompt_file)
        )

        with caplog.at_level(logging.INFO, logger="app.claude_runner"):
            await runner.run(prompt=secret_prompt, model="sonnet", json_schema=None)

        invocation_logs = [r.message for r in caplog.records if "Claude invocation" in r.message]
        assert len(invocation_logs) == 1
        assert f"web_tools_enabled={enabled == 'true'}" in invocation_logs[0]
        for record in caplog.records:
            assert secret_prompt not in record.getMessage()
            assert WEB_PROMPT_TEST_CONTENT not in record.getMessage()
        caplog.clear()


async def test_structured_output_is_parsed(runner_with_fake_process):
    envelope = {"is_error": False, "result": '{"status":"ok"}', "structured_output": {"status": "ok"}}
    process = FakeProcess(stdout=json.dumps(envelope).encode())
    runner_with_fake_process(process)

    runner = ClaudeRunner(_settings())
    result = await runner.run(prompt="hi", model="sonnet", json_schema={"type": "object"})

    assert result.structured == {"status": "ok"}


async def test_rate_limit_error_detected_via_status_code(runner_with_fake_process):
    envelope = {"is_error": True, "result": "overloaded", "api_error_status": 429}
    process = FakeProcess(stdout=json.dumps(envelope).encode(), returncode=1)
    runner_with_fake_process(process)

    runner = ClaudeRunner(_settings())
    with pytest.raises(ClaudeUpstreamError) as exc_info:
        await runner.run(prompt="hi", model="sonnet", json_schema=None)
    assert exc_info.value.is_rate_limited is True
    assert exc_info.value.api_error_status == 429


async def test_rate_limit_error_detected_via_message_keyword(runner_with_fake_process):
    envelope = {"is_error": True, "result": "You have hit your usage limit for today", "api_error_status": None}
    process = FakeProcess(stdout=json.dumps(envelope).encode(), returncode=1)
    runner_with_fake_process(process)

    runner = ClaudeRunner(_settings())
    with pytest.raises(ClaudeUpstreamError) as exc_info:
        await runner.run(prompt="hi", model="sonnet", json_schema=None)
    assert exc_info.value.is_rate_limited is True


async def test_generic_upstream_error_is_not_rate_limited(runner_with_fake_process):
    envelope = {"is_error": True, "result": "model not found", "api_error_status": 404}
    process = FakeProcess(stdout=json.dumps(envelope).encode(), returncode=1)
    runner_with_fake_process(process)

    runner = ClaudeRunner(_settings())
    with pytest.raises(ClaudeUpstreamError) as exc_info:
        await runner.run(prompt="hi", model="sonnet", json_schema=None)
    assert exc_info.value.is_rate_limited is False
    assert exc_info.value.api_error_status == 404


async def test_invalid_json_stdout_raises(runner_with_fake_process):
    process = FakeProcess(stdout=b"not json at all", returncode=0)
    runner_with_fake_process(process)

    runner = ClaudeRunner(_settings())
    with pytest.raises(ClaudeInvalidOutputError):
        await runner.run(prompt="hi", model="sonnet", json_schema=None)


async def test_missing_result_field_raises(runner_with_fake_process):
    envelope = {"is_error": False, "session_id": "abc"}
    process = FakeProcess(stdout=json.dumps(envelope).encode())
    runner_with_fake_process(process)

    runner = ClaudeRunner(_settings())
    with pytest.raises(ClaudeInvalidOutputError):
        await runner.run(prompt="hi", model="sonnet", json_schema=None)


async def test_empty_stdout_raises(runner_with_fake_process):
    process = FakeProcess(stdout=b"", returncode=1)
    runner_with_fake_process(process)

    runner = ClaudeRunner(_settings())
    with pytest.raises(ClaudeInvalidOutputError):
        await runner.run(prompt="hi", model="sonnet", json_schema=None)


async def test_executable_not_found(monkeypatch):
    monkeypatch.setattr("app.claude_runner.claude_binary_path", lambda settings: None)
    runner = ClaudeRunner(_settings())
    with pytest.raises(ClaudeExecutableNotFoundError):
        await runner.run(prompt="hi", model="sonnet", json_schema=None)


async def test_timeout_kills_hanging_process(runner_with_fake_process):
    process = FakeProcess(stdout=b"", returncode=None, hang_seconds=5)
    runner_with_fake_process(process)

    runner = ClaudeRunner(_settings())
    with pytest.raises(ClaudeTimeoutError):
        await runner.run(prompt="hi", model="sonnet", json_schema=None, timeout_sec=0.05)
    assert process.killed is True


async def test_read_capped_enforces_byte_limit():
    reader = _reader(b"x" * 1000)
    data = await _read_capped(reader, cap=100)
    assert len(data) == 100


def test_no_shell_true_anywhere_in_runner_source():
    source = Path("app/claude_runner.py").read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "create_subprocess_shell" not in source
    assert "os.system" not in source


def test_prompt_is_never_interpolated_into_argv():
    source = Path("app/claude_runner.py").read_text(encoding="utf-8")
    # the prompt must only ever reach Claude via stdin.write(...), never via argv/f-strings
    assert re.search(r'argv\s*\+?=\s*\[[^\]]*\bprompt\b', source) is None
    assert "proc.stdin.write(prompt" in source
