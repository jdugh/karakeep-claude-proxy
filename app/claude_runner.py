"""Invokes the Claude Code CLI (`claude -p`) as a locked-down subprocess.

Security invariants (see PROMPT.md section 3.3 / 13):
- the subprocess is never run through a shell; argv is always a list and the
  prompt is always sent via stdin, never interpolated into argv;
- by default Claude is given no tools at all (`--tools ""`). If
  `settings.claude_enable_web_tools` is set (`CLAUDE_ENABLE_WEB_TOOLS=true`),
  exactly `WebSearch` and `WebFetch` are made available and pre-authorized
  (`--tools "WebSearch,WebFetch" --allowedTools "WebSearch,WebFetch"`), so
  Claude can use them without an interactive permission prompt. The content
  of `settings.claude_web_prompt_file` (default
  `/app/prompts/web-tools.txt`, re-read fresh on every request, never
  cached) is appended via `--append-system-prompt` (argv, never shell/stdin)
  telling Claude when to use them -- this only nudges behavior, it grants
  nothing by itself. No other built-in tool (Bash, Read, Write, Edit, ...)
  is ever enabled this way, and `--dangerously-skip-permissions` is never
  used;
- MCP is always disabled (`--disallowedTools "mcp__*"`), regardless of the
  web-tools setting, as is project/user config (`--setting-sources ""`) and
  interactive permission prompts (`--permission-prompts none`);
- every invocation is bounded by a timeout and a hard cap on stdout/stderr
  size, and the subprocess is always killed/reaped, even on timeout or
  cancellation.
"""
from __future__ import annotations

import asyncio
import asyncio.subprocess
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, claude_binary_path

logger = logging.getLogger(__name__)

# A single structured JSON response line is expected; this just guards
# against a runaway/misbehaving process consuming unbounded memory.
MAX_STDOUT_BYTES = 10 * 1024 * 1024
MAX_STDERR_BYTES = 256 * 1024

_READ_CHUNK = 65536


class ClaudeRunnerError(Exception):
    """Base class for all ClaudeRunner failures."""


class ClaudeExecutableNotFoundError(ClaudeRunnerError):
    pass


class ClaudeTimeoutError(ClaudeRunnerError):
    pass


class ClaudeInvalidOutputError(ClaudeRunnerError):
    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


class ClaudeUpstreamError(ClaudeRunnerError):
    """Claude Code ran to completion but reported is_error=true."""

    def __init__(self, message: str, api_error_status: int | None, is_rate_limited: bool):
        super().__init__(message)
        self.api_error_status = api_error_status
        self.is_rate_limited = is_rate_limited


class WebPromptFileError(ClaudeRunnerError):
    """CLAUDE_WEB_PROMPT_FILE is missing, unreadable, or empty while
    CLAUDE_ENABLE_WEB_TOOLS=true. Raised both at startup (fail fast) and,
    if the file disappears later, on the affected request."""


def read_web_prompt_file(path: str) -> str:
    """Reads CLAUDE_WEB_PROMPT_FILE fresh -- no caching, so an edit to a
    mounted file takes effect on the very next request."""
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise WebPromptFileError(
            f"CLAUDE_ENABLE_WEB_TOOLS=true but CLAUDE_WEB_PROMPT_FILE {path!r} "
            f"could not be read: {exc}"
        ) from exc
    content = content.strip()
    if not content:
        raise WebPromptFileError(
            f"CLAUDE_ENABLE_WEB_TOOLS=true but CLAUDE_WEB_PROMPT_FILE {path!r} is empty"
        )
    return content


@dataclass
class ClaudeResult:
    text: str
    structured: dict[str, Any] | None
    session_id: str | None
    duration_ms: int | None


async def _read_capped(stream: asyncio.StreamReader, cap: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await stream.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > cap:
            overflow = total - cap
            chunks.append(chunk[: len(chunk) - overflow])
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _looks_rate_limited(message: str) -> bool:
    lowered = message.lower()
    return any(
        kw in lowered for kw in ("rate limit", "quota", "usage limit", "too many requests")
    )


class ClaudeRunner:
    def __init__(self, settings: Settings):
        self._settings = settings

    async def version(self) -> str | None:
        binary = claude_binary_path(self._settings)
        if not binary:
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                binary,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            return stdout.decode("utf-8", errors="replace").strip() or None
        except (OSError, asyncio.TimeoutError):
            return None

    async def run(
        self,
        *,
        prompt: str,
        model: str,
        json_schema: dict[str, Any] | None,
        timeout_sec: float | None = None,
    ) -> ClaudeResult:
        settings = self._settings
        binary = claude_binary_path(settings)
        if not binary:
            raise ClaudeExecutableNotFoundError(
                f"Claude Code executable {settings.claude_binary!r} not found on PATH"
            )

        timeout = timeout_sec if timeout_sec is not None else settings.claude_timeout_sec

        if settings.claude_enable_web_tools:
            web_prompt = read_web_prompt_file(settings.claude_web_prompt_file)
            tool_args = [
                "--tools", "WebSearch,WebFetch",
                "--allowedTools", "WebSearch,WebFetch",
                "--append-system-prompt", web_prompt,
            ]
        else:
            tool_args = ["--tools", ""]

        argv = [
            binary,
            "-p",
            "--output-format", "json",
            "--model", model,
            *tool_args,
            "--disallowedTools", "mcp__*",
            "--permission-prompts", "none",
            "--setting-sources", "",
            "--disable-slash-commands",
            "--no-session-persistence",
        ]
        if json_schema is not None:
            argv += ["--json-schema", json.dumps(json_schema)]

        logger.info("Claude invocation: web_tools_enabled=%s", settings.claude_enable_web_tools)

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def _communicate() -> tuple[bytes, bytes]:
            assert proc.stdin is not None
            try:
                proc.stdin.write(prompt.encode("utf-8"))
                await proc.stdin.drain()
            finally:
                proc.stdin.close()
            stdout, stderr = await asyncio.gather(
                _read_capped(proc.stdout, MAX_STDOUT_BYTES),
                _read_capped(proc.stderr, MAX_STDERR_BYTES),
            )
            await proc.wait()
            return stdout, stderr

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(_communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            await self._kill(proc)
            raise ClaudeTimeoutError(f"Claude Code did not finish within {timeout}s")
        except asyncio.CancelledError:
            await self._kill(proc)
            raise

        if proc.returncode is None:
            await self._kill(proc)

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")

        envelope = self._parse_envelope(stdout_text, stderr_text, proc.returncode)
        return self._to_result(envelope)

    @staticmethod
    async def _kill(proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            proc.kill()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            logger.warning("Claude Code process %s did not exit after kill", proc.pid)

    def _parse_envelope(
        self, stdout_text: str, stderr_text: str, returncode: int | None
    ) -> dict[str, Any]:
        stdout_text = stdout_text.strip()
        if not stdout_text:
            raise ClaudeInvalidOutputError(
                f"Claude Code produced no output (exit code {returncode})", raw=stderr_text
            )
        try:
            envelope = json.loads(stdout_text)
        except json.JSONDecodeError as exc:
            raise ClaudeInvalidOutputError(
                f"Claude Code returned non-JSON output: {exc}", raw=stdout_text
            ) from exc
        if not isinstance(envelope, dict):
            raise ClaudeInvalidOutputError(
                "Claude Code output was not a JSON object", raw=stdout_text
            )

        if envelope.get("is_error"):
            message = str(envelope.get("result") or "Claude Code reported an error")
            api_error_status = envelope.get("api_error_status")
            is_rate_limited = api_error_status == 429 or _looks_rate_limited(message)
            raise ClaudeUpstreamError(message, api_error_status, is_rate_limited)

        return envelope

    @staticmethod
    def _to_result(envelope: dict[str, Any]) -> ClaudeResult:
        result = envelope.get("result")
        if not isinstance(result, str):
            raise ClaudeInvalidOutputError(
                "Claude Code output is missing a textual 'result' field",
                raw=json.dumps(envelope)[:2000],
            )
        structured = envelope.get("structured_output")
        if structured is not None and not isinstance(structured, dict):
            structured = None
        return ClaudeResult(
            text=result,
            structured=structured,
            session_id=envelope.get("session_id"),
            duration_ms=envelope.get("duration_ms"),
        )
