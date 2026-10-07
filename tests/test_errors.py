from __future__ import annotations

from app.claude_runner import (
    ClaudeExecutableNotFoundError,
    ClaudeInvalidOutputError,
    ClaudeTimeoutError,
    ClaudeUpstreamError,
)


def _post(client, auth_headers, extra=None):
    payload = {"model": "claude-sonnet", "messages": [{"role": "user", "content": "hi"}]}
    if extra:
        payload.update(extra)
    return client.post("/v1/chat/completions", headers=auth_headers, json=payload)


def test_claude_timeout_maps_to_504(client, fake_runner, auth_headers):
    fake_runner.next_exception = ClaudeTimeoutError("timed out")
    resp = _post(client, auth_headers)
    assert resp.status_code == 504
    assert resp.json()["error"]["code"] == "claude_timeout"


def test_claude_executable_missing_maps_to_503(client, fake_runner, auth_headers):
    fake_runner.next_exception = ClaudeExecutableNotFoundError("not found")
    resp = _post(client, auth_headers)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "service_not_ready"


def test_claude_rate_limit_maps_to_429(client, fake_runner, auth_headers):
    fake_runner.next_exception = ClaudeUpstreamError("quota exceeded", 429, True)
    resp = _post(client, auth_headers)
    assert resp.status_code == 429
    assert resp.json()["error"]["code"] == "claude_rate_limited"


def test_claude_generic_upstream_error_maps_to_502(client, fake_runner, auth_headers):
    fake_runner.next_exception = ClaudeUpstreamError("boom", 500, False)
    resp = _post(client, auth_headers)
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "claude_error"


def test_claude_invalid_output_maps_to_502_without_leaking_raw_stdout(client, fake_runner, auth_headers):
    fake_runner.next_exception = ClaudeInvalidOutputError("bad json", raw="<<<secret internal detail>>>")
    resp = _post(client, auth_headers)
    assert resp.status_code == 502
    assert "secret internal detail" not in resp.text


def test_no_python_traceback_leaks_to_client(client, fake_runner, auth_headers):
    fake_runner.next_exception = RuntimeError("unexpected bug")
    resp = _post(client, auth_headers)
    assert resp.status_code == 500
    assert "Traceback" not in resp.text
    assert "RuntimeError" not in resp.text


def test_payload_too_large_rejected(client, fake_runner, auth_headers):
    huge = "x" * (3 * 1024 * 1024)
    resp = _post(client, auth_headers, extra={"messages": [{"role": "user", "content": huge}]})
    assert resp.status_code == 413


def test_invalid_json_body_rejected(client, fake_runner, auth_headers):
    resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        content=b"not json",
    )
    assert resp.status_code == 400


def test_missing_messages_field_rejected(client, fake_runner, auth_headers):
    resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"model": "claude-sonnet"},
    )
    assert resp.status_code == 400
