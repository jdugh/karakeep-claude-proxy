from __future__ import annotations

import json

from app.claude_runner import ClaudeResult


def test_json_schema_passes_schema_to_runner_and_returns_json_string(client, fake_runner, auth_headers):
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["tags"],
        "additionalProperties": False,
    }
    fake_runner.next_result = ClaudeResult(
        text='{"tags": ["docker", "ia"]}', structured={"tags": ["docker", "ia"]}, session_id="s", duration_ms=1
    )
    resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "claude-sonnet",
            "messages": [{"role": "user", "content": "Tag this"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "result", "strict": True, "schema": schema}},
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert json.loads(content) == {"tags": ["docker", "ia"]}
    assert fake_runner.calls[0]["json_schema"] == schema


def test_json_object_uses_generic_object_schema(client, fake_runner, auth_headers):
    fake_runner.next_result = ClaudeResult(
        text='{"status": "ok"}', structured={"status": "ok"}, session_id="s", duration_ms=1
    )
    resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "claude-sonnet",
            "messages": [{"role": "user", "content": "status"}],
            "response_format": {"type": "json_object"},
        },
    )
    assert resp.status_code == 200
    assert json.loads(resp.json()["choices"][0]["message"]["content"]) == {"status": "ok"}
    sent_schema = fake_runner.calls[0]["json_schema"]
    assert sent_schema["type"] == "object"


def test_json_schema_requires_schema_payload(client, fake_runner, auth_headers):
    resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "claude-sonnet",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema"},
        },
    )
    assert resp.status_code == 400
    assert fake_runner.calls == []


def test_structured_request_without_structured_output_is_upstream_error(client, fake_runner, auth_headers):
    fake_runner.next_result = ClaudeResult(text="not json", structured=None, session_id="s", duration_ms=1)
    resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "claude-sonnet",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "r", "schema": {"type": "object"}},
            },
        },
    )
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "upstream_error"


def test_plain_text_request_has_no_schema(client, fake_runner, auth_headers):
    resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"model": "claude-sonnet", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert fake_runner.calls[0]["json_schema"] is None
