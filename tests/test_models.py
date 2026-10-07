from __future__ import annotations

from app.config import ConfigError, load_settings


def test_models_endpoint_lists_configured_aliases(client, auth_headers):
    resp = client.get("/v1/models", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = {m["id"] for m in body["data"]}
    assert ids == {"claude-sonnet", "claude-haiku", "claude-opus"}
    for model in body["data"]:
        assert model["object"] == "model"


def test_models_endpoint_exposes_no_secrets(client, auth_headers):
    resp = client.get("/v1/models", headers=auth_headers)
    text = resp.text
    assert "test-oauth-token" not in text
    assert "test-proxy-key" not in text


def test_load_settings_honors_bool_and_int_overrides_from_the_given_dict():
    """Regression guard: load_settings(env_dict) must read booleans/ints from
    the given dict, not silently fall back to the real process environment
    (a bug this introduced, since _env_bool/_env_int used to ignore their
    `env` argument)."""
    settings = load_settings(
        {
            "MODEL_MAP": "a=sonnet",
            "DEFAULT_MODEL": "a",
            "CLAUDE_TIMEOUT_SEC": "42",
            "MAX_CONCURRENT_REQUESTS": "7",
            "ENABLE_TEST_CLAUDE_ENDPOINT": "true",
            "CLAUDE_ENABLE_WEB_TOOLS": "true",
        }
    )
    assert settings.claude_timeout_sec == 42
    assert settings.max_concurrent_requests == 7
    assert settings.enable_test_claude_endpoint is True
    assert settings.claude_enable_web_tools is True


def test_model_map_parsing():
    settings = load_settings(
        {
            "PROXY_API_KEY": "x",
            "CLAUDE_CODE_OAUTH_TOKEN": "y",
            "MODEL_MAP": "a=sonnet, b = opus ,c=haiku",
            "DEFAULT_MODEL": "a",
        }
    )
    assert settings.model_map == {"a": "sonnet", "b": "opus", "c": "haiku"}
    assert settings.resolve_model("b") == "opus"
    assert settings.resolve_model("unknown") is None


def test_model_map_rejects_bad_entry():
    try:
        load_settings({"MODEL_MAP": "noequalsign", "DEFAULT_MODEL": "a"})
    except ConfigError:
        pass
    else:
        raise AssertionError("expected ConfigError")


def test_default_model_must_be_in_map():
    try:
        load_settings({"MODEL_MAP": "a=sonnet", "DEFAULT_MODEL": "missing"})
    except ConfigError:
        pass
    else:
        raise AssertionError("expected ConfigError")


def test_unknown_model_alias_rejected(client, auth_headers):
    resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"
