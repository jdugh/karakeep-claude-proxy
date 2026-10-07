from __future__ import annotations


def test_models_rejects_missing_key(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "invalid_api_key"


def test_models_rejects_wrong_key(client):
    resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 401


def test_models_accepts_correct_key(client, auth_headers):
    resp = client.get("/v1/models", headers=auth_headers)
    assert resp.status_code == 200


def test_health_and_root_require_no_auth(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/readyz").status_code in (200, 503)
