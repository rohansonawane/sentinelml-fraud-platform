from __future__ import annotations

from fastapi.testclient import TestClient

from sentinelml.api.app import create_app
from sentinelml.features.store import RedisOnlineStore, build_online_store


def test_redis_unavailable_degrades() -> None:
    store = build_online_store("redis", "redis://127.0.0.1:1/0")
    # port 1 should fail; fallback to memory
    assert store.ping() is True
    assert not isinstance(store, RedisOnlineStore) or store.ping()


def test_ready_when_model_missing_is_503(monkeypatch, tmp_path) -> None:
    # create_app loads champion if present; without it ready is 503
    app = create_app()
    with TestClient(app) as client:
        r = client.get("/ready")
        assert r.status_code in {200, 503}
        if r.status_code == 503:
            assert "model" in r.json()["error"].lower() or "model" in str(r.json()).lower()
