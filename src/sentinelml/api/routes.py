from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from sentinelml.data.schemas import MAX_AMOUNT, TransactionEvent
from sentinelml.features.engineering import FeatureState
from sentinelml.features.store import FeatureStore, build_online_store
from sentinelml.inference.service import InferenceService
from sentinelml.models.explain import attribution_disclaimer, tree_feature_contributions
from sentinelml.utils.logging import get_logger

log = get_logger("api")
router = APIRouter()


class ScoreRequest(BaseModel):
    schema_version: str = "1.0"
    event_id: str
    transaction_id: str
    event_time: str
    customer_id: str
    merchant_id: str
    amount: float = Field(gt=0, le=MAX_AMOUNT)
    currency: str = "USD"
    channel: str = "card_present"
    is_fraud: int | None = None
    features: dict[str, float] | None = None


class BatchScoreRequest(BaseModel):
    transactions: list[ScoreRequest] = Field(min_length=1, max_length=256)


def _svc(request: Request) -> InferenceService:
    return request.app.state.inference


def _state(request: Request) -> FeatureState:
    return request.app.state.feature_state


def _store(request: Request) -> FeatureStore:
    return request.app.state.feature_store


def _audit(request: Request):
    return request.app.state.audit


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request) -> dict[str, Any]:
    inf = _svc(request)
    redis_ok: bool | None = None
    try:
        redis_ok = request.app.state.feature_store.online.ping()
    except Exception:
        redis_ok = False
    if not inf.ready:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "model": inf.model_name, "online_store_ping": redis_ok}


@router.get("/metrics")
def metrics(request: Request) -> dict[str, Any]:
    stats = _audit(request).stats()
    return {"audit": stats, "processed": request.app.state.processed}


@router.get("/api/v1/model")
def model_info(request: Request) -> dict[str, Any]:
    inf = _svc(request)
    champion = _read_json("artifacts/champion.json")
    registry = _read_json("artifacts/registry.json")
    return {
        "loaded": inf.ready,
        "model_version": f"{inf.model_name}:0.1.0",
        "threshold": inf.threshold,
        "champion": champion,
        "registry": registry,
        "decision_policy": {
            "allow_max": inf.policy.allow_max,
            "review_max": inf.policy.review_max,
            "note": "Real blocking needs legal/business validation. Default REVIEW when ambiguous.",
        },
    }


@router.post("/api/v1/score")
def score(payload: ScoreRequest, request: Request) -> dict[str, Any]:
    inf = _svc(request)
    if not inf.ready:
        raise HTTPException(status_code=503, detail="model not loaded")
    row = payload.model_dump()
    try:
        event = TransactionEvent.model_validate({**row, "event_time": payload.event_time})
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"malformed transaction: {exc}") from exc
    features = payload.features or _state(request).extract_and_update(event.model_dump())
    _store(request).materialize_row(event.model_dump(), features)
    result = inf.score_features(features)
    inner = getattr(inf.model, "model", None)
    attribution = {}
    if inner is not None:
        try:
            from sentinelml.features.engineering import vectorize

            attribution = tree_feature_contributions(inner, vectorize(features))
        except Exception:
            attribution = {}
    body = {
        "transaction_id": payload.transaction_id,
        "customer_id": payload.customer_id,
        "merchant_id": payload.merchant_id,
        "amount": payload.amount,
        "event_time": payload.event_time,
        "features": features,
        "attribution": attribution,
        "attribution_note": attribution_disclaimer(),
        **result,
    }
    _audit(request).record(body)
    request.app.state.processed += 1
    if result["decision"] in {"REVIEW", "BLOCK"}:
        request.app.state.alerts.append(body)
    return body


@router.post("/api/v1/score/batch")
def score_batch(payload: BatchScoreRequest, request: Request) -> dict[str, Any]:
    if len(payload.transactions) > 256:
        raise HTTPException(status_code=413, detail="batch too large")
    results = [score(item, request) for item in payload.transactions]
    return {"n": len(results), "results": results}


@router.get("/api/v1/transactions")
def transactions(request: Request, limit: int = 100) -> dict[str, Any]:
    return {"items": _audit(request).recent(limit=limit)}


@router.get("/api/v1/alerts")
def alerts(request: Request, limit: int = 100) -> dict[str, Any]:
    items = _audit(request).recent(limit=limit, decision="REVIEW")
    items += _audit(request).recent(limit=limit, decision="BLOCK")
    return {"items": items[:limit]}


@router.get("/api/v1/stats")
def stats(request: Request) -> dict[str, Any]:
    inf = _svc(request)
    champion = _read_json("artifacts/champion.json")
    return {
        **_audit(request).stats(),
        "model": inf.model_name,
        "model_version": f"{inf.model_name}:0.1.0",
        "threshold": inf.threshold,
        "champion": champion,
        "processed_runtime": request.app.state.processed,
    }


@router.get("/api/v1/artifacts/{name}")
def artifacts(name: str) -> Any:
    allowed = {
        "metrics.json",
        "champion.json",
        "registry.json",
        "drift_report.json",
        "benchmark.json",
        "split_manifest.json",
        "environment.json",
        "stream_demo.json",
        "model_comparison.json",
        "test_results.json",
    }
    if name not in allowed:
        raise HTTPException(status_code=404, detail="unknown artifact")
    path = Path("artifacts") / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact not generated yet")
    return json.loads(path.read_text())


def _read_json(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def redis_status(url: str) -> dict[str, Any]:
    store = build_online_store("redis", url)
    from sentinelml.features.store import RedisOnlineStore

    if isinstance(store, RedisOnlineStore):
        return {"backend": "redis", "ok": store.ping()}
    return {"backend": "memory", "ok": True, "note": "Redis unavailable; using in-memory online store"}
