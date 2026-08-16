from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from sentinelml.api.audit import AuditStore
from sentinelml.api.routes import router
from sentinelml.features.engineering import FeatureState
from sentinelml.features.store import FeatureStore, build_online_store
from sentinelml.inference.service import InferenceService
from sentinelml.models.decision import DecisionPolicy
from sentinelml.utils.config import get_settings
from sentinelml.utils.logging import bind_request_id, configure_logging, get_logger

log = get_logger("api.app")
REQUESTS = Counter("sentinelml_requests_total", "API requests", ["path", "method"])
LATENCY = Histogram("sentinelml_request_latency_seconds", "API latency")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    policy = DecisionPolicy(
        allow_max=settings.decision_allow_max, review_max=settings.decision_review_max
    )
    model_dir = Path(settings.model_dir)
    app.state.settings = settings
    app.state.inference = InferenceService(
        policy=policy,
        bundle_path=model_dir / "champion.joblib",
        preprocessor_path=model_dir / "preprocessor.joblib",
    )
    app.state.feature_state = FeatureState(label_delay_days=settings.feature_label_delay_days)
    online = build_online_store(settings.online_store_backend, settings.redis_url)
    app.state.feature_store = FeatureStore(
        online=online,
        feature_state=app.state.feature_state,
        label_delay_days=settings.feature_label_delay_days,
    )
    app.state.audit = AuditStore(settings.audit_database_url)
    app.state.processed = 0
    app.state.alerts = []
    log.info("api_started", model_ready=app.state.inference.ready)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="SentinelML",
        description="Real-time fraud detection inference API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3102", "http://127.0.0.1:3102"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        bind_request_id(request.headers.get("x-request-id"))
        REQUESTS.labels(path=request.url.path, method=request.method).inc()
        cl = request.headers.get("content-length")
        if cl and int(cl) > get_settings().max_body_bytes:
            return JSONResponse({"detail": "payload too large"}, status_code=413)
        with LATENCY.time():
            response = await call_next(request)
        return response

    @app.get("/metrics/prometheus")
    def prometheus_metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.exception_handler(HTTPException)
    async def http_exc(_, exc: HTTPException):
        return JSONResponse(
            {"error": exc.detail, "status": exc.status_code}, status_code=exc.status_code
        )

    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "sentinelml.api.app:app", host=settings.api_host, port=settings.api_port, reload=False
    )
