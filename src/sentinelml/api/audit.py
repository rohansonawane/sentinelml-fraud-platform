"""SQLite audit of inferences/decisions. PostgreSQL + Alembic documented for prod."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class InferenceLog(Base):
    __tablename__ = "inference_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(128), index=True)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    customer_id: Mapped[str] = mapped_column(String(128), index=True)
    merchant_id: Mapped[str] = mapped_column(String(128))
    amount: Mapped[float] = mapped_column(Float)
    fraud_probability: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(32), index=True)
    risk_level: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(128))
    threshold: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)
    features_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class AuditStore:
    def __init__(self, url: str = "sqlite:///artifacts/audit.db") -> None:
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, future=True, connect_args=connect_args)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def record(self, payload: dict[str, Any]) -> None:
        row = InferenceLog(
            transaction_id=str(payload["transaction_id"]),
            event_time=_parse_dt(payload.get("event_time")),
            customer_id=str(payload.get("customer_id", "")),
            merchant_id=str(payload.get("merchant_id", "")),
            amount=float(payload.get("amount", 0.0)),
            fraud_probability=float(payload["fraud_probability"]),
            decision=str(payload["decision"]),
            risk_level=str(payload.get("risk_level", "")),
            model_version=str(payload.get("model_version", "")),
            threshold=float(payload.get("threshold", 0.0)),
            latency_ms=float(payload.get("latency_ms", 0.0)),
            features_json=__import__("json").dumps(payload.get("features") or {}, default=str),
        )
        with self.Session() as session:
            session.add(row)
            session.commit()

    def recent(self, limit: int = 100, decision: str | None = None) -> list[dict[str, Any]]:
        stmt = select(InferenceLog).order_by(InferenceLog.id.desc()).limit(limit)
        if decision:
            stmt = (
                select(InferenceLog)
                .where(InferenceLog.decision == decision)
                .order_by(InferenceLog.id.desc())
                .limit(limit)
            )
        with self.Session() as session:
            rows = session.scalars(stmt).all()
        return [_to_dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        with self.Session() as session:
            rows = session.scalars(select(InferenceLog)).all()
        if not rows:
            return {"n": 0, "alerts": 0, "avg_latency_ms": 0.0, "avg_probability": 0.0}
        n = len(rows)
        alerts = sum(1 for r in rows if r.decision in {"REVIEW", "BLOCK"})
        return {
            "n": n,
            "alerts": alerts,
            "avg_latency_ms": float(sum(r.latency_ms for r in rows) / n),
            "avg_probability": float(sum(r.fraud_probability for r in rows) / n),
            "allow": sum(1 for r in rows if r.decision == "ALLOW"),
            "review": sum(1 for r in rows if r.decision == "REVIEW"),
            "block": sum(1 for r in rows if r.decision == "BLOCK"),
        }


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if value is None:
        return datetime.now(timezone.utc)
    text = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _to_dict(row: InferenceLog) -> dict[str, Any]:
    import json

    return {
        "id": row.id,
        "transaction_id": row.transaction_id,
        "event_time": row.event_time.isoformat() if row.event_time else None,
        "customer_id": row.customer_id,
        "merchant_id": row.merchant_id,
        "amount": row.amount,
        "fraud_probability": row.fraud_probability,
        "decision": row.decision,
        "risk_level": row.risk_level,
        "model_version": row.model_version,
        "threshold": row.threshold,
        "latency_ms": row.latency_ms,
        "features": json.loads(row.features_json or "{}"),
    }
