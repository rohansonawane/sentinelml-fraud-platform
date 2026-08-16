from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"
MAX_AMOUNT = 1_000_000.0
MAX_STRING = 128


class TransactionEvent(BaseModel):
    """Canonical streaming / storage schema for a transaction event."""

    schema_version: str = SCHEMA_VERSION
    event_id: str
    transaction_id: str
    event_time: datetime
    customer_id: str
    merchant_id: str
    amount: float = Field(gt=0, le=MAX_AMOUNT)
    currency: str = "USD"
    channel: str = "card_present"
    is_fraud: int | None = None

    @field_validator("customer_id", "merchant_id", "event_id", "transaction_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or len(value) > MAX_STRING:
            raise ValueError("identifier missing or too long")
        return value

    def idempotency_key(self) -> str:
        return f"{self.schema_version}:{self.transaction_id}"


class EnrichedTransaction(TransactionEvent):
    features: dict[str, float] = Field(default_factory=dict)


class PredictionEvent(BaseModel):
    schema_version: str = SCHEMA_VERSION
    event_id: str
    transaction_id: str
    event_time: datetime
    customer_id: str
    merchant_id: str
    amount: float
    fraud_probability: float
    risk_level: str
    decision: str
    threshold: float
    model_version: str
    latency_ms: float
    features: dict[str, float] = Field(default_factory=dict)
    attribution: dict[str, float] = Field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
