from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from sentinelml.data.schemas import TransactionEvent
from sentinelml.streaming.broker import Broker

TOPICS = {
    "raw": "transactions.raw",
    "enriched": "transactions.enriched",
    "predictions": "fraud.predictions",
    "alerts": "fraud.alerts",
    "feedback": "model.feedback",
}


class TransactionProducer:
    def __init__(self, broker: Broker, topic: str = TOPICS["raw"]) -> None:
        self.broker = broker
        self.topic = topic

    def send(self, event: TransactionEvent | dict[str, Any]) -> None:
        payload = event.model_dump(mode="json") if isinstance(event, TransactionEvent) else dict(event)
        payload.setdefault("schema_version", "1.0")
        self.broker.publish(self.topic, payload, key=str(payload.get("transaction_id")))

    def send_many(self, events: Iterable[TransactionEvent | dict[str, Any]]) -> int:
        n = 0
        for event in events:
            self.send(event)
            n += 1
        return n


def row_to_event(row: dict[str, Any]) -> dict[str, Any]:
    event_time = row["event_time"]
    if isinstance(event_time, datetime):
        event_time_s = event_time.isoformat()
    else:
        event_time_s = str(event_time)
    return {
        "schema_version": str(row.get("schema_version", "1.0")),
        "event_id": str(row["event_id"]),
        "transaction_id": str(row["transaction_id"]),
        "event_time": event_time_s,
        "customer_id": str(row["customer_id"]),
        "merchant_id": str(row["merchant_id"]),
        "amount": float(row["amount"]),
        "currency": str(row.get("currency", "USD")),
        "channel": str(row.get("channel", "card_present")),
        "is_fraud": int(row["is_fraud"]) if row.get("is_fraud") is not None else None,
    }
