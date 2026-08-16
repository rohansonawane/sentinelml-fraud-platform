from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from sentinelml.api.app import create_app
from sentinelml.streaming.broker import InMemoryBroker
from sentinelml.streaming.processor import StreamProcessor
from sentinelml.streaming.producer import TOPICS, TransactionProducer, row_to_event


def test_stream_demo_all_processed(tiny_tx) -> None:
    events = [row_to_event(r) for r in tiny_tx.head(50).sort_values("event_time").to_dict(orient="records")]
    broker = InMemoryBroker()
    producer = TransactionProducer(broker)
    proc = StreamProcessor(broker)
    producer.send_many(events)
    proc.drain(max_messages=500)
    assert proc.processed == len(events)
    assert proc.errors == 0
    assert broker.qsize(TOPICS["raw"]) == 0
    assert broker.qsize(TOPICS["enriched"]) == len(events)


def test_malformed_json_increments_errors() -> None:
    broker = InMemoryBroker()
    proc = StreamProcessor(broker)
    broker.publish(TOPICS["raw"], {"nope": True})
    proc.drain()
    assert proc.errors == 1
    assert proc.processed == 0


def test_oversized_payload_rejected() -> None:
    app = create_app()
    with TestClient(app) as client:
        huge = "C" * 200
        body = {
            "event_id": "E",
            "transaction_id": "T",
            "event_time": datetime.now(timezone.utc).isoformat(),
            "customer_id": huge,
            "merchant_id": "M1",
            "amount": 10,
        }
        r = client.post("/api/v1/score", json=body)
        assert r.status_code in {413, 422, 503}


def test_health() -> None:
    app = create_app()
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
