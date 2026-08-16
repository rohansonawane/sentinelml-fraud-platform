from __future__ import annotations

from datetime import datetime
from typing import Any

from sentinelml.data.schemas import TransactionEvent
from sentinelml.features.engineering import FeatureState
from sentinelml.features.store import FeatureStore
from sentinelml.inference.service import InferenceService
from sentinelml.models.explain import attribution_disclaimer, tree_feature_contributions
from sentinelml.streaming.broker import Broker
from sentinelml.streaming.producer import TOPICS
from sentinelml.utils.logging import get_logger

log = get_logger("stream_processor")


class StreamProcessor:
    def __init__(
        self,
        broker: Broker,
        state: FeatureState | None = None,
        store: FeatureStore | None = None,
        inference: InferenceService | None = None,
        label_delay_days: int = 7,
    ) -> None:
        self.broker = broker
        self.state = state or FeatureState(label_delay_days=label_delay_days)
        self.store = store or FeatureStore(
            feature_state=self.state, label_delay_days=label_delay_days
        )
        self.inference = inference
        self.processed = 0
        self.alerts = 0
        self.errors = 0
        self.last_latencies: list[float] = []

    def process_event(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        try:
            event = TransactionEvent.model_validate(raw)
        except Exception:
            self.errors += 1
            log.warning("malformed_event")
            return None
        row = event.model_dump()
        features = self.state.extract_and_update(row)
        self.store.materialize_row(row, features)
        enriched = {**row, "features": features, "event_time": event.event_time.isoformat()}
        self.broker.publish(TOPICS["enriched"], enriched, key=event.transaction_id)

        if self.inference is None or not self.inference.ready:
            self.processed += 1
            return enriched

        scored = self.inference.score_features(features)
        attribution: dict[str, float] = {}
        inner = getattr(self.inference.model, "model", None)
        if inner is not None:
            try:
                from sentinelml.features.engineering import vectorize

                attribution = tree_feature_contributions(inner, vectorize(features))
            except Exception:
                attribution = {}
        pred = {
            "schema_version": "1.0",
            "event_id": f"P-{event.event_id}",
            "transaction_id": event.transaction_id,
            "event_time": event.event_time.isoformat()
            if isinstance(event.event_time, datetime)
            else str(event.event_time),
            "customer_id": event.customer_id,
            "merchant_id": event.merchant_id,
            "amount": event.amount,
            "features": features,
            "attribution": attribution,
            "attribution_note": attribution_disclaimer(),
            "is_fraud": event.is_fraud,
            **scored,
        }
        self.broker.publish(TOPICS["predictions"], pred, key=event.transaction_id)
        self.last_latencies.append(float(scored["latency_ms"]))
        if scored["decision"] in {"REVIEW", "BLOCK"}:
            self.broker.publish(TOPICS["alerts"], pred, key=event.transaction_id)
            self.alerts += 1
        self.processed += 1
        return pred

    def drain(self, topic: str = TOPICS["raw"], max_messages: int = 10_000) -> int:
        n = 0
        batch = self.broker.consume(topic, max_messages=max_messages)
        for raw in batch:
            self.process_event(raw)
            n += 1
        return n
