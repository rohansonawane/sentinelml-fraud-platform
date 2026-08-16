from __future__ import annotations

from datetime import datetime, timezone

from sentinelml.data.schemas import TransactionEvent
from sentinelml.features.engineering import FeatureState
from sentinelml.inference.service import InferenceService
from sentinelml.models.baseline import AmountRuleBaseline
from sentinelml.streaming.broker import InMemoryBroker
from sentinelml.streaming.processor import StreamProcessor
from sentinelml.streaming.producer import TOPICS, TransactionProducer
from sentinelml.training.calibration import ProbabilityCalibrator


def _toy_inference() -> InferenceService:
    svc = InferenceService.__new__(InferenceService)
    model = AmountRuleBaseline(quantile=0.8)
    import numpy as np

    x = np.array([[10.0] + [0.0] * 17, [500.0] + [0.0] * 17, [40.0] + [0.0] * 17])
    model.fit(x, np.array([0, 1, 0]))
    svc.model = model
    svc.calibrator = ProbabilityCalibrator("none")
    svc.threshold = 0.5
    svc.scaled = False
    svc.scaler = None
    svc.model_name = "rule_amount"
    from sentinelml.features.engineering import FEATURE_NAMES
    from sentinelml.models.decision import DecisionPolicy

    svc.feature_names = FEATURE_NAMES
    svc.policy = DecisionPolicy()
    svc.bundle_path = None  # type: ignore[assignment]
    return svc


def test_in_process_pipeline() -> None:
    broker = InMemoryBroker()
    producer = TransactionProducer(broker)
    inf = _toy_inference()
    proc = StreamProcessor(broker, state=FeatureState(label_delay_days=7), inference=inf)
    ev = TransactionEvent(
        event_id="E1",
        transaction_id="T1",
        event_time=datetime(2025, 6, 1, 12, tzinfo=timezone.utc),
        customer_id="C1",
        merchant_id="M1",
        amount=999.0,
        is_fraud=1,
    )
    producer.send(ev)
    n = proc.drain()
    assert n == 1
    assert proc.processed == 1
    preds = broker.peek_all(TOPICS["predictions"])
    assert len(preds) == 1
    assert "fraud_probability" in preds[0]
    assert preds[0]["decision"] in {"ALLOW", "REVIEW", "BLOCK"}
    assert "features" in preds[0]


def test_idempotent_publish() -> None:
    broker = InMemoryBroker()
    producer = TransactionProducer(broker)
    ev = {
        "schema_version": "1.0",
        "event_id": "E1",
        "transaction_id": "T1",
        "event_time": "2025-01-01T00:00:00+00:00",
        "customer_id": "C",
        "merchant_id": "M",
        "amount": 5.0,
    }
    producer.send(ev)
    producer.send(ev)
    assert broker.qsize(TOPICS["raw"]) == 1
