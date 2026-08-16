from __future__ import annotations

from sentinelml.features.engineering import FEATURE_NAMES, FeatureState, vectorize
from sentinelml.features.store import FeatureStore, InMemoryOnlineStore


def test_vectorize_length() -> None:
    feats = {n: 1.0 for n in FEATURE_NAMES}
    assert vectorize(feats).shape == (len(FEATURE_NAMES),)


def test_online_store_roundtrip() -> None:
    store = FeatureStore(online=InMemoryOnlineStore())
    feats = {n: float(i) for i, n in enumerate(FEATURE_NAMES)}
    store.materialize_row({"customer_id": "C", "merchant_id": "M"}, feats)
    got = store.get_online("C", "M")
    assert got == feats


def test_consistency_on_repeated_state() -> None:
    state = FeatureState(label_delay_days=7)
    row = {
        "event_time": __import__("datetime").datetime(
            2025, 1, 2, tzinfo=__import__("datetime").timezone.utc
        ),
        "customer_id": "C1",
        "merchant_id": "M1",
        "amount": 10.0,
        "is_fraud": 0,
    }
    a = state.extract(row)
    b = state.extract(row)
    assert a == b
