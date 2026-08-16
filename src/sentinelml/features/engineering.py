"""Chronological feature state machine.

The same class is used for offline training replay and online inference.
Features for transaction t are computed from state accumulated strictly
before t. Merchant fraud rates use delayed labels (default 7 days).
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

FEATURE_NAMES: list[str] = [
    "amount",
    "hour",
    "day_of_week",
    "is_weekend",
    "tx_count_10m",
    "tx_count_1h",
    "tx_count_24h",
    "amount_sum_1h",
    "amount_sum_24h",
    "avg_amount_7d",
    "amount_vs_customer_mean",
    "amount_zscore_customer",
    "merchant_tx_count_24h",
    "merchant_fraud_rate_historical",
    "merchant_avg_amount",
    "new_merchant_for_customer",
    "time_since_previous_transaction",
    "merchant_frequency_customer",
]

WINDOW_10M = 10 * 60
WINDOW_1H = 3600
WINDOW_24H = 86400
WINDOW_7D = 7 * 86400


def _ts(value: datetime | pd.Timestamp) -> float:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _dt(value: datetime | pd.Timestamp) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


@dataclass
class _CustomerState:
    events: deque[tuple[float, float, str]] = field(default_factory=deque)  # ts, amount, merchant
    amount_sum: float = 0.0
    amount_sq: float = 0.0
    n: int = 0
    last_ts: float | None = None
    merchant_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class _MerchantState:
    events: deque[tuple[float, float]] = field(default_factory=deque)  # ts, amount
    amount_sum: float = 0.0
    n: int = 0
    delayed_n: int = 0
    delayed_fraud: int = 0


@dataclass
class _PendingLabel:
    available_ts: float
    merchant_id: str
    is_fraud: int


class FeatureState:
    """Online/offline identical feature extractor."""

    def __init__(self, label_delay_days: int = 7) -> None:
        self.label_delay = timedelta(days=label_delay_days)
        self.customers: dict[str, _CustomerState] = defaultdict(_CustomerState)
        self.merchants: dict[str, _MerchantState] = defaultdict(_MerchantState)
        self.pending: deque[_PendingLabel] = deque()

    def _expire_customer(self, state: _CustomerState, now_ts: float) -> None:
        cutoff = now_ts - WINDOW_7D
        while state.events and state.events[0][0] < cutoff:
            state.events.popleft()

    def _expire_merchant(self, state: _MerchantState, now_ts: float) -> None:
        cutoff = now_ts - WINDOW_24H
        while state.events and state.events[0][0] < cutoff:
            state.events.popleft()

    def _apply_due_labels(self, now_ts: float) -> None:
        while self.pending and self.pending[0].available_ts <= now_ts:
            item = self.pending.popleft()
            m = self.merchants[item.merchant_id]
            m.delayed_n += 1
            m.delayed_fraud += int(item.is_fraud)

    def extract(self, row: dict[str, object]) -> dict[str, float]:
        """Compute features WITHOUT mutating state with the current tx."""
        event_time = _dt(row["event_time"])  # type: ignore[arg-type]
        now_ts = _ts(event_time)
        self._apply_due_labels(now_ts)

        cid = str(row["customer_id"])
        mid = str(row["merchant_id"])
        amount = float(row["amount"])
        c = self.customers[cid]
        m = self.merchants[mid]
        self._expire_customer(c, now_ts)
        self._expire_merchant(m, now_ts)

        def count_since(seconds: int) -> int:
            cutoff = now_ts - seconds
            return sum(1 for ts, _, _ in c.events if ts >= cutoff)

        def sum_since(seconds: int) -> float:
            cutoff = now_ts - seconds
            return float(sum(amt for ts, amt, _ in c.events if ts >= cutoff))

        n_hist = c.n
        mean = (c.amount_sum / n_hist) if n_hist else amount
        var = (c.amount_sq / n_hist - mean * mean) if n_hist else 0.0
        std = float(np.sqrt(max(var, 1e-6)))
        z = (amount - mean) / std if n_hist else 0.0
        avg_7d = (
            float(sum(amt for _, amt, _ in c.events) / len(c.events)) if c.events else amount
        )
        merchant_freq = float(c.merchant_counts.get(mid, 0))
        new_merchant = 0.0 if merchant_freq > 0 else 1.0
        time_since = (now_ts - c.last_ts) if c.last_ts is not None else float(WINDOW_7D)
        merch_avg = (m.amount_sum / m.n) if m.n else amount
        merch_rate = (m.delayed_fraud / m.delayed_n) if m.delayed_n else 0.0

        return {
            "amount": amount,
            "hour": float(event_time.hour),
            "day_of_week": float(event_time.weekday()),
            "is_weekend": 1.0 if event_time.weekday() >= 5 else 0.0,
            "tx_count_10m": float(count_since(WINDOW_10M)),
            "tx_count_1h": float(count_since(WINDOW_1H)),
            "tx_count_24h": float(count_since(WINDOW_24H)),
            "amount_sum_1h": sum_since(WINDOW_1H),
            "amount_sum_24h": sum_since(WINDOW_24H),
            "avg_amount_7d": avg_7d,
            "amount_vs_customer_mean": amount / mean if mean else 1.0,
            "amount_zscore_customer": z,
            "merchant_tx_count_24h": float(len(m.events)),
            "merchant_fraud_rate_historical": float(merch_rate),
            "merchant_avg_amount": float(merch_avg),
            "new_merchant_for_customer": new_merchant,
            "time_since_previous_transaction": float(time_since),
            "merchant_frequency_customer": merchant_freq,
        }

    def update(self, row: dict[str, object]) -> None:
        """Record the current tx into state (called AFTER extract)."""
        event_time = _dt(row["event_time"])  # type: ignore[arg-type]
        now_ts = _ts(event_time)
        cid = str(row["customer_id"])
        mid = str(row["merchant_id"])
        amount = float(row["amount"])
        c = self.customers[cid]
        m = self.merchants[mid]
        c.events.append((now_ts, amount, mid))
        c.amount_sum += amount
        c.amount_sq += amount * amount
        c.n += 1
        c.last_ts = now_ts
        c.merchant_counts[mid] = c.merchant_counts.get(mid, 0) + 1
        m.events.append((now_ts, amount))
        m.amount_sum += amount
        m.n += 1
        label = int(row["is_fraud"]) if row.get("is_fraud") is not None else 0
        available = now_ts + self.label_delay.total_seconds()
        self.pending.append(_PendingLabel(available_ts=available, merchant_id=mid, is_fraud=label))

    def extract_and_update(self, row: dict[str, object]) -> dict[str, float]:
        feats = self.extract(row)
        self.update(row)
        return feats


def transform_frame(
    df: pd.DataFrame,
    state: FeatureState | None = None,
    label_delay_days: int = 7,
) -> tuple[pd.DataFrame, FeatureState]:
    state = state or FeatureState(label_delay_days=label_delay_days)
    ordered = df.sort_values("event_time")
    rows: list[dict[str, float]] = []
    index: list[object] = []
    for idx, rec in ordered.iterrows():
        feats = state.extract_and_update(rec.to_dict())
        rows.append(feats)
        index.append(idx)
    feat_df = pd.DataFrame(rows, index=index)[FEATURE_NAMES]
    overlap = [c for c in FEATURE_NAMES if c in ordered.columns]
    base = ordered.drop(columns=overlap)
    out = base.join(feat_df)
    return out.sort_index(), state


def features_from_iterable(
    rows: Iterable[dict[str, object]], state: FeatureState
) -> list[dict[str, float]]:
    return [state.extract_and_update(row) for row in rows]


def vectorize(features: dict[str, float]) -> np.ndarray:
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)


def matrix_from_frame(df: pd.DataFrame) -> np.ndarray:
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    return df[FEATURE_NAMES].to_numpy(dtype=np.float64)
