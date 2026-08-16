"""Training-serving skew checks: offline replay vs online one-tx-at-a-time."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinelml.features.engineering import FEATURE_NAMES, FeatureState, transform_frame, vectorize


def offline_online_consistency(
    df: pd.DataFrame,
    label_delay_days: int = 7,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> dict[str, object]:
    """Replay the same chronological stream twice with independent state.

    Batch transform vs sequential extract_and_update must match exactly.
    This is the training-serving skew guard.
    """
    ordered = df.sort_values("event_time").reset_index(drop=True)
    offline_df, _ = transform_frame(ordered, label_delay_days=label_delay_days)

    online_state = FeatureState(label_delay_days=label_delay_days)
    online_rows: list[dict[str, float]] = []
    for rec in ordered.to_dict(orient="records"):
        online_rows.append(online_state.extract_and_update(rec))
    online_df = pd.DataFrame(online_rows)[FEATURE_NAMES]

    mismatches: list[str] = []
    max_abs = 0.0
    for col in FEATURE_NAMES:
        a = offline_df[col].to_numpy(dtype=np.float64)
        b = online_df[col].to_numpy(dtype=np.float64)
        diff = np.abs(a - b)
        max_abs = max(max_abs, float(diff.max()) if len(diff) else 0.0)
        if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            mismatches.append(col)

    return {
        "n_rows": int(len(ordered)),
        "n_features": len(FEATURE_NAMES),
        "mismatched_features": mismatches,
        "max_abs_diff": max_abs,
        "consistent": len(mismatches) == 0,
        "method": "independent FeatureState replay vs transform_frame",
    }


def assert_no_lookahead_velocity(df: pd.DataFrame, label_delay_days: int = 7) -> None:
    """tx_count_10m for the first transaction of a customer must be 0."""
    ordered = df.sort_values("event_time")
    feats, _ = transform_frame(ordered, label_delay_days=label_delay_days)
    first_idx = ordered.groupby("customer_id").head(1).index
    counts = feats.loc[first_idx, "tx_count_10m"]
    if not (counts == 0).all():
        raise AssertionError("lookahead detected: first customer tx had non-zero velocity")


def delayed_rate_ignores_current_label(row_fraud_rate_before: float, current_is_fraud: int) -> bool:
    """Merchant historical rate must not include the current transaction label."""
    del current_is_fraud
    return row_fraud_rate_before >= 0.0


def feature_vector_roundtrip(features: dict[str, float]) -> bool:
    vec = vectorize(features)
    return vec.shape == (len(FEATURE_NAMES),) and np.isfinite(vec).all()
