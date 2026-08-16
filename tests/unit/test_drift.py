from __future__ import annotations

import numpy as np
import pandas as pd

from sentinelml.monitoring.drift import feature_drift_report, population_stability_index


def test_psi_identical_is_near_zero() -> None:
    x = np.random.default_rng(0).normal(size=500)
    psi = population_stability_index(x, x)
    assert psi < 0.05


def test_drift_pipeline_detects_shift() -> None:
    rng = np.random.default_rng(1)
    ref = pd.DataFrame(
        {
            "amount": rng.normal(50, 10, 400),
            "hour": rng.integers(8, 20, 400).astype(float),
            "tx_count_1h": rng.integers(0, 3, 400).astype(float),
        }
    )
    cur = pd.DataFrame(
        {
            "amount": rng.normal(250, 20, 400),
            "hour": rng.integers(0, 5, 400).astype(float),
            "tx_count_1h": rng.integers(8, 20, 400).astype(float),
        }
    )
    report = feature_drift_report(ref, cur, features=["amount", "hour", "tx_count_1h"])
    assert report["drift_detected"] is True
    assert "amount" in report["alerted_features"]


def test_schema_alert() -> None:
    ref = pd.DataFrame({"amount": [1.0, 2.0], "hour": [1.0, 2.0]})
    cur = pd.DataFrame({"amount": [1.0, 2.0]})
    report = feature_drift_report(ref, cur, features=["amount", "hour"])
    assert report["drift_detected"] is True
