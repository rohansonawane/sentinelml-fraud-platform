#!/usr/bin/env python3
"""Induce drift: amount scale, new merchants, activity pattern, fraud prevalence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentinelml.features.engineering import transform_frame
from sentinelml.monitoring.drift import feature_drift_report, prediction_drift
from sentinelml.utils.config import load_yaml


def main() -> None:
    cfg = load_yaml("configs/model.yaml")
    src = Path(cfg["data_path"])
    if not src.exists():
        raise SystemExit("generate data first")
    df = pd.read_parquet(src).sort_values("event_time").reset_index(drop=True)
    n = len(df)
    mid = n // 2
    reference = df.iloc[:mid].copy()
    current = df.iloc[mid:].copy()

    rng = np.random.default_rng(7)
    current = current.copy()
    current["amount"] = current["amount"] * rng.uniform(2.5, 6.0, len(current))
    current["merchant_id"] = [f"M_NEW_{i:05d}" for i in range(len(current))]
    # activity pattern: shift hours into night
    times = pd.to_datetime(current["event_time"], utc=True)
    current["event_time"] = [
        t.replace(hour=int(h))
        for t, h in zip(times, rng.integers(0, 5, len(current)), strict=True)
    ]
    # prevalence change
    current["is_fraud"] = (rng.random(len(current)) < 0.18).astype(int)

    ref_f, _ = transform_frame(reference, label_delay_days=int(cfg.get("label_delay_days", 7)))
    cur_f, _ = transform_frame(current, label_delay_days=int(cfg.get("label_delay_days", 7)))
    report = feature_drift_report(ref_f, cur_f)

    # prediction drift using amount as a crude score proxy if model missing
    pred = prediction_drift(ref_f["amount"].to_numpy(), cur_f["amount"].to_numpy())
    report["prediction_drift"] = pred
    report["induced"] = {
        "amount_scale": "x2.5-6.0",
        "new_merchants": True,
        "activity_hours": "shifted to 00:00-04:00",
        "fraud_prevalence": float(current["is_fraud"].mean()),
        "reference_prevalence": float(reference["is_fraud"].mean()),
    }
    report["performance_drift"] = {
        "note": "Labels exist on the simulator; prevalence shifted by construction.",
        "ref_prevalence": float(reference["is_fraud"].mean()),
        "cur_prevalence": float(current["is_fraud"].mean()),
        "alert": True,
    }
    if not report["drift_detected"]:
        raise SystemExit("drift demo failed to trigger detection")
    out = Path("artifacts/drift_report.json")
    out.write_text(json.dumps(report, indent=2, default=str))
    print(
        json.dumps(
            {
                "drift_detected": report["drift_detected"],
                "alerted_features": report["alerted_features"][:12],
                "n_reference": report["n_reference"],
                "n_current": report["n_current"],
                "backend": report["backend"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
