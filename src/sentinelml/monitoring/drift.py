"""Deterministic PSI / JS drift detector (Evidently optional)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sentinelml.features.engineering import FEATURE_NAMES

try:
    import evidently  # noqa: F401

    HAS_EVIDENTLY = True
except Exception:  # pragma: no cover
    HAS_EVIDENTLY = False


def _hist(a: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(a, bins=bins)
    counts = counts.astype(np.float64) + 1e-6
    return counts / counts.sum()


def population_stability_index(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    qs = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(expected, qs))
    if len(bins) < 3:
        return 0.0
    e = _hist(expected, bins)
    a = _hist(actual, bins)
    return float(np.sum((a - e) * np.log(a / e)))


def jensen_shannon(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    qs = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(np.concatenate([expected, actual]), qs))
    if len(bins) < 3:
        return 0.0
    p = _hist(expected, bins)
    q = _hist(actual, bins)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return float(0.5 * (kl_pm + kl_qm))


def missing_rate(series: pd.Series) -> float:
    return float(series.isna().mean())


def schema_diff(expected_cols: list[str], actual_cols: list[str]) -> dict[str, list[str]]:
    exp, act = set(expected_cols), set(actual_cols)
    return {"missing_in_actual": sorted(exp - act), "unexpected": sorted(act - exp)}


def prediction_drift(ref_p: np.ndarray, cur_p: np.ndarray) -> dict[str, float]:
    return {
        "psi": population_stability_index(ref_p, cur_p),
        "js": jensen_shannon(ref_p, cur_p),
        "mean_ref": float(np.mean(ref_p)),
        "mean_cur": float(np.mean(cur_p)),
    }


def feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_alert: float = 0.2,
    js_alert: float = 0.1,
    features: list[str] | None = None,
) -> dict[str, Any]:
    features = features or [c for c in FEATURE_NAMES if c in reference.columns]
    per: dict[str, Any] = {}
    alerts: list[str] = []
    for col in features:
        if col not in current.columns:
            alerts.append(f"schema:{col}")
            continue
        ref = pd.to_numeric(reference[col], errors="coerce").dropna().to_numpy()
        cur = pd.to_numeric(current[col], errors="coerce").dropna().to_numpy()
        psi = population_stability_index(ref, cur) if len(ref) and len(cur) else 0.0
        js = jensen_shannon(ref, cur) if len(ref) and len(cur) else 0.0
        miss = missing_rate(current[col]) if col in current else 1.0
        per[col] = {
            "psi": psi,
            "js": js,
            "missing_rate": miss,
            "mean_ref": float(np.mean(ref)) if len(ref) else None,
            "mean_cur": float(np.mean(cur)) if len(cur) else None,
            "alert": bool(psi >= psi_alert or js >= js_alert or miss > 0.05),
        }
        if per[col]["alert"]:
            alerts.append(col)
    sch = schema_diff(list(reference.columns), list(current.columns))
    if sch["missing_in_actual"] or sch["unexpected"]:
        alerts.append("schema")
    return {
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "features": per,
        "schema": sch,
        "alerted_features": alerts,
        "drift_detected": len(alerts) > 0,
        "psi_alert_threshold": psi_alert,
        "js_alert_threshold": js_alert,
        "backend": "evidently" if HAS_EVIDENTLY else "psi_js",
        "note": "Evidently extra is optional; this report uses a deterministic PSI/JS detector.",
    }
