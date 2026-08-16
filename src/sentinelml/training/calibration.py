from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from sentinelml.training.metrics import classification_metrics, reliability_bins


class ProbabilityCalibrator:
    def __init__(self, method: str = "none") -> None:
        self.method = method
        self._iso: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> None:
        p = np.asarray(p, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y).astype(int)
        if self.method == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(p.ravel(), y)
        elif self.method == "platt":
            self._platt = LogisticRegression(max_iter=200)
            self._platt.fit(p, y)
        elif self.method != "none":
            raise ValueError(f"unknown calibration method {self.method}")

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if self.method == "none":
            return p
        if self.method == "isotonic" and self._iso is not None:
            return np.clip(self._iso.predict(p), 0.0, 1.0)
        if self.method == "platt" and self._platt is not None:
            return self._platt.predict_proba(p.reshape(-1, 1))[:, 1]
        return p


def choose_calibration(
    y_val: np.ndarray,
    p_val: np.ndarray,
    threshold: float,
    fp_cost: float,
    fn_cost: float,
    pr_auc_drop: float = 0.02,
    cost_increase_ratio: float = 0.15,
) -> dict[str, Any]:
    """Pick Platt/isotonic only if Brier improves without wrecking business metrics."""
    base = classification_metrics(y_val, p_val, threshold, fp_cost, fn_cost)
    reliability = reliability_bins(y_val, p_val)
    best_name = "none"
    best_p = p_val
    best_brier = base["brier"]
    results = {
        "none": {
            "brier": base["brier"],
            "pr_auc": base["pr_auc"],
            "expected_cost": base["expected_cost"],
        }
    }
    for method in ("platt", "isotonic"):
        try:
            cal = ProbabilityCalibrator(method)
            cal.fit(p_val, y_val)
            p2 = cal.transform(p_val)
            m = classification_metrics(y_val, p2, threshold, fp_cost, fn_cost)
        except Exception:
            results[method] = {"error": "fit_failed"}
            continue
        results[method] = {
            "brier": m["brier"],
            "pr_auc": m["pr_auc"],
            "expected_cost": m["expected_cost"],
        }
        hurts_pr = (base["pr_auc"] - m["pr_auc"]) > pr_auc_drop
        hurts_cost = m["expected_cost"] > base["expected_cost"] * (1 + cost_increase_ratio)
        if m["brier"] < best_brier and not hurts_pr and not hurts_cost:
            best_brier = m["brier"]
            best_name = method
            best_p = p2
    chosen = ProbabilityCalibrator(best_name)
    chosen.fit(p_val, y_val)
    return {
        "method": best_name,
        "calibrator": chosen,
        "val_comparison": results,
        "reliability_uncalibrated": reliability,
        "reliability_chosen": reliability_bins(y_val, best_p),
        "note": (
            "Calibration applied only if it improved validation Brier without a "
            f">{pr_auc_drop} PR-AUC drop or >{cost_increase_ratio:.0%} cost increase."
        ),
    }
