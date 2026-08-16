from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def precision_at_k(y: np.ndarray, p: np.ndarray, k: int) -> float:
    if k <= 0:
        return float("nan")
    k = min(k, len(y))
    idx = np.argsort(-p)[:k]
    return float(y[idx].mean())


def recall_at_k(y: np.ndarray, p: np.ndarray, k: int) -> float:
    total = float(y.sum())
    if total == 0 or k <= 0:
        return float("nan")
    k = min(k, len(y))
    idx = np.argsort(-p)[:k]
    return float(y[idx].sum() / total)


def expected_cost(
    y: np.ndarray,
    pred: np.ndarray,
    fp_cost: float = 1.0,
    fn_cost: float = 25.0,
) -> float:
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return float(fp * fp_cost + fn * fn_cost)


def classification_metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    fp_cost: float = 1.0,
    fn_cost: float = 25.0,
    ks: tuple[int, ...] = (50, 100, 200),
) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    metrics: dict[str, Any] = {
        "pr_auc": _safe_ap(y, p),
        "roc_auc": _safe_auc(y, p),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": fpr,
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else 0.0,
        "expected_cost": expected_cost(y, pred, fp_cost, fn_cost),
        "fp_cost_unit": fp_cost,
        "fn_cost_unit": fn_cost,
        "cost_note": "MODELING ASSUMPTIONS, not bank costs.",
    }
    for k in ks:
        if k < len(y):
            metrics[f"precision_at_{k}"] = precision_at_k(y, p, k)
            metrics[f"recall_at_{k}"] = recall_at_k(y, p, k)
    prec, rec, thr = precision_recall_curve(y, p)
    fpr_curve, tpr_curve, roc_thr = roc_curve(y, p)
    metrics["pr_curve"] = {
        "precision": prec.tolist()[:400],
        "recall": rec.tolist()[:400],
        "thresholds": thr.tolist()[:399],
    }
    metrics["roc_curve"] = {
        "fpr": fpr_curve.tolist()[:400],
        "tpr": tpr_curve.tolist()[:400],
        "thresholds": roc_thr.tolist()[:400],
    }
    return metrics


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, list[float]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    mean_p: list[float] = []
    mean_y: list[float] = []
    counts: list[float] = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            mean_p.append(float("nan"))
            mean_y.append(float("nan"))
            counts.append(0.0)
            continue
        mean_p.append(float(p[mask].mean()))
        mean_y.append(float(y[mask].mean()))
        counts.append(float(mask.sum()))
    return {"bin_pred": mean_p, "bin_true": mean_y, "bin_count": counts}
