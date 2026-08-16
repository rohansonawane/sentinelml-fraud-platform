"""Thresholds are fit on VALIDATION only, then locked for test."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve

from sentinelml.training.metrics import expected_cost


def maximize_f1(y: np.ndarray, p: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, p)
    if len(thresholds) == 0:
        return 0.5
    f1 = (2 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1))])


def maximize_recall_at_max_fpr(y: np.ndarray, p: np.ndarray, max_fpr: float = 0.05) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(-p)
    y_sorted = y[order]
    fp = 0
    tp = 0
    n_neg = int((y == 0).sum())
    best_thr = 1.0
    best_recall = -1.0
    n_pos = max(int(y.sum()), 1)
    last_p = None
    for score, label in zip(p[order], y_sorted, strict=True):
        if label == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / n_neg if n_neg else 0.0
        if fpr > max_fpr:
            break
        recall = tp / n_pos
        if recall >= best_recall:
            best_recall = recall
            best_thr = float(score)
        last_p = score
    if last_p is None:
        return 0.5
    return best_thr


def minimize_configured_cost(
    y: np.ndarray,
    p: np.ndarray,
    fp_cost: float = 1.0,
    fn_cost: float = 25.0,
    grid: int = 101,
) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    best_thr = 0.5
    best_cost = float("inf")
    for thr in np.linspace(0.01, 0.99, grid):
        pred = (p >= thr).astype(int)
        cost = expected_cost(y, pred, fp_cost, fn_cost)
        if cost < best_cost:
            best_cost = cost
            best_thr = float(thr)
    return best_thr


def lock_thresholds(
    y_val: np.ndarray,
    p_val: np.ndarray,
    max_fpr: float = 0.05,
    fp_cost: float = 1.0,
    fn_cost: float = 25.0,
) -> dict[str, float]:
    locked = {
        "maximize_f1": maximize_f1(y_val, p_val),
        "maximize_recall_at_max_fpr": maximize_recall_at_max_fpr(y_val, p_val, max_fpr=max_fpr),
        "minimize_configured_cost": minimize_configured_cost(
            y_val, p_val, fp_cost=fp_cost, fn_cost=fn_cost
        ),
    }
    return locked


def f1_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    pred = (p >= threshold).astype(int)
    return float(f1_score(y, pred, zero_division=0))
