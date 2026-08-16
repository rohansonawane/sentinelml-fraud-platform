from __future__ import annotations

import numpy as np

from sentinelml.models.decision import DecisionPolicy, apply_binary_threshold, decide, risk_level
from sentinelml.training.thresholds import lock_thresholds, maximize_f1, maximize_recall_at_max_fpr


def test_risk_bands() -> None:
    policy = DecisionPolicy(allow_max=0.2, review_max=0.7)
    assert risk_level(0.05, policy) == "low"
    assert risk_level(0.4, policy) == "medium"
    assert risk_level(0.9, policy) == "high"
    assert decide(0.05, policy) == "ALLOW"
    assert decide(0.4, policy) == "REVIEW"
    assert decide(0.9, policy) == "BLOCK"


def test_ambiguous_defaults_to_review() -> None:
    policy = DecisionPolicy(allow_max=0.2, review_max=0.7)
    assert decide(0.50, policy, threshold=0.50) == "REVIEW"
    assert decide(float("nan"), policy) == "REVIEW"


def test_binary_threshold() -> None:
    assert apply_binary_threshold(0.8, 0.5) == 1
    assert apply_binary_threshold(0.2, 0.5) == 0


def test_thresholds_locked_from_val_only() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    p = np.clip(y * 0.7 + rng.random(10) * 0.3, 0, 1)
    locked = lock_thresholds(y, p, max_fpr=0.5, fp_cost=1, fn_cost=25)
    assert set(locked) == {
        "maximize_f1",
        "maximize_recall_at_max_fpr",
        "minimize_configured_cost",
    }
    for v in locked.values():
        assert 0.0 <= v <= 1.0


def test_maximize_f1_separable() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    thr = maximize_f1(y, p)
    pred = (p >= thr).astype(int)
    assert pred.sum() >= 1


def test_max_fpr_constraint() -> None:
    y = np.array([0] * 90 + [1] * 10)
    p = np.linspace(0, 1, 100)
    thr = maximize_recall_at_max_fpr(y, p, max_fpr=0.05)
    pred = p >= thr
    fp = ((pred) & (y == 0)).sum()
    tn_fp = (y == 0).sum()
    assert fp / tn_fp <= 0.05 + 1e-9
