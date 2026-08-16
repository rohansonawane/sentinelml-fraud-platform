from __future__ import annotations

import numpy as np

from sentinelml.models.baseline import DummyBaseline, LogisticModel
from sentinelml.training.metrics import classification_metrics
from sentinelml.training.selection import pick_champion


def test_metrics_do_not_use_accuracy_as_primary() -> None:
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9])
    m = classification_metrics(y, p, threshold=0.5, fp_cost=1, fn_cost=25)
    assert "pr_auc" in m and "roc_auc" in m
    assert "accuracy" not in m
    assert "expected_cost" in m
    assert m["confusion_matrix"]["tp"] >= 1


def test_dummy_majority_on_imbalance() -> None:
    x = np.zeros((20, 3))
    y = np.array([0] * 18 + [1] * 2)
    m = DummyBaseline()
    m.fit(x, y)
    p = m.predict_proba(x)
    assert (p == 0).all()


def test_logistic_trains() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 4))
    y = (x[:, 0] > 0.5).astype(int)
    m = LogisticModel(max_iter=200)
    m.fit(x, y)
    p = m.predict_proba(x)
    assert p.shape == (80,)
    assert 0 <= p.min() <= p.max() <= 1


def test_champion_not_auroc_only() -> None:
    rows = [
        {"model": "a", "pr_auc": 0.9, "recall": 0.1, "fpr": 0.01, "expected_cost": 100, "latency_ms_p50": 1, "rss_mb": 10, "roc_auc": 0.99},
        {"model": "b", "pr_auc": 0.7, "recall": 0.8, "fpr": 0.04, "expected_cost": 20, "latency_ms_p50": 2, "rss_mb": 12, "roc_auc": 0.8},
    ]
    weights = {
        "pr_auc": 0.35,
        "recall": 0.20,
        "fpr_inverted": 0.15,
        "cost_inverted": 0.20,
        "latency_inverted": 0.05,
        "memory_inverted": 0.05,
    }
    picked = pick_champion(rows, weights)
    assert "selection_score" in picked["champion"]
    assert "AUROC-only" not in picked["champion"]["why"] or "Not selected on AUROC" in picked["champion"]["why"]
