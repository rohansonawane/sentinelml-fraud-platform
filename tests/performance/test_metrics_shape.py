from __future__ import annotations

import numpy as np

from sentinelml.training.metrics import classification_metrics


def test_precision_at_k_defined() -> None:
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    m = classification_metrics(y, p, threshold=0.5, ks=(3,))
    assert m["precision_at_3"] == 1.0
    assert 0 <= m["recall_at_3"] <= 1
