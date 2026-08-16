"""Isolation Forest anomaly detector.

Not claimed superior to supervised models. Scores are inverted so higher
means more anomalous / more fraud-like, then ranked to [0, 1] via a
training-set CDF (empirical).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestModel:
    name = "isolation_forest"
    version = "0.1.0"

    def __init__(self, n_estimators: int = 80, contamination: float = 0.03) -> None:
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=1,
        )
        self._ref: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> None:
        del y
        self.model.fit(x)
        raw = -self.model.score_samples(x)
        self._ref = np.sort(raw)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raw = -self.model.score_samples(x)
        if self._ref is None or len(self._ref) == 0:
            return np.clip(raw, 0.0, 1.0)
        # Empirical CDF against training anomaly scores.
        ranks = np.searchsorted(self._ref, raw, side="right") / len(self._ref)
        return np.clip(ranks.astype(np.float64), 0.0, 1.0)

    def get_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "framework": "sklearn",
            "note": "Anomaly detector. Not claimed superior to supervised models.",
        }
