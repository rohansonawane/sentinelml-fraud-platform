from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sentinelml.models.base import ensure_2d_proba


class DummyBaseline:
    name = "dummy_majority"
    version = "0.1.0"

    def __init__(self) -> None:
        self.model = DummyClassifier(strategy="most_frequent")

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(x)
        if proba.shape[1] == 1:
            # only one class seen
            if int(self.model.classes_[0]) == 1:
                return np.ones(len(x), dtype=np.float64)
            return np.zeros(len(x), dtype=np.float64)
        return ensure_2d_proba(proba)

    def get_metadata(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version, "framework": "sklearn"}


class AmountRuleBaseline:
    """Simple rule: flag if amount exceeds a high percentile of train amounts."""

    name = "rule_amount"
    version = "0.1.0"

    def __init__(self, quantile: float = 0.97) -> None:
        self.quantile = quantile
        self.threshold = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        del y
        amounts = x[:, 0]
        self.threshold = float(np.quantile(amounts, self.quantile))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        amounts = x[:, 0]
        # Smooth step around the amount threshold.
        scale = max(self.threshold, 1.0)
        return 1.0 / (1.0 + np.exp(-(amounts - self.threshold) / (0.15 * scale)))

    def get_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "framework": "rule",
            "amount_threshold": self.threshold,
        }


class LogisticModel:
    name = "logistic_regression"
    version = "0.1.0"

    def __init__(self, C: float = 1.0, max_iter: int = 400) -> None:
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return ensure_2d_proba(self.model.predict_proba(x))

    def get_metadata(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version, "framework": "sklearn"}
