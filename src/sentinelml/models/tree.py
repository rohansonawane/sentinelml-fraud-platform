from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from sentinelml.models.base import ensure_2d_proba

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover
    lgb = None
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except Exception:  # pragma: no cover
    xgb = None
    HAS_XGBOOST = False


class LightGBMModel:
    name = "lightgbm"
    version = "0.1.0"

    def __init__(
        self,
        n_estimators: int = 80,
        max_depth: int = 4,
        learning_rate: float = 0.08,
        num_leaves: int = 24,
        **kwargs: Any,
    ) -> None:
        if not HAS_LIGHTGBM:
            raise RuntimeError("lightgbm is not installed")
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            class_weight="balanced",
            verbosity=-1,
            n_jobs=1,
            **kwargs,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return ensure_2d_proba(self.model.predict_proba(x))

    def feature_importances(self) -> np.ndarray:
        return np.asarray(self.model.feature_importances_, dtype=np.float64)

    def get_metadata(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version, "framework": "lightgbm"}


class HistGBDTModel:
    name = "hist_gradient_boosting"
    version = "0.1.0"

    def __init__(
        self,
        max_iter: int = 80,
        max_depth: int = 4,
        learning_rate: float = 0.08,
        **kwargs: Any,
    ) -> None:
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            class_weight="balanced",
            **kwargs,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return ensure_2d_proba(self.model.predict_proba(x))

    def get_metadata(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version, "framework": "sklearn"}


class XGBoostModel:
    name = "xgboost"
    version = "0.1.0"

    def __init__(self, n_estimators: int = 80, max_depth: int = 4, learning_rate: float = 0.08) -> None:
        if not HAS_XGBOOST:
            raise RuntimeError("xgboost is not installed")
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            n_jobs=1,
            tree_method="hist",
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        # scale_pos_weight from class imbalance
        pos = max(int(y.sum()), 1)
        neg = max(int(len(y) - pos), 1)
        self.model.set_params(scale_pos_weight=neg / pos)
        self.model.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return ensure_2d_proba(self.model.predict_proba(x))

    def get_metadata(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version, "framework": "xgboost"}
