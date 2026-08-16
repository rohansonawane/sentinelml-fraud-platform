from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import optuna

    HAS_OPTUNA = True
except Exception:  # pragma: no cover
    optuna = None
    HAS_OPTUNA = False


def tune_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 8,
    timeout_seconds: int = 90,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score

    from sentinelml.models.tree import HAS_LIGHTGBM, LightGBMModel

    if not HAS_OPTUNA or not HAS_LIGHTGBM:
        return {"skipped": True, "reason": "optuna or lightgbm unavailable", "best_params": {}}

    metric_fn = metric_fn or (lambda y, p: float(average_precision_score(y, p)))

    def objective(trial: Any) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 40, 120),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 48),
        }
        model = LightGBMModel(**params)
        model.fit(x_train, y_train)
        p = model.predict_proba(x_val)
        return metric_fn(y_val, p)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)
    return {
        "skipped": False,
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
    }
