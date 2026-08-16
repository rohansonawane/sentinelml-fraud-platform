"""Feature contribution / model attribution, not causal proof.

SHAP is used for trees when the optional extra is installed. Otherwise tree
feature importances (gain/split counts) are returned.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sentinelml.features.engineering import FEATURE_NAMES

HAS_SHAP = False
try:
    import shap

    HAS_SHAP = True
except Exception:  # pragma: no cover
    shap = None


def tree_feature_contributions(model: Any, x_row: np.ndarray) -> dict[str, float]:
    """Return per-feature attribution for a single row.

    Label as feature contribution / model attribution, not causal proof.
    """
    x_row = np.asarray(x_row, dtype=np.float64).reshape(1, -1)
    if HAS_SHAP and shap is not None and hasattr(model, "predict"):
        try:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(x_row)
            if isinstance(values, list):
                values = values[1] if len(values) > 1 else values[0]
            arr = np.asarray(values).reshape(-1)
            return {FEATURE_NAMES[i]: float(arr[i]) for i in range(min(len(arr), len(FEATURE_NAMES)))}
        except Exception:
            pass
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=np.float64)
        return {FEATURE_NAMES[i]: float(imp[i]) for i in range(min(len(imp), len(FEATURE_NAMES)))}
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        contrib = coef * x_row.reshape(-1)[: len(coef)]
        return {FEATURE_NAMES[i]: float(contrib[i]) for i in range(min(len(contrib), len(FEATURE_NAMES)))}
    return {name: 0.0 for name in FEATURE_NAMES}


def attribution_disclaimer() -> str:
    return (
        "Values are feature contribution / model attribution, not causal proof. "
        "They explain this model's score, not why fraud occurred."
    )
