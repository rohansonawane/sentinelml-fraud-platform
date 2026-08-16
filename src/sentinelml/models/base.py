from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class Predictor(Protocol):
    name: str
    version: str

    def fit(self, x: np.ndarray, y: np.ndarray) -> None: ...
    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...
    def get_metadata(self) -> dict[str, Any]: ...


@dataclass
class ModelMetadata:
    name: str
    version: str
    framework: str
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def ensure_2d_proba(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        return np.clip(scores, 0.0, 1.0)
    if scores.shape[1] == 1:
        return np.clip(scores[:, 0], 0.0, 1.0)
    return np.clip(scores[:, 1], 0.0, 1.0)
