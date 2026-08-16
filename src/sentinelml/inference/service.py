from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sentinelml.features.engineering import FEATURE_NAMES, vectorize
from sentinelml.models.decision import DecisionPolicy, decide, risk_level
from sentinelml.training.calibration import ProbabilityCalibrator


class InferenceService:
    def __init__(
        self,
        bundle_path: str | Path = "artifacts/champion.joblib",
        policy: DecisionPolicy | None = None,
        preprocessor_path: str | Path = "artifacts/preprocessor.joblib",
    ) -> None:
        self.bundle_path = Path(bundle_path)
        self.policy = policy or DecisionPolicy()
        self.model: Any = None
        self.calibrator: ProbabilityCalibrator | None = None
        self.threshold = 0.5
        self.scaled = False
        self.scaler = None
        self.model_name = "unloaded"
        self.feature_names = FEATURE_NAMES
        self._load(preprocessor_path)

    def _load(self, preprocessor_path: str | Path) -> None:
        if not self.bundle_path.exists():
            return
        bundle = joblib.load(self.bundle_path)
        self.model = bundle["model"]
        self.calibrator = bundle.get("calibrator")
        self.threshold = float(bundle.get("threshold", 0.5))
        self.scaled = bool(bundle.get("scaled", False))
        self.model_name = str(bundle.get("name", self.model.name if hasattr(self.model, "name") else "unknown"))
        self.feature_names = list(bundle.get("feature_names", FEATURE_NAMES))
        pre = Path(preprocessor_path)
        if pre.exists():
            self.scaler = joblib.load(pre).get("scaler")

    @property
    def ready(self) -> bool:
        return self.model is not None

    def score_features(self, features: dict[str, float]) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("model bundle not loaded")
        t0 = time.perf_counter()
        vec = vectorize({k: float(features[k]) for k in self.feature_names}).reshape(1, -1)
        if self.scaled and self.scaler is not None:
            vec = self.scaler.transform(vec)
        proba = float(self.model.predict_proba(vec)[0])
        if self.calibrator is not None:
            proba = float(self.calibrator.transform(np.array([proba]))[0])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        decision = decide(proba, self.policy, threshold=self.threshold)
        return {
            "fraud_probability": proba,
            "risk_level": risk_level(proba, self.policy),
            "decision": decision,
            "threshold": self.threshold,
            "model_version": f"{self.model_name}:0.1.0",
            "latency_ms": latency_ms,
        }
