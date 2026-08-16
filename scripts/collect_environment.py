#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    import sentinelml

    disk = shutil.disk_usage("/")
    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "sentinelml_version": sentinelml.__version__,
        "cpu_count": os.cpu_count(),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "torch": _mod("torch"),
        "lightgbm": _mod("lightgbm"),
        "sklearn": _mod("sklearn"),
        "mlflow": _mod("mlflow"),
        "optuna": _mod("optuna"),
        "shap": _mod("shap"),
        "feast": _mod("feast"),
        "evidently": _mod("evidently"),
        "kafka": _mod("kafka"),
        "mps_available": _mps(),
        "cuda_available": False,
        "notes": (
            "Apple Silicon M4, no NVIDIA GPU. Docker daemon was down at discovery; "
            "Kafka was not started. Feast/Evidently/SHAP are optional extras."
        ),
    }
    path = Path("artifacts/environment.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def _mod(name: str) -> dict:
    try:
        mod = __import__(name)
        return {"installed": True, "version": getattr(mod, "__version__", "unknown")}
    except Exception as exc:
        return {"installed": False, "error": str(exc)}


def _mps() -> bool:
    try:
        import torch

        return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()
