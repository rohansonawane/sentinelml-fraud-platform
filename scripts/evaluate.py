#!/usr/bin/env python3
"""Re-evaluate the saved champion on the locked test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinelml.inference.service import InferenceService
from sentinelml.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    artifacts = Path(cfg.get("artifacts_dir", "artifacts"))
    metrics_path = artifacts / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit("Run scripts/train.py first")
    metrics = json.loads(metrics_path.read_text())
    champ = metrics["champion"]
    print(
        json.dumps(
            {
                "model": champ["model"],
                "test": {
                    k: champ["test"][k]
                    for k in (
                        "pr_auc",
                        "roc_auc",
                        "precision",
                        "recall",
                        "f1",
                        "fpr",
                        "expected_cost",
                        "brier",
                        "confusion_matrix",
                    )
                },
                "threshold": champ["operating_threshold"],
                "calibration": champ["calibration"]["method"],
            },
            indent=2,
        )
    )
    svc = InferenceService(bundle_path=artifacts / "champion.joblib")
    if not svc.ready:
        raise SystemExit("champion bundle failed to load")
    print(f"bundle_ok model={svc.model_name} threshold={svc.threshold}")


if __name__ == "__main__":
    main()
