"""Local MLflow registry helpers. Aliases: champion / challenger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def configure_mlflow(tracking_uri: str, experiment: str) -> None:
    Path("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)


def log_metrics_safe(metrics: dict[str, Any], prefix: str = "") -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and value == value and abs(value) != float("inf"):
            mlflow.log_metric(f"{prefix}{key}" if prefix else key, float(value))


def set_alias(model_name: str, alias: str, version: str | int) -> dict[str, Any]:
    client = MlflowClient()
    try:
        client.set_registered_model_alias(model_name, alias, str(version))
        return {"ok": True, "alias": alias, "version": str(version)}
    except Exception as exc:
        return {"ok": False, "alias": alias, "error": str(exc)}


def promotion_gate(
    candidate: dict[str, Any],
    champion: dict[str, Any] | None,
    min_pr_auc_delta: float = 0.0,
    max_fpr: float = 0.15,
) -> dict[str, Any]:
    """Promote only if the candidate beats gates."""
    reasons: list[str] = []
    promote = True
    if candidate.get("fpr", 1.0) > max_fpr:
        promote = False
        reasons.append(f"FPR {candidate.get('fpr')} exceeds max_fpr={max_fpr}")
    if champion is not None:
        delta = float(candidate.get("pr_auc") or 0) - float(champion.get("pr_auc") or 0)
        if delta < min_pr_auc_delta:
            promote = False
            reasons.append(f"PR-AUC delta {delta:.4f} < {min_pr_auc_delta}")
        cand_cost = float(candidate.get("expected_cost") or 0)
        champ_cost = float(champion.get("expected_cost") or 0)
        if cand_cost > champ_cost:
            # still allow if PR-AUC lift is large and FPR ok
            if delta < 0.02:
                promote = False
                reasons.append("higher expected cost without >=0.02 PR-AUC lift")
    if promote:
        reasons.append("gates passed")
    return {"promote": promote, "reasons": reasons}
