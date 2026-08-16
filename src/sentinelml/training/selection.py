from __future__ import annotations

from typing import Any


def invert_minmax(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return float(1.0 - (min(max(value, lo), hi) - lo) / (hi - lo))


def selection_score(row: dict[str, Any], weights: dict[str, float]) -> float:
    """PR-AUC + recall + FPR + cost + latency + memory. Not AUROC-only."""
    pr = float(row.get("pr_auc") or 0.0)
    recall = float(row.get("recall") or 0.0)
    fpr_inv = 1.0 - float(row.get("fpr") or 0.0)
    cost_inv = invert_minmax(
        float(row.get("expected_cost") or 0.0), 0.0, float(row.get("cost_scale") or 1.0)
    )
    lat_inv = invert_minmax(float(row.get("latency_ms_p50") or 0.0), 0.05, 50.0)
    mem_inv = invert_minmax(float(row.get("rss_mb") or 1.0), 1.0, 800.0)
    return (
        weights.get("pr_auc", 0.35) * pr
        + weights.get("recall", 0.20) * recall
        + weights.get("fpr_inverted", 0.15) * fpr_inv
        + weights.get("cost_inverted", 0.20) * cost_inv
        + weights.get("latency_inverted", 0.05) * lat_inv
        + weights.get("memory_inverted", 0.05) * mem_inv
    )


def pick_champion(rows: list[dict[str, Any]], weights: dict[str, float]) -> dict[str, Any]:
    costs = [float(r.get("expected_cost") or 0.0) for r in rows]
    scale = max(costs) if costs else 1.0
    scored = []
    for r in rows:
        item = dict(r)
        item["cost_scale"] = scale
        item["selection_score"] = selection_score(item, weights)
        scored.append(item)
    scored.sort(key=lambda x: x["selection_score"], reverse=True)
    champion = scored[0]
    champion["why"] = (
        f"Highest composite selection_score={champion['selection_score']:.4f} using "
        f"PR-AUC, recall, inverted FPR, inverted expected cost, latency, and memory. "
        "Not selected on AUROC alone. Isolation Forest is never auto-champion over a "
        "supervised model unless it wins this score."
    )
    return {"champion": champion, "ranked": scored}
