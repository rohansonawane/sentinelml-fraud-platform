"""Decision engine SEPARATE from the model.

low  -> ALLOW
mid  -> REVIEW
high -> BLOCK (documented: real blocking needs legal/business validation)
Default REVIEW when ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecisionPolicy:
    allow_max: float = 0.20
    review_max: float = 0.70
    default_ambiguous: str = "REVIEW"
    high_action: str = "BLOCK"

    def __post_init__(self) -> None:
        if not 0.0 <= self.allow_max <= self.review_max <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= allow_max <= review_max <= 1")


def risk_level(probability: float, policy: DecisionPolicy) -> str:
    if probability < 0 or probability > 1:
        return "unknown"
    if probability <= policy.allow_max:
        return "low"
    if probability <= policy.review_max:
        return "medium"
    return "high"


def decide(
    probability: float,
    policy: DecisionPolicy,
    threshold: float | None = None,
) -> str:
    """Map probability to ALLOW / REVIEW / BLOCK.

    `threshold` is the binary operating point locked on validation. When the
    score is within a small band of the threshold, default to REVIEW.
    """
    if probability != probability:  # NaN
        return policy.default_ambiguous
    if threshold is not None and abs(probability - threshold) < 0.02:
        return policy.default_ambiguous
    level = risk_level(probability, policy)
    if level == "low":
        return "ALLOW"
    if level == "medium":
        return "REVIEW"
    return policy.high_action


def apply_binary_threshold(probability: float, threshold: float) -> int:
    return int(probability >= threshold)
