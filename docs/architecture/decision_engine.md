# Decision engine

The model outputs `fraud_probability`. A **separate** policy maps that score to an action:

| Probability band (defaults) | Risk | Action |
|---|---|---|
| ≤ `allow_max` (0.20) | low | ALLOW |
| ≤ `review_max` (0.70) | medium | REVIEW |
| > `review_max` | high | BLOCK |

If the score is within 0.02 of the locked binary threshold, or if the probability is NaN, the engine returns **REVIEW**.

These defaults are research settings, not an issuer policy. **Real blocking of payments needs legal and business validation** (false-positive customer harm, regulation, dispute cost). See `sentinelml.models.decision`.
