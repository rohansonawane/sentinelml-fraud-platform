# Interview guide (this repository)

Walk the interviewer through **this code**, not generic fraud-ML talking points.

## 5 minutes

1. **Problem:** rare fraud, delayed labels, streaming features. Accuracy is the wrong metric.
2. **Leakage control:** one `FeatureState` class for train and serve; merchant fraud rate is delayed (`label_delay_days=7`). Temporal split only — hashes in `artifacts/split_manifest.json`.
3. **Decision ≠ model:** `sentinelml.models.decision` maps probability → ALLOW / REVIEW / BLOCK; ambiguous → REVIEW.
4. **Honest ops:** Kafka client exists; smoke uses `InMemoryBroker` because Docker was down. Numbers come from `artifacts/*.json`.

Show: `src/sentinelml/features/engineering.py` and `tests/unit/test_no_lookahead.py`.

## 15 minutes

5. Simulator scenarios (compromised card / merchant / amount) in `data/simulator.py`.
6. Model zoo actually trained: dummy, amount rule, logreg, LightGBM, HistGBDT, Torch MLP (sklearn MLP fallback), Isolation Forest (not claimed better than supervised).
7. Thresholds **locked on VAL** (`maximize_f1`, `maximize_recall_at_max_fpr`, `minimize_configured_cost` with FP=1, FN=25 **assumptions**).
8. Champion score = PR-AUC + recall + inverted FPR + inverted cost + latency + memory — not AUROC-only (`training/selection.py`).
9. Calibration: Platt/isotonic only if Brier improves without wrecking PR-AUC/cost.
10. API: `/api/v1/score` returns measured `latency_ms`, `model_version`, `decision`.
11. Drift: PSI/JS; `make drift-demo` must alert.

Show: `artifacts/metrics.json` champion row and a failing lookahead test if they ask “how do you know?”.

## 30 minutes

12. Online store: Redis if ping succeeds, else memory; training-serving skew test `offline_online_consistency`.
13. Promotion gate + MLflow aliases champion/challenger.
14. SQLite audit vs Postgres+Alembic on compose port **5433** (do not smash host :5432).
15. Explainability labeled as attribution, not causal proof (`models/explain.py`).
16. Load test: `scripts/benchmark.py` vs TestClient; do not quote 5000 TPS.
17. What you would change in production: real Kafka, Feast materialization, delayed labels from ops, cost model from finance, canary + shadow traffic.

If they ask for a number, open the JSON. If the JSON is missing, say **Not benchmarked on this hardware.**
