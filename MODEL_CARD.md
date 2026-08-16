# Model card — SentinelML fraud ranker

## Model details

- **Champion (this host):** XGBoost 0.1.0, threshold 0.2648 locked on validation with `minimize_configured_cost`. Calibration: none. See `artifacts/champion.json`.
- **Challenger:** PyTorch MLP. Aliases in `artifacts/registry.json`.
- **Also trained:** dummy majority, amount rule, logistic regression, LightGBM (Optuna 4 trials, VAL PR-AUC 0.8473), sklearn HistGradientBoosting, sklearn MLP, Isolation Forest (anomaly; not claimed superior).
- **Features:** 18 fields in `sentinelml.features.engineering.FEATURE_NAMES`. Merchant fraud rate uses a 7-day label delay.
- **Why this champion:** composite selection_score 0.8665 (PR-AUC, recall, inverted FPR, inverted expected cost, latency, memory) — not AUROC-only. LightGBM’s test PR-AUC was slightly higher (0.8185 vs 0.8083) but XGBoost won on cost/FPR/latency.

## Metrics (test split, from `artifacts/metrics.json`)

| PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Brier | Cost |
|---|---|---|---|---|---|---|---|
| 0.8083 | 0.9216 | 0.5614 | 0.8421 | 0.6737 | 0.0225 | 0.0174 | 175.0 |

Confusion: TN 1088, FP 25, FN 6, TP 32. Cost units are modeling assumptions (FP=1, FN=25), not bank losses.

## Intended use

Research / portfolio demonstration. **Not** a production issuer detector.

## Ethical / operational notes

- Synthetic labels; fraud evolves.
- BLOCK can harm legitimate customers. Legal/business validation required.
- Attribution is feature contribution, **not causal proof**. SHAP was not installed on this host.
- Isolation Forest comparison only.

## Training data

See `DATASET_CARD.md` and `artifacts/split_manifest.json`.
