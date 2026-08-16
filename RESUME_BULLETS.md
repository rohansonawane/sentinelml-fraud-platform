# Resume bullets

Figures below are from `artifacts/*.json` on an Apple M4 (16 GB, no NVIDIA GPU). Do not reuse on another host.

- Built **SentinelML**, a real-time fraud research platform: chronological `FeatureState` (no lookahead, 7-day delayed merchant fraud rate), temporal TRAIN/VAL/TEST (3608/1241/1151 txs), FastAPI scoring, in-process stream processor with a Kafka-shaped API, and a Next.js monitoring UI.
- Trained dummy, logistic, LightGBM, HistGBDT, XGBoost, PyTorch MLP, sklearn MLP, and Isolation Forest on a Handbook-style simulator; **XGBoost champion** with test **PR-AUC 0.808**, **recall 0.842**, **FPR 0.022**, expected cost 175 (FP=1, FN=25 assumptions), selected on a composite score not AUROC-only (LightGBM had higher PR-AUC 0.818 but lost on cost/FPR).
- Locked the operating threshold on validation (`minimize_configured_cost` → 0.265) before touching test; calibration skipped after it failed to improve Brier without hurting business metrics.
- Stream demo: **500/500** events processed at **99.97 events/s** (target 100; Kafka not started). Drift demo triggered PSI/JS alerts. TestClient load: **319 rps**, p95 **4.2 ms**, **0% errors** (200 requests). **34/34** pytest passing.
