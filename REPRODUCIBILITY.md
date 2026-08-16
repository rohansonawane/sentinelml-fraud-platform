# Reproducibility

## Environment

- Python **3.11** (`uv python pin 3.11`), Apple Silicon M4, 16 GB RAM, no NVIDIA GPU.
- Exact package set: `uv.lock` after `uv sync --extra dev`.
- Hardware snapshot: `python scripts/collect_environment.py` → `artifacts/environment.json`.

## Commands

```bash
uv python pin 3.11
uv sync --extra dev
make data          # or make smoke
make train
make evaluate
make stream-demo   # in-process broker
make drift-demo
make test
```

Seeds: simulator `--seed 42`, Optuna TPESampler seed 42, IsolationForest `random_state=42`.

## Splits and hashes

`artifacts/split_manifest.json` records `train_start`, `train_end`, `val_end`, `test_end`, row counts, fraud rates, and SHA-256 hashes of each partition.

## What we did not reproduce here

- Live Kafka (Docker daemon down at discovery; compose file is present).
- `feast apply` (Feast extra optional; `feature_repo/` layout is real).
- CUDA training (`configs/cuda/` is documentation only).
- OpenML download (optional secondary dataset).

## Artifacts that must exist for a claims-bearing README

`metrics.json`, `champion.json`, `split_manifest.json`, `model_comparison.csv`, `environment.json`. Stream/drift/load numbers only if `stream_demo.json` / `drift_report.json` / `benchmark.json` exist.
