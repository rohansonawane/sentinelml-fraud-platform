# GPU setup — train SentinelML on a real NVIDIA box

This MacBook trained **6,000 simulated transactions**. The NVIDIA machine trains **hundreds of thousands to a million** Handbook-style simulated payments with GPU XGBoost and a larger PyTorch MLP.

There is no public “real bank” dump in this repo. Scaling the simulator (and optional OpenML credit-card file) is the honest large run. GPU metrics go in `artifacts/gpu/` and `docs/GPU_MODEL_COMPARISON.md`. Do **not** paste them into the Mac table in README.md.

## 1. Clone

```bash
git clone https://github.com/rohansonawane/sentinelml-fraud-platform.git
cd sentinelml-fraud-platform
```

Need: Ubuntu (or similar), NVIDIA driver, CUDA matching the PyTorch wheel, Python 3.11, [uv](https://docs.astral.sh/uv/).

## 2. Python env + CUDA

```bash
make gpu-setup
make gpu-check
```

If `gpu-check` says `torch.cuda.is_available() is False`:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
make gpu-check
```

Pick the index for **your** CUDA (`cu118`, `cu121`, `cu124`, `cu126`): https://pytorch.org/get-started/locally/

XGBoost 2.x uses `tree_method=hist` and `device=cuda`. Pip LightGBM is usually CPU; it still trains on the GPU box, just on CPU cores.

## 3. Data (synthetic, large)

```bash
# ~200k transactions (recommended first GPU run)
make gpu-data

# ~1M transactions (hours)
make gpu-data-full
```

Optional secondary set (PCA features, different schema — not the default trainer):

```bash
uv run python scripts/download_data.py   # prints license notes
```

## 4. Train / evaluate / serve

```bash
make gpu-train                          # configs/cuda/large.yaml
make gpu-eval
make gpu-serve                          # API :8102, loads artifacts/gpu/champion.joblib
make ui                                 # dashboard :3102
```

Million-row config:

```bash
make gpu-train GPU_CONFIG=configs/cuda/full.yaml
```

`scripts/check_gpu.py` refuses to start on Apple Silicon.

## What “real” means here

| | Mac smoke | This GPU path |
|---|---|---|
| Rows | 6,000 | 200,000 (large) or 1,000,000 (full) |
| XGBoost | CPU hist, 80 trees | CUDA hist, 500–800 trees |
| MLP | 32-16, 12 epochs, CPU | 256-128-64 (or larger), CUDA AMP |
| Optuna | 4–8 trials, 90s | 40–80 trials, 30–120 min |
| Metrics file | `artifacts/` | `artifacts/gpu/` |

Still synthetic. PR-AUC on this generator is **not** a bank portfolio SLA.
