# CUDA / GPU configs (not executed on this machine)

This MacBook Pro M4 has no NVIDIA GPU. Files here document a production GPU
profile for a different host. Do **not** treat numbers in comments as measured
results from this repository's smoke run.

Suggested GPU training (unexecuted):

- CUDA 12.x PyTorch wheels
- Larger LightGBM `n_estimators`
- MLP hidden `[128, 64]` and more epochs
- Optuna trials >= 40

Any README metric must still come from `artifacts/*.json` produced on the
machine that actually ran the command.
