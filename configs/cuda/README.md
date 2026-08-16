# CUDA / GPU configs

Files in this folder are for a **Linux NVIDIA workstation**. They were **not** executed on the Apple Silicon smoke machine.

| Config | Rows (simulator) | Intent |
|---|---|---|
| `large.yaml` | ~200k | First real GPU run (`make gpu-train`) |
| `full.yaml` | ~1M | Longer run (`make gpu-train GPU_CONFIG=configs/cuda/full.yaml`) |

Playbook: [GPU_SETUP.md](../../GPU_SETUP.md).

Any number quoted as a GPU result must come from `artifacts/gpu/metrics.json` on the machine that ran the command.
