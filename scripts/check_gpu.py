#!/usr/bin/env python3
"""Refuse GPU training on Apple Silicon; require CUDA on the target box."""

from __future__ import annotations

import platform
import shutil
import sys


def main() -> int:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print(
            "This Mac is the smoke machine (no NVIDIA GPU).\n"
            "Clone the repo on the Linux NVIDIA box and run:\n"
            "  make gpu-setup && make gpu-check && make gpu-data && make gpu-train\n"
            "See GPU_SETUP.md.",
            file=sys.stderr,
        )
        return 1
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found. Install NVIDIA drivers/CUDA.", file=sys.stderr)
        return 2
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Run make gpu-setup.", file=sys.stderr)
        return 3
    print(f"torch {torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print(
            "torch.cuda.is_available() is False. Install a CUDA wheel:\n"
            "  uv pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "See GPU_SETUP.md.",
            file=sys.stderr,
        )
        return 4
    print(f"device_count={torch.cuda.device_count()}")
    print(f"name={torch.cuda.get_device_name(0)}")
    try:
        import xgboost as xgb

        print(f"xgboost {xgb.__version__}")
    except Exception as exc:
        print(f"xgboost import failed: {exc}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
