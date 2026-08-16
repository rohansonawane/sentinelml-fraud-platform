"""CPU vs CUDA. Never assume a GPU on the Apple Silicon smoke machine."""

from __future__ import annotations

import os


def requested_device(cfg: dict | None = None) -> str:
    env = os.environ.get("SENTINELML_DEVICE", "").strip().lower()
    if env:
        return env
    if cfg:
        return str(cfg.get("device", "auto")).strip().lower() or "auto"
    return "auto"


def cuda_is_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_torch_device(requested: str = "auto") -> str:
    """Return 'cuda' or 'cpu'. Does not auto-select MPS (smoke stays CPU)."""
    req = (requested or "auto").lower()
    if req in {"cpu", "mps"}:
        return "cpu"
    if req in {"cuda", "gpu"}:
        if not cuda_is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. See GPU_SETUP.md."
            )
        return "cuda"
    if req in {"auto", "", "default"}:
        return "cuda" if cuda_is_available() else "cpu"
    raise RuntimeError(f"Unknown SENTINELML_DEVICE={requested}")


def resolve_xgboost_device(requested: str = "auto") -> str:
    """XGBoost 2.x: device='cuda' | 'cpu' with tree_method='hist'."""
    req = (requested or "auto").lower()
    if req in {"cpu", "mps"}:
        return "cpu"
    if req in {"cuda", "gpu"}:
        return "cuda"
    return "cuda" if cuda_is_available() else "cpu"
