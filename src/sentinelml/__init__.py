"""SentinelML: real-time fraud detection and MLOps."""

from __future__ import annotations

import os

# Torch and LightGBM both ship OpenMP. On macOS that combination SIGSEGVs
# (__kmp_unregister_library) unless the duplicate runtime is allowed and
# thread counts stay at 1. Set these before any native library import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
