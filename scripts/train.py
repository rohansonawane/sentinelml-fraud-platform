#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from sentinelml.training.pipeline import train_from_config
from sentinelml.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--n-optuna-trials", type=int, default=None)
    args = parser.parse_args()
    configure_logging()
    result = train_from_config(args.config, n_optuna_trials=args.n_optuna_trials)
    champ = result["champion"]
    print(
        f"champion={champ['model']} test_pr_auc={champ['pr_auc']:.4f} "
        f"recall={champ['recall']:.4f} fpr={champ['fpr']:.4f} "
        f"cost={champ['expected_cost']:.1f}"
    )


if __name__ == "__main__":
    main()
