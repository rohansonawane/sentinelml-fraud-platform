#!/usr/bin/env python3
"""Optional OpenML credit-card fraud download. Not required for smoke.

The primary training path uses the Handbook-style simulator. This script
documents a secondary public dataset and refuses to run unless explicitly
requested, so CI never pulls a large file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

OPENML_CREDITCARD = "https://www.openml.org/d/1597"
NOTE = """
Secondary dataset (NOT used by make smoke):

- OpenML Credit Card Fraud (dataset 1597 / Kaggle creditcard.csv analogue)
- Highly imbalanced, PCA-anonymized features (V1..V28), Amount, Time, Class
- License/terms: check OpenML + original ULB/Kaggle terms before redistribution

This repository trains on the synthetic simulator so results are reproducible
without a network download. To fetch OpenML data you would typically:

    pip install openml
    python -c "import openml; openml.datasets.get_dataset(1597)"

Do not commit the downloaded CSV.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Document optional OpenML download")
    parser.add_argument("--fetch", action="store_true", help="attempt download (network)")
    parser.add_argument("--out", default="data/raw/openml_creditcard.parquet")
    args = parser.parse_args()
    print(NOTE)
    print(f"OpenML URL: {OPENML_CREDITCARD}")
    if not args.fetch:
        print("Skipping fetch (pass --fetch to download). Smoke path does not need this.")
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        (Path("data/raw") / "README.md").write_text(NOTE)
        return
    try:
        import openml  # type: ignore

        ds = openml.datasets.get_dataset(1597)
        x, y, *_ = ds.get_data(target=ds.default_target_attribute)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        x.assign(Class=y).to_parquet(out, index=False)
        print(f"wrote {out}")
    except Exception as exc:
        print(f"BLOCKER: OpenML download failed ({exc}). Simulator data is sufficient.")


if __name__ == "__main__":
    main()
