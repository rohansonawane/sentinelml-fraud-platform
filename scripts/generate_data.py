#!/usr/bin/env python3
"""Generate Fraud Detection Handbook-style synthetic transactions."""

from __future__ import annotations

import argparse
from pathlib import Path

from sentinelml.data.simulator import SimulatorConfig, generate_transactions
from sentinelml.utils.logging import configure_logging, get_logger

log = get_logger("generate_data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-transactions", type=int, default=8000)
    parser.add_argument("--n-customers", type=int, default=500)
    parser.add_argument("--n-merchants", type=int, default=100)
    parser.add_argument("--n-days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/generated/transactions.parquet")
    args = parser.parse_args()
    configure_logging()
    cfg = SimulatorConfig(
        n_transactions=args.n_transactions,
        n_customers=args.n_customers,
        n_merchants=args.n_merchants,
        n_days=args.n_days,
        seed=args.seed,
    )
    df = generate_transactions(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)
    log.info(
        "wrote_transactions",
        path=str(out),
        n=len(df),
        fraud_rate=float(df["is_fraud"].mean()),
        start=str(df["event_time"].min()),
        end=str(df["event_time"].max()),
    )
    print(f"wrote {len(df)} rows to {out} fraud_rate={df['is_fraud'].mean():.4f}")


if __name__ == "__main__":
    main()
