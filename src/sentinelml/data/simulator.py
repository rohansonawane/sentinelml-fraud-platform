"""Fraud Detection Handbook-style transaction simulator.

Customers have geographic locations and spending profiles. Merchants (terminals)
are placed on the same plane. Legitimate transactions occur near a customer's
home location. Three fraud scenarios are injected:

1. Compromised merchant: a burst of fraudulent txs at one terminal.
2. Compromised card: unusual amounts at distant merchants.
3. Prevalence shift-ready flags stored on merchants for drift demos.

All timestamps are generated in chronological order. Labels are known at
generation time for training, but feature engineering applies a label delay.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


@dataclass
class SimulatorConfig:
    n_customers: int = 500
    n_merchants: int = 100
    n_transactions: int = 8000
    start: datetime = datetime(2025, 1, 1, tzinfo=timezone.utc)
    n_days: int = 90
    seed: int = 42
    base_fraud_rate: float = 0.025
    compromised_merchant_frac: float = 0.04
    compromised_card_frac: float = 0.03


def _ids(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i:05d}" for i in range(n)]


def generate_customers(cfg: SimulatorConfig, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": _ids("C", cfg.n_customers),
            "x": rng.uniform(0, 100, cfg.n_customers),
            "y": rng.uniform(0, 100, cfg.n_customers),
            "mean_amount": rng.uniform(20, 180, cfg.n_customers),
            "std_amount": rng.uniform(5, 40, cfg.n_customers),
            "mean_nb_tx_per_day": rng.uniform(0.4, 3.5, cfg.n_customers),
        }
    )


def generate_merchants(cfg: SimulatorConfig, rng: np.random.Generator) -> pd.DataFrame:
    n_compromised = max(1, int(cfg.n_merchants * cfg.compromised_merchant_frac))
    compromised = np.zeros(cfg.n_merchants, dtype=bool)
    compromised[rng.choice(cfg.n_merchants, n_compromised, replace=False)] = True
    return pd.DataFrame(
        {
            "merchant_id": _ids("M", cfg.n_merchants),
            "x": rng.uniform(0, 100, cfg.n_merchants),
            "y": rng.uniform(0, 100, cfg.n_merchants),
            "category": rng.choice(
                ["grocery", "gas", "online", "restaurant", "travel", "other"],
                cfg.n_merchants,
            ),
            "is_compromised_scenario": compromised,
        }
    )


def _nearest_merchants(
    customers: pd.DataFrame, merchants: pd.DataFrame, k: int = 8
) -> dict[str, list[str]]:
    mx = merchants["x"].to_numpy()
    my = merchants["y"].to_numpy()
    mids = merchants["merchant_id"].to_numpy()
    mapping: dict[str, list[str]] = {}
    for row in customers.itertuples(index=False):
        dist = (mx - row.x) ** 2 + (my - row.y) ** 2
        idx = np.argpartition(dist, min(k, len(dist) - 1))[:k]
        mapping[row.customer_id] = [str(mids[i]) for i in idx]
    return mapping


def generate_transactions(cfg: SimulatorConfig | None = None) -> pd.DataFrame:
    cfg = cfg or SimulatorConfig()
    rng = np.random.default_rng(cfg.seed)
    customers = generate_customers(cfg, rng)
    merchants = generate_merchants(cfg, rng)
    nearby = _nearest_merchants(customers, merchants)

    compromised_cards = set(
        rng.choice(
            customers["customer_id"].to_numpy(),
            size=max(1, int(cfg.n_customers * cfg.compromised_card_frac)),
            replace=False,
        ).tolist()
    )
    compromised_merchants = set(
        merchants.loc[merchants["is_compromised_scenario"], "merchant_id"].tolist()
    )

    customer_rows = customers.set_index("customer_id")
    merchant_ids = merchants["merchant_id"].to_numpy()

    records: list[dict[str, object]] = []
    weights = customer_rows["mean_nb_tx_per_day"].to_numpy()
    weights = weights / weights.sum()
    chosen = rng.choice(customers["customer_id"].to_numpy(), size=cfg.n_transactions, p=weights)

    seconds_span = cfg.n_days * 24 * 3600
    raw_offsets = rng.integers(0, seconds_span, size=cfg.n_transactions)

    for tx_idx, (cid, offset) in enumerate(zip(chosen, raw_offsets, strict=True)):
        crow = customer_rows.loc[cid]
        event_time = cfg.start + timedelta(seconds=int(offset))
        # Bias toward 8:00-22:00 local-ish by wrapping hour if in deep night for legit txs.
        hour = event_time.hour
        is_night = hour < 6 or hour >= 23
        amount = float(max(1.0, rng.normal(crow.mean_amount, crow.std_amount)))

        is_fraud = 0
        scenario = "legit"
        if cid in compromised_cards and rng.random() < 0.35:
            is_fraud = 1
            scenario = "compromised_card"
            amount = float(amount * rng.uniform(3.0, 9.0))
            merchant_id = str(rng.choice(merchant_ids))
        elif rng.random() < cfg.base_fraud_rate * 0.4:
            is_fraud = 1
            scenario = "amount_anomaly"
            amount = float(amount * rng.uniform(4.0, 12.0))
            merchant_id = str(rng.choice(nearby[cid]))
        else:
            merchant_id = str(rng.choice(nearby[cid]))
            if merchant_id in compromised_merchants and rng.random() < 0.25:
                is_fraud = 1
                scenario = "compromised_merchant"

        if is_night and is_fraud == 0 and rng.random() < 0.7:
            # shift legit night txs into daytime
            event_time = event_time.replace(hour=int(rng.integers(9, 21)))

        records.append(
            {
                "transaction_id": f"T{tx_idx:08d}",
                "event_id": f"E{tx_idx:08d}",
                "event_time": event_time,
                "customer_id": cid,
                "merchant_id": merchant_id,
                "amount": round(amount, 2),
                "currency": "USD",
                "channel": str(
                    rng.choice(["card_present", "card_not_present", "online"], p=[0.6, 0.25, 0.15])
                ),
                "is_fraud": is_fraud,
                "fraud_scenario": scenario,
                "schema_version": "1.0",
            }
        )
        # transaction_id rewritten after chronological sort

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("event_time").reset_index(drop=True)
    df["transaction_id"] = [f"T{i:08d}" for i in range(len(df))]
    df["event_id"] = [f"E{i:08d}" for i in range(len(df))]
    return df


def attach_profiles(
    transactions: pd.DataFrame, cfg: SimulatorConfig, rng: np.random.Generator | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = rng or np.random.default_rng(cfg.seed)
    customers = generate_customers(cfg, rng)
    merchants = generate_merchants(cfg, rng)
    return customers, merchants
