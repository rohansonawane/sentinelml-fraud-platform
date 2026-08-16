"""Temporal train/val/test split. No random row shuffle. No look-ahead."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from sentinelml.utils.hashing import dataframe_hash, sha256_json


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end: datetime
    val_end: datetime
    test_end: datetime
    train_start: datetime
    hashes: dict[str, str]


def temporal_split(
    df: pd.DataFrame,
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
    time_col: str = "event_time",
) -> TemporalSplit:
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("fractions must be in (0, 1)")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train+val must leave a test remainder")

    ordered = df.sort_values(time_col).reset_index(drop=True).copy()
    ordered[time_col] = pd.to_datetime(ordered[time_col], utc=True)
    t0 = ordered[time_col].min()
    t1 = ordered[time_col].max()
    span = t1 - t0
    train_end = t0 + span * train_fraction
    val_end = t0 + span * (train_fraction + val_fraction)

    train = ordered.loc[ordered[time_col] <= train_end].copy()
    val = ordered.loc[(ordered[time_col] > train_end) & (ordered[time_col] <= val_end)].copy()
    test = ordered.loc[ordered[time_col] > val_end].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("temporal split produced an empty partition")

    hashes = {
        "train": dataframe_hash(train, ["transaction_id", "event_time", "amount", "is_fraud"]),
        "val": dataframe_hash(val, ["transaction_id", "event_time", "amount", "is_fraud"]),
        "test": dataframe_hash(test, ["transaction_id", "event_time", "amount", "is_fraud"]),
        "full": dataframe_hash(ordered, ["transaction_id", "event_time", "amount", "is_fraud"]),
        "boundaries": sha256_json(
            {
                "train_start": str(t0),
                "train_end": str(train_end),
                "val_end": str(val_end),
                "test_end": str(t1),
            }
        ),
    }
    return TemporalSplit(
        train=train,
        val=val,
        test=test,
        train_start=t0.to_pydatetime(),
        train_end=train_end.to_pydatetime(),
        val_end=val_end.to_pydatetime(),
        test_end=t1.to_pydatetime(),
        hashes=hashes,
    )


def split_manifest(split: TemporalSplit) -> dict[str, object]:
    return {
        "method": "temporal",
        "lookahead": False,
        "train_start": split.train_start.isoformat(),
        "train_end": split.train_end.isoformat(),
        "val_end": split.val_end.isoformat(),
        "test_end": split.test_end.isoformat(),
        "n_train": int(len(split.train)),
        "n_val": int(len(split.val)),
        "n_test": int(len(split.test)),
        "fraud_rate_train": float(split.train["is_fraud"].mean()),
        "fraud_rate_val": float(split.val["is_fraud"].mean()),
        "fraud_rate_test": float(split.test["is_fraud"].mean()),
        "hashes": split.hashes,
        "notes": (
            "TRAIN is earlier, VAL later, TEST latest. Rolling features use only "
            "information strictly before the current transaction. Merchant fraud "
            "rate uses delayed labels."
        ),
    }
