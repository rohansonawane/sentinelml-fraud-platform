from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return sha256_text(blob)


def dataframe_hash(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    subset = df if columns is None else df[columns]
    payload = pd.util.hash_pandas_object(subset, index=True).values.tobytes()
    return sha256_bytes(payload)
