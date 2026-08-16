"""Feast-compatible OfflineStore / OnlineStore.

A real `feature_repo/` layout is included so Feast can be used when installed.
This module is the smoke-friendly implementation: Redis optional, in-memory
online store always available, parquet offline store.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from sentinelml.features.engineering import FEATURE_NAMES, FeatureState, vectorize

try:
    import redis as redis_lib
except Exception:  # pragma: no cover
    redis_lib = None  # type: ignore[assignment]


class OfflineStore(ABC):
    @abstractmethod
    def write_training(self, df: pd.DataFrame, name: str) -> Path: ...

    @abstractmethod
    def get_historical_features(self, name: str) -> pd.DataFrame: ...


class OnlineStore(ABC):
    @abstractmethod
    def upsert_features(self, entity_key: str, features: dict[str, float]) -> None: ...

    @abstractmethod
    def get_online_features(self, entity_key: str) -> dict[str, float] | None: ...

    @abstractmethod
    def ping(self) -> bool: ...


class ParquetOfflineStore(OfflineStore):
    def __init__(self, root: str | Path = "data/generated") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_training(self, df: pd.DataFrame, name: str) -> Path:
        path = self.root / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return path

    def get_historical_features(self, name: str) -> pd.DataFrame:
        path = self.root / f"{name}.parquet"
        return pd.read_parquet(path)


class InMemoryOnlineStore(OnlineStore):
    def __init__(self) -> None:
        self._data: dict[str, dict[str, float]] = {}

    def upsert_features(self, entity_key: str, features: dict[str, float]) -> None:
        self._data[entity_key] = dict(features)

    def get_online_features(self, entity_key: str) -> dict[str, float] | None:
        value = self._data.get(entity_key)
        return dict(value) if value is not None else None

    def ping(self) -> bool:
        return True

    def clear(self) -> None:
        self._data.clear()


class RedisOnlineStore(OnlineStore):
    def __init__(self, url: str = "redis://localhost:6379/2", prefix: str = "sml") -> None:
        if redis_lib is None:
            raise RuntimeError("redis package not installed")
        self.prefix = prefix
        self.client = redis_lib.Redis.from_url(
            url, decode_responses=True, socket_connect_timeout=0.4
        )

    def _key(self, entity_key: str) -> str:
        return f"{self.prefix}:feat:{entity_key}"

    def upsert_features(self, entity_key: str, features: dict[str, float]) -> None:
        self.client.set(self._key(entity_key), json.dumps(features))

    def get_online_features(self, entity_key: str) -> dict[str, float] | None:
        raw = self.client.get(self._key(entity_key))
        if raw is None:
            return None
        data = json.loads(raw)
        return {k: float(v) for k, v in data.items()}

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False


class FeatureStore:
    """Training retrieval (offline) + inference retrieval (online)."""

    def __init__(
        self,
        offline: OfflineStore | None = None,
        online: OnlineStore | None = None,
        feature_state: FeatureState | None = None,
        label_delay_days: int = 7,
    ) -> None:
        self.offline = offline or ParquetOfflineStore()
        self.online = online or InMemoryOnlineStore()
        self.state = feature_state or FeatureState(label_delay_days=label_delay_days)
        self.entities = ("customer_id", "merchant_id")
        self.feature_views = {
            "transaction_realtime": FEATURE_NAMES,
        }
        self.feature_service = "fraud_scoring_v1"

    def materialize_row(self, row: dict[str, Any], features: dict[str, float]) -> None:
        key = f"{row['customer_id']}|{row['merchant_id']}"
        self.online.upsert_features(key, features)
        self.online.upsert_features(f"customer:{row['customer_id']}", features)

    def get_online(self, customer_id: str, merchant_id: str) -> dict[str, float] | None:
        return self.online.get_online_features(f"{customer_id}|{merchant_id}")

    def online_vector(self, customer_id: str, merchant_id: str) -> list[float] | None:
        feats = self.get_online(customer_id, merchant_id)
        if feats is None:
            return None
        return vectorize(feats).tolist()


def build_online_store(
    backend: str = "memory", redis_url: str = "redis://localhost:6379/2"
) -> OnlineStore:
    if backend == "redis":
        try:
            store = RedisOnlineStore(url=redis_url)
            if store.ping():
                return store
        except Exception:
            pass
        return InMemoryOnlineStore()
    return InMemoryOnlineStore()
