from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = Field(default="local", alias="SENTINELML_ENV")
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8102
    model_dir: str = "artifacts"
    stream_backend: str = "memory"
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_raw: str = "transactions.raw"
    kafka_topic_enriched: str = "transactions.enriched"
    kafka_topic_predictions: str = "fraud.predictions"
    kafka_topic_alerts: str = "fraud.alerts"
    kafka_topic_feedback: str = "model.feedback"
    online_store_backend: str = "memory"
    redis_url: str = "redis://localhost:6379/2"
    feature_label_delay_days: int = 7
    audit_database_url: str = "sqlite:///artifacts/audit.db"
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "sentinelml"
    decision_allow_max: float = 0.20
    decision_review_max: float = 0.70
    cost_false_positive: float = 1.0
    cost_false_negative: float = 25.0
    max_batch_size: int = 256
    max_body_bytes: int = 1_048_576


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def get_settings() -> Settings:
    return Settings()
