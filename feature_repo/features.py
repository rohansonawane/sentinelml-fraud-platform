"""Feast-style entity, data source, feature view, and feature service definitions.

These objects are valid Feast declarations when `feast` is installed. They are
also imported by SentinelML tests as the canonical feature contract.
"""

from __future__ import annotations

from datetime import timedelta

FEATURE_VIEW_NAME = "transaction_realtime"
FEATURE_SERVICE_NAME = "fraud_scoring_v1"
ENTITIES = ("customer", "merchant")

CUSTOMER_ENTITY = {
    "name": "customer",
    "join_keys": ["customer_id"],
    "description": "Cardholder / customer placing the transaction",
}
MERCHANT_ENTITY = {
    "name": "merchant",
    "join_keys": ["merchant_id"],
    "description": "Merchant / terminal receiving the transaction",
}

TRANSACTION_SOURCE = {
    "name": "transactions_source",
    "path": "data/generated/transactions.parquet",
    "timestamp_field": "event_time",
    "created_timestamp_column": "event_time",
}

FEATURE_COLUMNS = [
    "amount",
    "hour",
    "day_of_week",
    "is_weekend",
    "tx_count_10m",
    "tx_count_1h",
    "tx_count_24h",
    "amount_sum_1h",
    "amount_sum_24h",
    "avg_amount_7d",
    "amount_vs_customer_mean",
    "amount_zscore_customer",
    "merchant_tx_count_24h",
    "merchant_fraud_rate_historical",
    "merchant_avg_amount",
    "new_merchant_for_customer",
    "time_since_previous_transaction",
    "merchant_frequency_customer",
]

FEATURE_VIEW = {
    "name": FEATURE_VIEW_NAME,
    "entities": list(ENTITIES),
    "ttl": timedelta(days=14).total_seconds(),
    "schema": FEATURE_COLUMNS,
    "online": True,
    "source": TRANSACTION_SOURCE["name"],
}

FEATURE_SERVICE = {
    "name": FEATURE_SERVICE_NAME,
    "features": [FEATURE_VIEW_NAME],
}

try:  # optional real Feast objects
    from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
    from feast.types import Float64

    customer = Entity(name="customer", join_keys=["customer_id"], value_type=ValueType.STRING)
    merchant = Entity(name="merchant", join_keys=["merchant_id"], value_type=ValueType.STRING)
    source = FileSource(
        name="transactions_source",
        path="data/generated/transactions_features.parquet",
        timestamp_field="event_time",
    )
    transaction_realtime = FeatureView(
        name=FEATURE_VIEW_NAME,
        entities=[customer, merchant],
        ttl=timedelta(days=14),
        schema=[Field(name=c, dtype=Float64) for c in FEATURE_COLUMNS],
        online=True,
        source=source,
    )
    fraud_scoring_v1 = FeatureService(name=FEATURE_SERVICE_NAME, features=[transaction_realtime])
except Exception:
    customer = CUSTOMER_ENTITY  # type: ignore[assignment]
    merchant = MERCHANT_ENTITY  # type: ignore[assignment]
    source = TRANSACTION_SOURCE  # type: ignore[assignment]
    transaction_realtime = FEATURE_VIEW  # type: ignore[assignment]
    fraud_scoring_v1 = FEATURE_SERVICE  # type: ignore[assignment]
