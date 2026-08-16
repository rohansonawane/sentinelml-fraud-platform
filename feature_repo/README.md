# SentinelML feature repository

Layout matches Feast (`feature_store.yaml`, entities, feature views, feature service).

**Smoke path:** `sentinelml.features.store.FeatureStore` implements OfflineStore
(parquet) and OnlineStore (Redis if reachable, else in-memory). Training retrieval
and online inference retrieval share `FeatureState`.

**When Feast is installed:** `cd feature_repo && feast apply`.

**Training-serving skew:** `tests/unit/test_feature_consistency.py` and
`sentinelml.features.consistency.offline_online_consistency`.

Kafka / Feast were not required to be live for `make smoke`.
