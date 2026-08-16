# Cloud profile (not executed)

Do not create paid GPU instances from this repository.

If you later run on a cloud VM:

1. Point `MLFLOW_TRACKING_URI` at a remote tracking server.
2. Use Redis as the online store and Kafka as the broker.
3. Replace SQLite with PostgreSQL via `deploy/kubernetes` and Alembic.
4. Re-run `scripts/benchmark.py` and overwrite artifacts. Do not copy MacBook
   smoke numbers into a cloud report.
