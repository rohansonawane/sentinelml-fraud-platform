# Contributing

1. Use Python 3.11 via `uv`. Do not commit `.venv`, `node_modules`, `.env`, raw dumps, or large checkpoints.
2. `make lint` and `make test` before a PR. `make smoke` on your machine if you change training.
3. Metrics in docs must come from `artifacts/*.json`. If you did not measure it, write `Not benchmarked on this hardware.`
4. Feature changes must keep the no-lookahead tests green (`tests/unit/test_no_lookahead.py`, `test_feature_consistency.py`).
5. Do not add a random split. Do not optimize accuracy as the primary metric.
6. Kafka/Feast extras are optional. Keep the in-process broker working.

MIT license. See `ARCHITECTURE.md` for the intended layout.
